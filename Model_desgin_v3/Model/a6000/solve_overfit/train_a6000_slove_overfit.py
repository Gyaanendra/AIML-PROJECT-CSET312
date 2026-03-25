# train.py
# Dual-signal fake news classifier
# Architecture: RoBERTa-base (full fine-tune) + GPT-2 statistical features → fused linear head

# GPU  : RTX A6000 (48 GB) — configured for 32 GB max utilisation
# torch: 2.7.1+cu118 | transformers: 5.3.0 | peft: 0.18.1
#
# pip install peft transformers datasets accelerate scikit-learn tqdm pandas nltk

import os
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ═══════════════════════════════════════════════════════════════════
# CONFIG  ← ALL tunable hyperparameters live here
# ═══════════════════════════════════════════════════════════════════

# ── Paths & model names ────────────────────────────────────────────
DATA_PATH  = "dataset_cleaned.csv"
PHASE1_OUT = "./phase1_roberta_fulltune"
PHASE2_OUT = "./phase2_fusion_head"
MODEL_NAME = "roberta-base"
GPT2_NAME  = "gpt2"

# ── Sequence & seed ───────────────────────────────────────────────
MAX_LENGTH = 512
SEED       = 42

# ══════════════════════════════════════════════════════════════════
# PHASE 1 — RoBERTa full fine-tune
# ══════════════════════════════════════════════════════════════════

# ── Batch sizes ────────────────────────────────────────────────────
# OOM during Phase 1 training?  Lower PHASE1_TRAIN_BATCH first (try 32 → 16),
# then double PHASE1_GRAD_ACCUM to keep the effective batch the same.
PHASE1_TRAIN_BATCH  = 64      # A6000 48 GB — safe at 64.  OOM? 32 → 16
PHASE1_EVAL_BATCH   = 128     # eval only (no activations stored).  OOM? 64 → 32
PHASE1_GRAD_ACCUM   = 1       # effective batch = TRAIN_BATCH × GRAD_ACCUM
                               # halve TRAIN_BATCH → double this to compensate

# ── CLS embedding extraction batch (Phase 1 → Phase 2 handoff) ───
# Inference-only; no gradients kept. OOM here usually means VRAM is
# still partially occupied — free it or reduce this value.
CLS_EXTRACT_BATCH   = 128     # OOM? 64 → 32

# ── Regularisation ────────────────────────────────────────────────
PHASE1_LR              = 1e-5   # ↓ was 2e-5 — slower convergence → less memorisation
PHASE1_WARMUP_RATIO    = 0.10   # fraction of total steps for warmup (was fixed 500)
PHASE1_WEIGHT_DECAY    = 0.05   # ↑ was 0.01 — stronger L2 for 125 M-param model
PHASE1_LABEL_SMOOTHING = 0.10   # prevents overconfident logits; set 0.0 to disable
PHASE1_HEAD_DROPOUT    = 0.25   # dropout before RoBERTa classifier head (was 0.1 in HF)
PHASE1_MAX_GRAD_NORM   = 1.0    # gradient clipping

# ── LLRD — layer-wise LR decay ────────────────────────────────────
LLRD_DECAY             = 0.85   # LR_layer = PHASE1_LR × LLRD_DECAY^depth
                                 # set 1.0 to disable (uniform LR for all layers)

# ── Schedule ──────────────────────────────────────────────────────
PHASE1_EPOCHS          = 8
PHASE1_PATIENCE        = 3      # early-stop: epochs without val_f1 improvement
PHASE1_PATIENCE_THRESH = 0.001  # minimum Δ to count as improvement

# ── Step-level val logging ─────────────────────────────────────────
# Every PHASE1_VAL_LOG_STEPS training steps, run a full val pass and
# write val_loss to step_logs.csv (alongside the train-step row).
# Gives intra-epoch loss curves; lower = more resolution, slower training.
# Set to 0 to disable — val is then recorded at epoch boundaries only.
PHASE1_VAL_LOG_STEPS   = 100    # 0 = epoch-only  |  e.g. 100 / 200 / 500

# ══════════════════════════════════════════════════════════════════
# PHASE 2 — Fusion head
# ══════════════════════════════════════════════════════════════════

# ── Batch sizes ────────────────────────────────────────────────────
# OOM during Phase 2?  The fusion head itself is tiny, but the pre-computed
# CLS vectors (768-dim float32 × N samples) live in RAM/VRAM.
# Fix order: (1) reduce CLS_EXTRACT_BATCH above, (2) reduce PHASE2_TRAIN_BATCH.
# Smaller batches also add gradient noise — a regularisation benefit.
PHASE2_TRAIN_BATCH  = 64      # ↓ was 512 — 512 gave too few steps/epoch → OOM + overfit
                               # OOM? try 32 → 16.  More regularisation? try 16–32.
PHASE2_EVAL_BATCH   = 128     # val/test loader.  OOM? 64 → 32

# ── Stat-feature computation batch (CPU-only) ─────────────────────
# GPT-2 perplexity runs on CPU; controls CPU RAM usage.  No GPU impact.
STAT_FEAT_BATCH     = 64      # CPU RAM tight? try 32 → 16

# ── Architecture ──────────────────────────────────────────────────
SEMANTIC_DIM        = 768     # RoBERTa CLS dim       (fixed — do not change)
STAT_DIM            = 3       # GPT-2 stat feature count (fixed — do not change)
PHASE2_HIDDEN_DIM   = 128     # ↓ was 256 — smaller head overfits less on small datasets
PHASE2_DROPOUT      = 0.50    # ↑ was 0.30 — aggressive dropout to force robust features

# ── Regularisation ────────────────────────────────────────────────
PHASE2_LR           = 5e-5    # ↓ was 3e-4 — gentler for MLP on frozen embeddings
PHASE2_WEIGHT_DECAY = 0.05    # ↑ was 0.01
PHASE2_WARMUP_FRAC  = 0.10    # fraction of total steps used for linear LR warmup
PHASE2_MAX_GRAD_NORM= 1.0

# ── Schedule ──────────────────────────────────────────────────────
PHASE2_EPOCHS       = 20
PHASE2_PATIENCE     = 5       # early-stop patience (epochs)

# ── Step-level val logging ────────────────────────────────────────
# Every PHASE2_VAL_LOG_STEPS training steps a full val pass is run
# and the result written to phase2_step_logs.csv.
# Set to 0 to disable (epoch-only logging).
PHASE2_VAL_LOG_STEPS = 50     # 0 = epoch-only  |  e.g. 25 / 50 / 100


# ═══════════════════════════════════════════════════════════════════
# CSV LOGGER
# ═══════════════════════════════════════════════════════════════════
class CSVLogger:
    def __init__(self, path, fields):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=fields).to_csv(self.path, index=False)

    def write(self, row):
        pd.DataFrame([row]).to_csv(self.path, mode="a", header=False, index=False)


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 TRAINER CALLBACK
#
# Produces two CSVs:
#
#   step_logs.csv
#     • one row per logging_steps: train_loss, lr, grad_norm
#     • one additional row per PHASE1_VAL_LOG_STEPS: val_loss added
#       (train_loss left blank on that row to keep columns aligned)
#
#   epoch_logs.csv
#     • one row per epoch: full train + val summary
# ═══════════════════════════════════════════════════════════════════
class LogCallback(TrainerCallback):
    def __init__(self, step_log: CSVLogger, epoch_log: CSVLogger,
                 val_log_steps: int, trainer_ref: list):
        self.step_log      = step_log
        self.epoch_log     = epoch_log
        self.val_log_steps = val_log_steps
        self._trainer      = trainer_ref   # filled in after Trainer is constructed

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step

        # ── Train-step row ─────────────────────────────────────
        self.step_log.write({
            "epoch":         round(state.epoch or 0, 4),
            "global_step":   step,
            "train_loss":    logs.get("loss", ""),
            "val_loss":      "",
            "learning_rate": logs.get("learning_rate", 0.0),
            "grad_norm":     logs.get("grad_norm", 0.0),
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
        })
        tqdm.write(
            f" [STEP {step:>6}] "
            f"train_loss={logs.get('loss', 0):.4f}  "
            f"lr={logs.get('learning_rate', 0):.2e}  "
            f"grad_norm={logs.get('grad_norm', 0):.4f}"
        )

        # ── Mid-step val row (optional) ─────────────────────────
        if (self.val_log_steps > 0
                and step > 0
                and step % self.val_log_steps == 0
                and self._trainer[0] is not None):
            metrics  = self._trainer[0].evaluate()
            val_loss = metrics.get("eval_loss", 0.0)
            self.step_log.write({
                "epoch":         round(state.epoch or 0, 4),
                "global_step":   step,
                "train_loss":    "",        # blank — this is a val row
                "val_loss":      val_loss,
                "learning_rate": logs.get("learning_rate", 0.0),
                "grad_norm":     logs.get("grad_norm", 0.0),
                "timestamp":     datetime.now().isoformat(timespec="seconds"),
            })
            tqdm.write(
                f" [STEP {step:>6}] "
                f"↳ val_loss={val_loss:.4f}  "
                f"val_acc={metrics.get('eval_accuracy', 0):.4f}  "
                f"val_f1={metrics.get('eval_f1', 0):.4f}"
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        train_loss = next(
            (l["loss"] for l in reversed(state.log_history)
             if "loss" in l and "eval" not in l), 0.0,
        )
        grad_norm = next(
            (l["grad_norm"] for l in reversed(state.log_history)
             if "grad_norm" in l), 0.0,
        )
        self.epoch_log.write({
            "epoch":         round(state.epoch or 0, 4),
            "global_step":   state.global_step,
            "train_loss":    train_loss,
            "val_loss":      metrics.get("eval_loss", 0.0),
            "val_accuracy":  metrics.get("eval_accuracy", 0.0),
            "val_f1_macro":  metrics.get("eval_f1", 0.0),
            "val_f1_real":   metrics.get("eval_f1_real", 0.0),
            "val_f1_fake":   metrics.get("eval_f1_fake", 0.0),
            "grad_norm":     grad_norm,
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
        })
        tqdm.write(
            f" [EPOCH {int(state.epoch or 0):>2}] "
            f"val_loss={metrics.get('eval_loss', 0):.4f}  "
            f"val_acc={metrics.get('eval_accuracy', 0):.4f}  "
            f"val_f1={metrics.get('eval_f1', 0):.4f}"
        )


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(eval_pred):
    preds  = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    f1_cls = f1_score(labels, preds, average=None, labels=[0, 1], zero_division=0)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="macro", zero_division=0),
        "f1_real":   float(f1_cls[0]),
        "f1_fake":   float(f1_cls[1]),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels, preds, average="macro", zero_division=0),
    }


# ═══════════════════════════════════════════════════════════════════
# LAYER-WISE LEARNING RATE DECAY (LLRD)
# ═══════════════════════════════════════════════════════════════════
def get_llrd_optimizer(model, base_lr: float, decay: float, weight_decay: float):
    """
    AdamW with per-layer LRs (LLRD):
      classifier head  → base_lr
      encoder layer N  → base_lr × decay^1
      encoder layer N-1→ base_lr × decay^2   (deeper = smaller LR)
      embeddings       → base_lr × decay^(N+1)

    Set LLRD_DECAY = 1.0 in CONFIG to use uniform LR everywhere.
    """
    no_decay     = {"bias", "LayerNorm.weight"}
    param_groups = []

    def _split(named_params):
        d, nd = [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            (nd if any(k in n for k in no_decay) else d).append(p)
        return d, nd

    def _add(d, nd, lr):
        if d:  param_groups.append({"params": d,  "lr": lr, "weight_decay": weight_decay})
        if nd: param_groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})

    _add(*_split(model.classifier.named_parameters()), base_lr)

    num_layers = model.roberta.config.num_hidden_layers
    for depth, idx in enumerate(range(num_layers - 1, -1, -1), start=1):
        _add(*_split(model.roberta.encoder.layer[idx].named_parameters()),
             base_lr * (decay ** depth))

    emb_lr = base_lr * (decay ** (num_layers + 1))
    _add(*_split(model.roberta.embeddings.named_parameters()), emb_lr)

    tqdm.write(
        f"  LLRD: {len(param_groups)} param groups | "
        f"head_lr={base_lr:.1e} | emb_lr={emb_lr:.2e} | decay={decay}"
    )
    return torch.optim.AdamW(param_groups, fused=True)


# ═══════════════════════════════════════════════════════════════════
# ROBERTA WITH CONFIGURABLE HEAD DROPOUT
# ═══════════════════════════════════════════════════════════════════
class RobertaWithHeadDropout(nn.Module):
    """
    Wraps RobertaForSequenceClassification, rebuilding the classifier
    head to use PHASE1_HEAD_DROPOUT (was fixed 0.1 inside HF):

        dense(768→768) → Tanh → Dropout(PHASE1_HEAD_DROPOUT) → Linear(768→2)

    Copies pretrained weights — nothing is re-initialised from scratch.
    Exposes .roberta, .classifier, .config so the HF Trainer and LLRD
    optimizer builder work unchanged.
    """
    def __init__(self, hf_model, head_dropout: float):
        super().__init__()
        self.roberta    = hf_model.roberta
        self.config     = hf_model.config
        self.num_labels = hf_model.num_labels
        h               = hf_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Dropout(head_dropout),
            nn.Linear(h, self.num_labels),
        )
        orig = hf_model.classifier
        self.classifier[0].weight.data = orig.dense.weight.data.clone()
        self.classifier[0].bias.data   = orig.dense.bias.data.clone()
        self.classifier[3].weight.data = orig.out_proj.weight.data.clone()
        self.classifier[3].bias.data   = orig.out_proj.bias.data.clone()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs    = self.roberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits     = self.classifier(cls_output)
        loss       = self.loss_fct(logits, labels) if labels is not None else None

        class _Out: pass
        out = _Out(); out.loss = loss; out.logits = logits
        return out

    def save_pretrained(self, path):
        import copy
        cfg = copy.deepcopy(self.config)
        hf  = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=self.num_labels, config=cfg)
        hf.roberta = self.roberta
        hf.classifier.dense.weight.data    = self.classifier[0].weight.data.clone()
        hf.classifier.dense.bias.data      = self.classifier[0].bias.data.clone()
        hf.classifier.out_proj.weight.data = self.classifier[3].weight.data.clone()
        hf.classifier.out_proj.bias.data   = self.classifier[3].bias.data.clone()
        hf.save_pretrained(path)


# ═══════════════════════════════════════════════════════════════════
# STATISTICAL FEATURES (CPU)
# ═══════════════════════════════════════════════════════════════════
def load_gpt2(device="cpu"):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tqdm.write("Loading GPT-2 for statistical features (CPU)...")
    tok = GPT2TokenizerFast.from_pretrained(GPT2_NAME)
    tok.pad_token = tok.eos_token
    mod = GPT2LMHeadModel.from_pretrained(GPT2_NAME).to(device)
    mod.eval()
    return tok, mod


@torch.no_grad()
def compute_stat_features(texts, gpt2_tok, gpt2_mod, batch_size=64):
    import nltk
    for res in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    from nltk.tokenize import sent_tokenize

    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  stat features", leave=False):
        for text in texts[i : i + batch_size]:
            text    = str(text) if text is not None else ""
            sents   = sent_tokenize(text) if text.strip() else [""]
            lengths = [len(s.split()) for s in sents]
            slv     = float(np.var(lengths)) if len(lengths) > 1 else 0.0
            if not text.strip():
                all_feats.append([0.0, slv, 0.0]); continue
            enc       = gpt2_tok(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(gpt2_mod.device)
            if input_ids.shape[1] < 2:
                all_feats.append([0.0, slv, 0.0]); continue
            loss = gpt2_mod(input_ids, labels=input_ids).loss.item()
            all_feats.append([loss, slv, -loss])
    return np.array(all_feats, dtype=np.float32)


def normalise_stat_features(train_f, val_f, test_f):
    mean = train_f.mean(axis=0)
    std  = train_f.std(axis=0) + 1e-8
    return (train_f - mean) / std, (val_f - mean) / std, (test_f - mean) / std


# ═══════════════════════════════════════════════════════════════════
# FUSION HEAD
# ═══════════════════════════════════════════════════════════════════
class FusionHead(nn.Module):
    """
    Shallow MLP fusing RoBERTa CLS embeddings with GPT-2 stat features.

      Linear(771→128) → ReLU → BatchNorm1d(128) → Dropout(0.5) → Linear(128→2)

    BatchNorm1d stabilises training when semantic (float32 ×768) and
    stat (normalised ×3) feature scales still diverge after normalisation,
    and helps with small batch sizes (PHASE2_TRAIN_BATCH=64).
    """
    def __init__(self,
                 semantic_dim: int   = SEMANTIC_DIM,
                 stat_dim:     int   = STAT_DIM,
                 hidden:       int   = PHASE2_HIDDEN_DIM,
                 dropout:      float = PHASE2_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim + stat_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, semantic_vec, stat_vec):
        return self.net(torch.cat([semantic_vec, stat_vec], dim=-1))


# ═══════════════════════════════════════════════════════════════════
# EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_cls_embeddings(model, tokenizer, texts, batch_size=128, device="cuda"):
    encoder = model.roberta; encoder.eval()
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  CLS embeddings", leave=False):
        enc = tokenizer(list(texts[i : i + batch_size]),
                        truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        all_embs.append(encoder(**enc).last_hidden_state[:, 0, :].cpu().float())
    return torch.cat(all_embs, dim=0)


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — RoBERTa full fine-tune
# ═══════════════════════════════════════════════════════════════════
def phase1(train_ds, val_ds, tokenizer):
    tqdm.write("\n" + "=" * 70)
    tqdm.write("PHASE 1 — RoBERTa full fine-tune")
    tqdm.write(
        f"  LR={PHASE1_LR:.1e}  warmup={PHASE1_WARMUP_RATIO}  "
        f"wd={PHASE1_WEIGHT_DECAY}  label_smooth={PHASE1_LABEL_SMOOTHING}  "
        f"head_drop={PHASE1_HEAD_DROPOUT}  LLRD={LLRD_DECAY}  "
        f"train_batch={PHASE1_TRAIN_BATCH}  val_log_steps={PHASE1_VAL_LOG_STEPS}"
    )
    tqdm.write("=" * 70)

    hf_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, device_map="auto")
    model    = RobertaWithHeadDropout(hf_model, head_dropout=PHASE1_HEAD_DROPOUT).to("cuda")
    total    = sum(p.numel() for p in model.parameters())
    tqdm.write(f"trainable params: {total:,}  (100 %)")

    Path(PHASE1_OUT).mkdir(exist_ok=True)
    step_log = CSVLogger(
        f"{PHASE1_OUT}/step_logs.csv",
        ["epoch", "global_step", "train_loss", "val_loss",
         "learning_rate", "grad_norm", "timestamp"],
    )
    epoch_log = CSVLogger(
        f"{PHASE1_OUT}/epoch_logs.csv",
        ["epoch", "global_step", "train_loss", "val_loss", "val_accuracy",
         "val_f1_macro", "val_f1_real", "val_f1_fake", "grad_norm", "timestamp"],
    )

    if LLRD_DECAY < 1.0:
        optimizer  = get_llrd_optimizer(model, PHASE1_LR, LLRD_DECAY, PHASE1_WEIGHT_DECAY)
        optimizers = (optimizer, None)
    else:
        tqdm.write("  LLRD disabled — uniform AdamW")
        optimizers = (None, None)

    trainer_ref = [None]
    callback    = LogCallback(step_log, epoch_log, PHASE1_VAL_LOG_STEPS, trainer_ref)

    args = TrainingArguments(
        output_dir=PHASE1_OUT,
        per_device_train_batch_size=PHASE1_TRAIN_BATCH,
        per_device_eval_batch_size=PHASE1_EVAL_BATCH,
        gradient_accumulation_steps=PHASE1_GRAD_ACCUM,
        fp16=False, bf16=True,
        gradient_checkpointing=False,
        optim="adamw_torch_fused" if LLRD_DECAY >= 1.0 else "adamw_torch",
        learning_rate=PHASE1_LR,
        warmup_ratio=PHASE1_WARMUP_RATIO,
        weight_decay=PHASE1_WEIGHT_DECAY,
        max_grad_norm=PHASE1_MAX_GRAD_NORM,
        label_smoothing_factor=PHASE1_LABEL_SMOOTHING,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        num_train_epochs=PHASE1_EPOCHS,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=PHASE1_PATIENCE,
                early_stopping_threshold=PHASE1_PATIENCE_THRESH,
            ),
            callback,
        ],
    )
    trainer_ref[0] = trainer   # inject so callback can call trainer.evaluate()

    trainer.train()

    best_path = f"{PHASE1_OUT}/best"
    Path(best_path).mkdir(exist_ok=True)
    model.save_pretrained(best_path)
    tokenizer.save_pretrained(best_path)
    tqdm.write(f"\n Phase 1 complete — model: {best_path}")
    tqdm.write(f"  CSVs: {PHASE1_OUT}/step_logs.csv  |  {PHASE1_OUT}/epoch_logs.csv")

    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    tqdm.write(" GPU cleared.")
    return best_path


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 HELPERS
# ═══════════════════════════════════════════════════════════════════
def _val_pass(head, loaders, crit, device):
    """Full val pass — returns (val_loss, f1_macro, accuracy, f1_real, f1_fake)."""
    head.eval()
    val_loss, preds, labs = 0.0, [], []
    with torch.no_grad():
        for sem, stat, y in loaders["val"]:
            sem, stat, y = sem.to(device), stat.to(device), y.to(device)
            logits    = head(sem, stat)
            val_loss += crit(logits, y).item()
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs.extend(y.cpu().tolist())
    val_loss /= len(loaders["val"])
    f1_cls    = f1_score(labs, preds, average=None, labels=[0, 1], zero_division=0)
    return (
        val_loss,
        float(f1_score(labs, preds, average="macro", zero_division=0)),
        float(accuracy_score(labs, preds)),
        float(f1_cls[0]),
        float(f1_cls[1]),
    )


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 — Fusion head
# ═══════════════════════════════════════════════════════════════════
def phase2(best_path, tokenizer, splits):
    tqdm.write("\n" + "=" * 70)
    tqdm.write("PHASE 2 — Fusion head")
    tqdm.write(
        f"  LR={PHASE2_LR:.1e}  hidden={PHASE2_HIDDEN_DIM}  dropout={PHASE2_DROPOUT}  "
        f"train_batch={PHASE2_TRAIN_BATCH}  wd={PHASE2_WEIGHT_DECAY}  "
        f"val_log_steps={PHASE2_VAL_LOG_STEPS}"
    )
    tqdm.write("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Frozen RoBERTa for embedding extraction ────────────────────
    tqdm.write("Reloading Phase-1 checkpoint (frozen)...")
    roberta = AutoModelForSequenceClassification.from_pretrained(
        best_path, num_labels=2, device_map="auto")
    roberta.eval()
    for p in roberta.parameters():
        p.requires_grad = False

    gpt2_tok, gpt2_mod = load_gpt2(device="cpu")

    split_names  = ["train", "val", "test"]
    split_texts  = [splits[n]["texts"]  for n in split_names]
    split_labels = [splits[n]["labels"] for n in split_names]

    # ── CLS embeddings ─────────────────────────────────────────────
    tqdm.write(f"\nExtracting CLS embeddings (batch={CLS_EXTRACT_BATCH})...")
    cls_vecs = []
    for name, texts in zip(split_names, split_texts):
        tqdm.write(f"  {name} ({len(texts):,} samples)...")
        cls_vecs.append(
            extract_cls_embeddings(roberta, tokenizer, texts,
                                   batch_size=CLS_EXTRACT_BATCH, device=device))
    del roberta; gc.collect(); torch.cuda.empty_cache()
    tqdm.write(" RoBERTa unloaded.")

    # ── Stat features ──────────────────────────────────────────────
    tqdm.write(f"\nComputing stat features (CPU, batch={STAT_FEAT_BATCH})...")
    stat_raw = []
    for name, texts in zip(split_names, split_texts):
        tqdm.write(f"  {name} ({len(texts):,} samples)...")
        stat_raw.append(compute_stat_features(texts, gpt2_tok, gpt2_mod,
                                              batch_size=STAT_FEAT_BATCH))
    del gpt2_mod, gpt2_tok; gc.collect()

    stat_n = list(normalise_stat_features(*stat_raw))

    # ── DataLoaders ────────────────────────────────────────────────
    dsets = {
        name: TensorDataset(
            cls_vecs[i],
            torch.tensor(stat_n[i], dtype=torch.float32),
            torch.tensor(list(split_labels[i]), dtype=torch.long),
        )
        for i, name in enumerate(split_names)
    }
    loaders = {
        n: DataLoader(
            dsets[n],
            batch_size=PHASE2_TRAIN_BATCH if n == "train" else PHASE2_EVAL_BATCH,
            shuffle=(n == "train"),
            num_workers=0, pin_memory=False,
        )
        for n in split_names
    }

    # ── Model / optimiser / LR scheduler ──────────────────────────
    head = FusionHead(SEMANTIC_DIM, STAT_DIM, PHASE2_HIDDEN_DIM, PHASE2_DROPOUT).to(device)

    total_steps  = PHASE2_EPOCHS * len(loaders["train"])
    warmup_steps = max(1, int(total_steps * PHASE2_WARMUP_FRAC))
    tqdm.write(f"  total steps={total_steps:,}  warmup_steps={warmup_steps}")

    opt = torch.optim.AdamW(head.parameters(), lr=PHASE2_LR, weight_decay=PHASE2_WEIGHT_DECAY)

    def lr_lambda(s):
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * p)))

    sch  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.CrossEntropyLoss()

    # ── Phase 2 CSV loggers ────────────────────────────────────────
    Path(PHASE2_OUT).mkdir(exist_ok=True)
    p2_step_log = CSVLogger(
        f"{PHASE2_OUT}/phase2_step_logs.csv",
        ["epoch", "global_step", "train_loss", "val_loss", "val_f1_macro",
         "val_accuracy", "learning_rate", "grad_norm", "timestamp"],
    )
    p2_epoch_log = CSVLogger(
        f"{PHASE2_OUT}/phase2_epoch_logs.csv",
        ["epoch", "global_step", "train_loss", "val_loss", "val_accuracy",
         "val_f1_macro", "val_f1_real", "val_f1_fake", "grad_norm", "timestamp"],
    )

    # ── Training loop ──────────────────────────────────────────────
    best_val_f1  = -1.0
    best_state   = None
    patience_cnt = 0
    global_step  = 0

    tqdm.write("\nTraining fusion head...")
    for epoch in range(1, PHASE2_EPOCHS + 1):
        head.train()
        ep_train_loss = 0.0
        ep_gnorm      = 0.0
        num_batches   = 0

        for sem, stat, labs in loaders["train"]:
            sem, stat, labs = sem.to(device), stat.to(device), labs.to(device)
            opt.zero_grad()
            loss  = crit(head(sem, stat), labs)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(head.parameters(), PHASE2_MAX_GRAD_NORM)
            ep_gnorm      += gnorm.item()
            num_batches   += 1
            opt.step(); sch.step()
            global_step   += 1
            ep_train_loss += loss.item()
            current_lr     = sch.get_last_lr()[0]
            frac_epoch     = round(epoch - 1 + num_batches / len(loaders["train"]), 4)

            # ── Step val log ──────────────────────────────────
            if PHASE2_VAL_LOG_STEPS > 0 and global_step % PHASE2_VAL_LOG_STEPS == 0:
                sv_loss, sv_f1, sv_acc, _, _ = _val_pass(head, loaders, crit, device)
                head.train()
                p2_step_log.write({
                    "epoch":        frac_epoch,
                    "global_step":  global_step,
                    "train_loss":   round(loss.item(), 5),
                    "val_loss":     round(sv_loss, 5),
                    "val_f1_macro": round(sv_f1,   4),
                    "val_accuracy": round(sv_acc,   4),
                    "learning_rate":round(current_lr, 8),
                    "grad_norm":    round(gnorm.item(), 4),
                    "timestamp":    datetime.now().isoformat(timespec="seconds"),
                })
                tqdm.write(
                    f"  [P2 STEP {global_step:>5}] "
                    f"train={loss.item():.4f}  val={sv_loss:.4f}  "
                    f"f1={sv_f1:.4f}  lr={current_lr:.2e}"
                )
            else:
                # Train-only step row (val cols blank)
                p2_step_log.write({
                    "epoch":        frac_epoch,
                    "global_step":  global_step,
                    "train_loss":   round(loss.item(), 5),
                    "val_loss":     "",
                    "val_f1_macro": "",
                    "val_accuracy": "",
                    "learning_rate":round(current_lr, 8),
                    "grad_norm":    round(gnorm.item(), 4),
                    "timestamp":    datetime.now().isoformat(timespec="seconds"),
                })

        ep_train_loss /= len(loaders["train"])
        ep_gnorm      /= max(num_batches, 1)

        # ── Epoch val pass ─────────────────────────────────────
        ev_loss, ev_f1, ev_acc, ev_f1r, ev_f1f = _val_pass(head, loaders, crit, device)
        head.train()

        p2_epoch_log.write({
            "epoch":        epoch,
            "global_step":  global_step,
            "train_loss":   round(ep_train_loss, 5),
            "val_loss":     round(ev_loss,  5),
            "val_accuracy": round(ev_acc,   4),
            "val_f1_macro": round(ev_f1,    4),
            "val_f1_real":  round(ev_f1r,   4),
            "val_f1_fake":  round(ev_f1f,   4),
            "grad_norm":    round(ep_gnorm, 4),
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
        })
        tqdm.write(
            f" [P2 EPOCH {epoch:02d}] "
            f"train={ep_train_loss:.4f}  val={ev_loss:.4f}  "
            f"f1={ev_f1:.4f}  acc={ev_acc:.4f}  "
            f"gnorm={ep_gnorm:.4f}  lr={current_lr:.2e}"
        )

        if ev_f1 > best_val_f1 + 0.001:
            best_val_f1  = ev_f1
            best_state   = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_cnt = 0
            tqdm.write(f"   ✓ new best val_f1={best_val_f1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= PHASE2_PATIENCE:
                tqdm.write(f" Early stopping at epoch {epoch}.")
                break

    # ── Test evaluation ─────────────────────────────────────────
    head.load_state_dict(best_state); head.eval()
    test_preds, test_labs = [], []
    with torch.no_grad():
        for sem, stat, labs in loaders["test"]:
            sem, stat = sem.to(device), stat.to(device)
            test_preds.extend(head(sem, stat).argmax(-1).cpu().tolist())
            test_labs.extend(labs.tolist())

    tqdm.write("\n" + classification_report(test_labs, test_preds,
                                            target_names=["Real", "Fake"]))
    torch.save({
        "head_state":   best_state,
        "stat_mean":    stat_raw[0].mean(axis=0).tolist(),
        "stat_std":     (stat_raw[0].std(axis=0) + 1e-8).tolist(),
        "semantic_dim": SEMANTIC_DIM,
        "stat_dim":     STAT_DIM,
        "hidden":       PHASE2_HIDDEN_DIM,
        "dropout":      PHASE2_DROPOUT,
        "best_val_f1":  best_val_f1,
    }, f"{PHASE2_OUT}/fusion_head.pt")

    tqdm.write(f"\n Phase 2 complete — model: {PHASE2_OUT}/fusion_head.pt")
    tqdm.write(f"  CSVs: {PHASE2_OUT}/phase2_step_logs.csv  |  {PHASE2_OUT}/phase2_epoch_logs.csv")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)

    tqdm.write("Loading data...")
    df = pd.read_csv(DATA_PATH)
    tqdm.write(f"Total: {len(df):,}  Real: {sum(df['label']==0):,}  Fake: {sum(df['label']==1):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset   = Dataset.from_pandas(df).map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
    )
    split1   = dataset.train_test_split(test_size=0.2,  seed=SEED)
    split2   = split1["test"].train_test_split(test_size=0.5, seed=SEED)
    train_ds = split1["train"]
    val_ds   = split2["train"]
    test_ds  = split2["test"]
    tqdm.write(f"Split → Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    splits = {
        "train": {"texts": train_ds["text"], "labels": train_ds["label"]},
        "val":   {"texts": val_ds["text"],   "labels": val_ds["label"]},
        "test":  {"texts": test_ds["text"],  "labels": test_ds["label"]},
    }

    best_path = phase1(train_ds, val_ds, tokenizer)
    phase2(best_path, tokenizer, splits)

    tqdm.write("\n" + "=" * 70)
    tqdm.write("ALL DONE — output files")
    tqdm.write(f"  Phase 1 step logs    : {PHASE1_OUT}/step_logs.csv")
    tqdm.write(f"  Phase 1 epoch logs   : {PHASE1_OUT}/epoch_logs.csv")
    tqdm.write(f"  Phase 1 model        : {PHASE1_OUT}/best/")
    tqdm.write(f"  Phase 2 step logs    : {PHASE2_OUT}/phase2_step_logs.csv")
    tqdm.write(f"  Phase 2 epoch logs   : {PHASE2_OUT}/phase2_epoch_logs.csv")
    tqdm.write(f"  Phase 2 model        : {PHASE2_OUT}/fusion_head.pt")
    tqdm.write("=" * 70)


if __name__ == "__main__":
    main()