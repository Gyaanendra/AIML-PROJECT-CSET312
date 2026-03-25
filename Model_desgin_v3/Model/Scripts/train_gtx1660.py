# train.py
# Dual-signal fake news classifier
# Architecture: RoBERTa-base (LoRA) + GPT-2 statistical features → fused linear head
#
# Two training phases:
#   Phase 1 — Fine-tune RoBERTa LoRA on your 240k (semantic branch)
#   Phase 2 — Freeze RoBERTa, extract embeddings + stat features, train fusion head
#
# GPU  : GTX 1660 Super (6 GB VRAM)
# torch: 2.7.1+cu118 | transformers: 5.3.0 | peft: 0.18.1
#
# pip install peft transformers datasets accelerate scikit-learn tqdm pandas nltk

import os, math, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
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
# CONFIG — edit these paths, nothing else needs changing
# ═══════════════════════════════════════════════════════════════════
DATA_PATH       = "/hhd/aiml_cset312/final/dataset_cleaned.csv"
PHASE1_OUT      = "./phase1_roberta_lora"      # LoRA checkpoint saved here
PHASE2_OUT      = "./phase2_fusion_head"       # final fused model saved here
MODEL_NAME      = "roberta-base"
GPT2_NAME       = "gpt2"                       # for perplexity features (small, ~500MB)
MAX_LENGTH      = 512
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.1
PHASE1_EPOCHS   = 8
PHASE2_EPOCHS   = 20
PHASE1_LR       = 2e-5
PHASE2_LR       = 1e-3
BATCH_SIZE      = 4
GRAD_ACCUM      = 8
SEED            = 42


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
# TRAINER CALLBACK
# ═══════════════════════════════════════════════════════════════════
class LogCallback(TrainerCallback):
    def __init__(self, step_log, epoch_log):
        self.step_log   = step_log
        self.epoch_log  = epoch_log

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % 100 == 0:
            self.step_log.write({
                "epoch":         round(state.epoch or 0, 4),
                "global_step":   state.global_step,
                "loss":          logs.get("loss", 0.0),
                "learning_rate": logs.get("learning_rate", 0.0),
                "timestamp":     datetime.now().isoformat(timespec="seconds"),
            })
            tqdm.write(f" [STEP {state.global_step}] loss={logs.get('loss',0):.4f}  lr={logs.get('learning_rate',0):.1e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            train_loss = next(
                (l["loss"] for l in reversed(state.log_history) if "loss" in l and "eval" not in l),
                0.0
            )
            self.epoch_log.write({
                "epoch":         round(state.epoch or 0, 4),
                "train_loss":    train_loss,
                "val_loss":      metrics.get("eval_loss", 0.0),
                "val_accuracy":  metrics.get("eval_accuracy", 0.0),
                "val_f1_macro":  metrics.get("eval_f1", 0.0),
                "val_f1_real":   metrics.get("eval_f1_real", 0.0),
                "val_f1_fake":   metrics.get("eval_f1_fake", 0.0),
                "timestamp":     datetime.now().isoformat(timespec="seconds"),
            })
            tqdm.write(
                f" [EPOCH {int(state.epoch or 0)}] "
                f"val_loss={metrics.get('eval_loss',0):.4f}  "
                f"val_acc={metrics.get('eval_accuracy',0):.4f}  "
                f"val_f1={metrics.get('eval_f1',0):.4f}"
            )


# ═══════════════════════════════════════════════════════════════════
# METRICS (phase 1 Trainer)
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
# STATISTICAL FEATURE EXTRACTION
# Three lightweight features that specifically catch AI-generated text:
#
#  1. GPT-2 perplexity  — AI text has LOW perplexity (model is confident).
#                         Human fake news is messier, higher perplexity.
#
#  2. Sentence length variance — AI writes uniformly. Humans vary wildly.
#                         Low variance = strong AI signal.
#
#  3. Mean token log-prob — related to perplexity but per-token.
#                         AI text tokens are more "expected" by GPT-2.
#
# All three are computed on CPU; no VRAM cost.
# ═══════════════════════════════════════════════════════════════════
def load_gpt2(device="cpu"):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tqdm.write("Loading GPT-2 small for statistical features...")
    gpt2_tok = GPT2TokenizerFast.from_pretrained(GPT2_NAME)
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_mod = GPT2LMHeadModel.from_pretrained(GPT2_NAME).to(device)
    gpt2_mod.eval()
    return gpt2_tok, gpt2_mod


@torch.no_grad()
def compute_stat_features(texts, gpt2_tok, gpt2_mod, batch_size=32):
    """
    Returns np.ndarray of shape (N, 3):
        col 0 — log-perplexity (lower = more AI-like)
        col 1 — sentence length variance (lower = more AI-like)
        col 2 — mean token log-probability (higher magnitude = more AI-like)
    All values are raw; normalisation happens before fusion.
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    from nltk.tokenize import sent_tokenize

    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  stat features", leave=False):
        batch_texts = texts[i : i + batch_size]
        feats = []
        for text in batch_texts:
            # ── sentence length variance ──────────────────────────
            sents   = sent_tokenize(str(text))
            lengths = [len(s.split()) for s in sents] if sents else [0]
            slv     = float(np.var(lengths)) if len(lengths) > 1 else 0.0

            # ── GPT-2 perplexity + mean token log-prob ────────────
            enc = gpt2_tok(
                str(text),
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = enc["input_ids"].to(gpt2_mod.device)
            if input_ids.shape[1] < 2:
                feats.append([0.0, slv, 0.0])
                continue

            outputs  = gpt2_mod(input_ids, labels=input_ids)
            log_ppl  = outputs.loss.item()          # cross-entropy = log-perplexity
            # mean token log-prob = -loss
            mean_lp  = -log_ppl

            feats.append([log_ppl, slv, mean_lp])

        all_feats.extend(feats)

    return np.array(all_feats, dtype=np.float32)


def normalise_stat_features(train_feats, val_feats, test_feats):
    """Z-score normalise using train stats only (no leakage)."""
    mean = train_feats.mean(axis=0)
    std  = train_feats.std(axis=0) + 1e-8
    return (
        (train_feats - mean) / std,
        (val_feats   - mean) / std,
        (test_feats  - mean) / std,
    )


# ═══════════════════════════════════════════════════════════════════
# FUSION HEAD
# ═══════════════════════════════════════════════════════════════════
class FusionHead(nn.Module):
    """
    Combines 768-dim RoBERTa [CLS] embedding with 3-dim stat features.
    Input : (semantic_vec [768], stat_vec [3]) → concat [771]
    Output: logits [2]
    """
    def __init__(self, semantic_dim=768, stat_dim=3, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim + stat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, semantic_vec, stat_vec):
        x = torch.cat([semantic_vec, stat_vec], dim=-1)
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════
# EMBEDDING EXTRACTION (phase 2 — RoBERTa frozen)
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_cls_embeddings(model, tokenizer, texts, batch_size=16, device="cuda"):
    """Extract [CLS] token hidden states from the frozen LoRA RoBERTa."""
    # We need the base encoder, not the classification head
    # Access through peft model → base_model → roberta
    encoder = model.base_model.model.roberta
    encoder.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  cls embeddings", leave=False):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        enc   = {k: v.to(device) for k, v in enc.items()}
        out   = encoder(**enc)
        cls   = out.last_hidden_state[:, 0, :]   # [CLS] token
        all_embeddings.append(cls.cpu().float())

    return torch.cat(all_embeddings, dim=0)


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — RoBERTa LoRA fine-tuning
# ═══════════════════════════════════════════════════════════════════
def phase1(train_ds, val_ds):
    tqdm.write("\n" + "="*70)
    tqdm.write("PHASE 1 — RoBERTa LoRA semantic fine-tuning")
    tqdm.write("="*70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tqdm.write(f"Loading {MODEL_NAME} (fp32, device_map=auto)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"],
        task_type=TaskType.SEQ_CLS,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    Path(PHASE1_OUT).mkdir(exist_ok=True)
    step_log  = CSVLogger(f"{PHASE1_OUT}/step_logs.csv",
                          ["epoch","global_step","loss","learning_rate","timestamp"])
    epoch_log = CSVLogger(f"{PHASE1_OUT}/epoch_logs.csv",
                          ["epoch","train_loss","val_loss","val_accuracy",
                           "val_f1_macro","val_f1_real","val_f1_fake","timestamp"])

    args = TrainingArguments(
        output_dir=PHASE1_OUT,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=PHASE1_EPOCHS,
        fp16=True,
        bf16=False,
        learning_rate=PHASE1_LR,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        fp16_full_eval=False,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001),
            LogCallback(step_log, epoch_log),
        ],
    )

    trainer.train()

    # Save LoRA weights + tokenizer
    best_path = f"{PHASE1_OUT}/best"
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    tqdm.write(f"\n Phase 1 done — LoRA model saved to {best_path}")
    return best_path, tokenizer


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 — Fusion head training
# ═══════════════════════════════════════════════════════════════════
def phase2(best_path, tokenizer, splits):
    tqdm.write("\n" + "="*70)
    tqdm.write("PHASE 2 — Fusion head training (semantic + statistical)")
    tqdm.write("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Reload best LoRA model (frozen) ──────────────────────────
    tqdm.write("Reloading best LoRA model (frozen for embedding extraction)...")
    from peft import PeftModel
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, device_map="auto"
    )
    lora_model = PeftModel.from_pretrained(base, best_path)
    lora_model.eval()
    # Freeze all LoRA params — only fusion head will train
    for p in lora_model.parameters():
        p.requires_grad = False

    # ── Load GPT-2 for stat features (CPU) ───────────────────────
    gpt2_tok, gpt2_mod = load_gpt2(device="cpu")

    # ── Extract features for all splits ──────────────────────────
    split_names  = ["train", "val", "test"]
    split_texts  = [splits[n]["texts"]  for n in split_names]
    split_labels = [splits[n]["labels"] for n in split_names]

    tqdm.write("\nExtracting CLS embeddings from frozen RoBERTa...")
    cls_vecs = []
    for name, texts in zip(split_names, split_texts):
        tqdm.write(f"  {name} ({len(texts)} samples)...")
        cls_vecs.append(extract_cls_embeddings(lora_model, tokenizer, texts,
                                               batch_size=16, device=device))

    tqdm.write("\nComputing statistical features (CPU)...")
    stat_raw = []
    for name, texts in zip(split_names, split_texts):
        tqdm.write(f"  {name}...")
        stat_raw.append(compute_stat_features(texts, gpt2_tok, gpt2_mod, batch_size=64))

    # Normalise stat features using train stats only
    stat_norm = normalise_stat_features(*stat_raw)

    # Build TensorDatasets
    datasets = {}
    for i, name in enumerate(split_names):
        sem   = cls_vecs[i]
        stat  = torch.tensor(stat_norm[i], dtype=torch.float32)
        labs  = torch.tensor(split_labels[i], dtype=torch.long)
        datasets[name] = TensorDataset(sem, stat, labs)

    loaders = {
        n: DataLoader(datasets[n],
                      batch_size=64,
                      shuffle=(n == "train"),
                      num_workers=2)
        for n in split_names
    }

    # ── Build and train fusion head ───────────────────────────────
    head = FusionHead(semantic_dim=768, stat_dim=3, hidden=256, dropout=0.3).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=PHASE2_LR, weight_decay=0.01)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PHASE2_EPOCHS)
    crit = nn.CrossEntropyLoss()

    Path(PHASE2_OUT).mkdir(exist_ok=True)
    head_log = CSVLogger(f"{PHASE2_OUT}/fusion_head_logs.csv",
                         ["epoch","train_loss","val_loss","val_accuracy",
                          "val_f1_macro","val_f1_real","val_f1_fake","timestamp"])

    best_val_f1   = 0.0
    best_state    = None
    patience_cnt  = 0
    patience      = 5

    tqdm.write("\nTraining fusion head...")
    for epoch in range(1, PHASE2_EPOCHS + 1):
        # train
        head.train()
        train_loss = 0.0
        for sem, stat, labs in loaders["train"]:
            sem, stat, labs = sem.to(device), stat.to(device), labs.to(device)
            opt.zero_grad()
            logits = head(sem, stat)
            loss   = crit(logits, labs)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(loaders["train"])
        sch.step()

        # val
        head.eval()
        val_loss, all_preds, all_labs = 0.0, [], []
        with torch.no_grad():
            for sem, stat, labs in loaders["val"]:
                sem, stat, labs = sem.to(device), stat.to(device), labs.to(device)
                logits  = head(sem, stat)
                val_loss += crit(logits, labs).item()
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labs.extend(labs.cpu().tolist())
        val_loss /= len(loaders["val"])

        f1_cls   = f1_score(all_labs, all_preds, average=None, labels=[0,1], zero_division=0)
        val_f1   = float(f1_score(all_labs, all_preds, average="macro", zero_division=0))
        val_acc  = float(accuracy_score(all_labs, all_preds))

        head_log.write({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 5),
            "val_loss":    round(val_loss, 5),
            "val_accuracy":round(val_acc, 4),
            "val_f1_macro":round(val_f1, 4),
            "val_f1_real": round(float(f1_cls[0]), 4),
            "val_f1_fake": round(float(f1_cls[1]), 4),
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
        })
        tqdm.write(
            f" [HEAD EPOCH {epoch:02d}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_f1={val_f1:.4f}  val_acc={val_acc:.4f}"
        )

        if val_f1 > best_val_f1 + 0.001:
            best_val_f1  = val_f1
            best_state   = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                tqdm.write(f" Early stopping fusion head at epoch {epoch}")
                break

    # Restore best
    head.load_state_dict(best_state)
    head.eval()

    # ── Final test evaluation ─────────────────────────────────────
    tqdm.write("\nFinal test evaluation...")
    test_preds, test_labs = [], []
    with torch.no_grad():
        for sem, stat, labs in loaders["test"]:
            sem, stat = sem.to(device), stat.to(device)
            preds = head(sem, stat).argmax(-1).cpu().tolist()
            test_preds.extend(preds)
            test_labs.extend(labs.tolist())

    tqdm.write("\n" + classification_report(test_labs, test_preds,
                                            target_names=["Real", "Fake"]))

    # Save fusion head + stat normalisation params
    torch.save({
        "head_state":      best_state,
        "stat_mean":       stat_raw[0].mean(axis=0).tolist(),
        "stat_std":        (stat_raw[0].std(axis=0) + 1e-8).tolist(),
        "semantic_dim":    768,
        "stat_dim":        3,
        "hidden":          256,
        "best_val_f1":     best_val_f1,
    }, f"{PHASE2_OUT}/fusion_head.pt")

    tqdm.write(f"\n Phase 2 done — fusion head saved to {PHASE2_OUT}/fusion_head.pt")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load data ─────────────────────────────────────────────────
    tqdm.write("Loading data...")
    df = pd.read_csv(DATA_PATH)
    tqdm.write(f"Total: {len(df)} | Real: {sum(df['label']==0)} | Fake: {sum(df['label']==1)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)

    dataset = Dataset.from_pandas(df).map(tokenize, batched=True)

    split1   = dataset.train_test_split(test_size=0.2, seed=SEED)
    split2   = split1["test"].train_test_split(test_size=0.5, seed=SEED)
    train_ds = split1["train"]
    val_ds   = split2["train"]
    test_ds  = split2["test"]

    tqdm.write(f"Split → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Raw text + labels needed for stat feature extraction
    splits = {
        "train": {"texts":  train_ds["text"],  "labels": train_ds["label"]},
        "val":   {"texts":  val_ds["text"],    "labels": val_ds["label"]},
        "test":  {"texts":  test_ds["text"],   "labels": test_ds["label"]},
    }

    # ── Phase 1 ───────────────────────────────────────────────────
    best_path, tokenizer = phase1(train_ds, val_ds)

    # ── Phase 2 ───────────────────────────────────────────────────
    phase2(best_path, tokenizer, splits)

    tqdm.write("\n" + "="*70)
    tqdm.write("ALL DONE")
    tqdm.write(f"  Phase 1 logs  : {PHASE1_OUT}/")
    tqdm.write(f"  Phase 1 model : {PHASE1_OUT}/best/")
    tqdm.write(f"  Phase 2 logs  : {PHASE2_OUT}/fusion_head_logs.csv")
    tqdm.write(f"  Phase 2 model : {PHASE2_OUT}/fusion_head.pt")
    tqdm.write("="*70)


if __name__ == "__main__":
    main()