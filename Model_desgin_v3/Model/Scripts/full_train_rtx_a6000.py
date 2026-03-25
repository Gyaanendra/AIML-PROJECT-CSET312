# train.py
# Dual-signal fake news classifier
# Architecture: RoBERTa-base (LoRA) + GPT-2 statistical features → fused linear head

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
# CONFIG
# ═══════════════════════════════════════════════════════════════════
DATA_PATH     = "dataset_cleaned.csv"
PHASE1_OUT    = "./phase1_roberta_fulltune"
PHASE2_OUT    = "./phase2_fusion_head"
MODEL_NAME    = "roberta-base"
GPT2_NAME     = "gpt2"
MAX_LENGTH    = 512
PHASE1_EPOCHS = 8
PHASE2_EPOCHS = 20
PHASE1_LR     = 2e-5
PHASE2_LR     = 3e-4          # A6000: lower than 1e-3 — full FT embeds are richer
BATCH_SIZE    = 64             # A6000: was 4 on GTX 1660
EVAL_BATCH    = 128            # A6000: separate larger eval batch
GRAD_ACCUM    = 1              # A6000: was 8 — no longer needed
SEED          = 42


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
        self.step_log  = step_log
        self.epoch_log = epoch_log

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % 100 == 0:
            self.step_log.write({
                "epoch":         round(state.epoch or 0, 4),
                "global_step":   state.global_step,
                "loss":          logs.get("loss", 0.0),
                "learning_rate": logs.get("learning_rate", 0.0),
                "grad_norm":     logs.get("grad_norm", 0.0),  # ← added
                "timestamp":     datetime.now().isoformat(timespec="seconds"),
            })
            tqdm.write(
                f" [STEP {state.global_step}] "
                f"loss={logs.get('loss', 0):.4f}  "
                f"lr={logs.get('learning_rate', 0):.1e}  "
                f"grad_norm={logs.get('grad_norm', 0):.4f}"  # ← added
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            train_loss = next(
                (l["loss"] for l in reversed(state.log_history)
                 if "loss" in l and "eval" not in l),
                0.0,
            )
            # Pull the most recent grad_norm logged before this eval
            grad_norm = next(
                (l["grad_norm"] for l in reversed(state.log_history)
                 if "grad_norm" in l),
                0.0,
            )  # ← added
            self.epoch_log.write({
                "epoch":         round(state.epoch or 0, 4),
                "train_loss":    train_loss,
                "val_loss":      metrics.get("eval_loss", 0.0),
                "val_accuracy":  metrics.get("eval_accuracy", 0.0),
                "val_f1_macro":  metrics.get("eval_f1", 0.0),
                "val_f1_real":   metrics.get("eval_f1_real", 0.0),
                "val_f1_fake":   metrics.get("eval_f1_fake", 0.0),
                "grad_norm":     grad_norm,  # ← added
                "timestamp":     datetime.now().isoformat(timespec="seconds"),
            })
            tqdm.write(
                f" [EPOCH {int(state.epoch or 0)}] "
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
# STATISTICAL FEATURES (CPU — unchanged from GTX 1660 version)
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
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        tqdm.write("  Downloading NLTK punkt_tab...")
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    from nltk.tokenize import sent_tokenize

    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  stat features", leave=False):
        chunk = texts[i : i + batch_size]
        for text in chunk:
            text = str(text) if text is not None else ""

            sents   = sent_tokenize(text) if text.strip() else [""]
            lengths = [len(s.split()) for s in sents]
            slv     = float(np.var(lengths)) if len(lengths) > 1 else 0.0

            if not text.strip():
                all_feats.append([0.0, slv, 0.0])
                continue

            enc = gpt2_tok(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            input_ids = enc["input_ids"].to(gpt2_mod.device)

            if input_ids.shape[1] < 2:
                all_feats.append([0.0, slv, 0.0])
                continue

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
    def __init__(self, semantic_dim=768, stat_dim=3, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim + stat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, semantic_vec, stat_vec):
        return self.net(torch.cat([semantic_vec, stat_vec], dim=-1))


# ═══════════════════════════════════════════════════════════════════
# EMBEDDING EXTRACTION
# A6000: batch_size 16 → 128 (fits easily in 32 GB)
# Full fine-tune path: model.roberta (no PEFT wrapper needed)
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_cls_embeddings(model, tokenizer, texts, batch_size=128, device="cuda"):
    """
    Extract 768-dim [CLS] from the frozen full fine-tuned RoBERTa.
    Full FT model path: RobertaForSequenceClassification → .roberta (encoder)
    No PEFT wrapper so path is simpler than LoRA version.
    """
    encoder = model.roberta
    encoder.eval()
    all_embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  CLS embeddings", leave=False):
        batch = list(texts[i : i + batch_size])
        enc   = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        enc  = {k: v.to(device) for k, v in enc.items()}
        out  = encoder(**enc)
        cls  = out.last_hidden_state[:, 0, :].cpu().float()
        all_embs.append(cls)

    return torch.cat(all_embs, dim=0)


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — Full fine-tune (no LoRA on A6000)
# ═══════════════════════════════════════════════════════════════════
def phase1(train_ds, val_ds, tokenizer):
    tqdm.write("\n" + "=" * 70)
    tqdm.write("PHASE 1 — RoBERTa full fine-tune (A6000, no LoRA)")
    tqdm.write("=" * 70)

    tqdm.write(f"Loading {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        device_map="auto",
    )
    # All 125M params are trainable — print for confirmation
    total = sum(p.numel() for p in model.parameters())
    tqdm.write(f"trainable params: {total:,} || all params: {total:,} || trainable%: 100.00")

    Path(PHASE1_OUT).mkdir(exist_ok=True)
    step_log  = CSVLogger(
        f"{PHASE1_OUT}/step_logs.csv",
        ["epoch", "global_step", "loss", "learning_rate", "grad_norm", "timestamp"],  # ← added grad_norm
    )
    epoch_log = CSVLogger(
        f"{PHASE1_OUT}/epoch_logs.csv",
        ["epoch", "train_loss", "val_loss", "val_accuracy",
         "val_f1_macro", "val_f1_real", "val_f1_fake", "grad_norm", "timestamp"],  # ← added grad_norm
    )

    args = TrainingArguments(
        output_dir=PHASE1_OUT,
        # ── A6000 batch config ──────────────────────────────────
        per_device_train_batch_size=BATCH_SIZE,    # 64  (was 4)
        per_device_eval_batch_size=EVAL_BATCH,     # 128 (new)
        gradient_accumulation_steps=GRAD_ACCUM,    # 1   (was 8)
        # ── precision: bf16 on Ampere ───────────────────────────
        fp16=False,                                # off (was True)
        bf16=True,                                 # on  (was False) — Ampere native
        # ── no gradient checkpointing needed ───────────────────
        gradient_checkpointing=False,              # off (was True)
        # ── faster optimizer on Ampere ─────────────────────────
        optim="adamw_torch_fused",                 # (was adamw_torch)
        # ── more dataloaders to feed large batches ─────────────
        dataloader_num_workers=8,                  # (was 2)
        dataloader_pin_memory=True,
        # ── unchanged ──────────────────────────────────────────
        num_train_epochs=PHASE1_EPOCHS,
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
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.001
            ),
            LogCallback(step_log, epoch_log),
        ],
    )

    trainer.train()

    best_path = f"{PHASE1_OUT}/best"
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    tqdm.write(f"\n Phase 1 done — saved to {best_path}")

    # Free GPU memory before Phase 2
    # On 32 GB this is optional but good practice
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    tqdm.write(" GPU cleared for Phase 2.")

    return best_path


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 — Fusion head (A6000 batch sizes)
# ═══════════════════════════════════════════════════════════════════
def phase2(best_path, tokenizer, splits):
    tqdm.write("\n" + "=" * 70)
    tqdm.write("PHASE 2 — Fusion head (semantic + statistical signals)")
    tqdm.write("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reload full fine-tuned model (frozen)
    tqdm.write("Reloading best checkpoint (frozen)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        best_path,
        num_labels=2,
        device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # GPT-2 stat features on CPU
    gpt2_tok, gpt2_mod = load_gpt2(device="cpu")

    split_names  = ["train", "val", "test"]
    split_texts  = [splits[n]["texts"]  for n in split_names]
    split_labels = [splits[n]["labels"] for n in split_names]

    # Extract CLS embeddings — A6000: batch 128
    tqdm.write("\nExtracting CLS embeddings from frozen RoBERTa...")
    cls_vecs = []
    for name, texts in zip(split_names, split_texts):
        tqdm.write(f"  {name} ({len(texts)} samples)...")
        cls_vecs.append(
            extract_cls_embeddings(model, tokenizer, texts,
                                   batch_size=128, device=device)
        )

    # Free RoBERTa
    del model
    gc.collect()
    torch.cuda.empty_cache()
    tqdm.write(" RoBERTa unloaded.")

    # Stat features
    tqdm.write("\nComputing statistical features (CPU)...")
    stat_raw = []
    for name, texts in zip(split_names, split_texts):
        tqdm.write(f"  {name} ({len(texts)} samples)...")
        stat_raw.append(compute_stat_features(texts, gpt2_tok, gpt2_mod, batch_size=64))

    del gpt2_mod, gpt2_tok
    gc.collect()

    stat_norm_tr, stat_norm_val, stat_norm_te = normalise_stat_features(*stat_raw)
    stat_norm = [stat_norm_tr, stat_norm_val, stat_norm_te]

    # Build TensorDatasets
    datasets_dict = {}
    for i, name in enumerate(split_names):
        sem  = cls_vecs[i]
        stat = torch.tensor(stat_norm[i], dtype=torch.float32)
        labs = torch.tensor(list(split_labels[i]), dtype=torch.long)
        datasets_dict[name] = TensorDataset(sem, stat, labs)

    # A6000: fusion batch 512 (was 64), num_workers=0 (in-memory TensorDataset)
    loaders = {
        n: DataLoader(
            datasets_dict[n],
            batch_size=512,            # A6000: was 64
            shuffle=(n == "train"),
            num_workers=0,             # must stay 0 — CUDA fork issue
            pin_memory=False,
        )
        for n in split_names
    }

    head = FusionHead(semantic_dim=768, stat_dim=3, hidden=256, dropout=0.3).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=PHASE2_LR, weight_decay=0.01)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PHASE2_EPOCHS)
    crit = nn.CrossEntropyLoss()

    Path(PHASE2_OUT).mkdir(exist_ok=True)
    head_log = CSVLogger(
        f"{PHASE2_OUT}/fusion_head_logs.csv",
        ["epoch", "train_loss", "val_loss", "val_accuracy",
         "val_f1_macro", "val_f1_real", "val_f1_fake", "grad_norm", "timestamp"],  # ← added grad_norm
    )

    best_val_f1  = -1.0
    best_state   = None
    patience_cnt = 0
    patience     = 5

    tqdm.write("\nTraining fusion head...")
    for epoch in range(1, PHASE2_EPOCHS + 1):
        head.train()
        train_loss  = 0.0
        total_gnorm = 0.0  # ← added
        num_batches = 0    # ← added
        for sem, stat, labs in loaders["train"]:
            sem, stat, labs = sem.to(device), stat.to(device), labs.to(device)
            opt.zero_grad()
            loss = crit(head(sem, stat), labs)
            loss.backward()
            # Compute grad norm before clipping, then clip  ← added block
            gnorm = nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            total_gnorm += gnorm.item()
            num_batches += 1
            # ── end added block
            opt.step()
            train_loss += loss.item()
        train_loss  /= len(loaders["train"])
        epoch_gnorm  = total_gnorm / max(num_batches, 1)  # ← added
        sch.step()

        head.eval()
        val_loss, all_preds, all_labs = 0.0, [], []
        with torch.no_grad():
            for sem, stat, labs in loaders["val"]:
                sem, stat, labs = sem.to(device), stat.to(device), labs.to(device)
                logits    = head(sem, stat)
                val_loss += crit(logits, labs).item()
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labs.extend(labs.cpu().tolist())
        val_loss /= len(loaders["val"])

        f1_cls  = f1_score(all_labs, all_preds, average=None, labels=[0, 1], zero_division=0)
        val_f1  = float(f1_score(all_labs, all_preds, average="macro", zero_division=0))
        val_acc = float(accuracy_score(all_labs, all_preds))

        head_log.write({
            "epoch":        epoch,
            "train_loss":   round(train_loss,  5),
            "val_loss":     round(val_loss,    5),
            "val_accuracy": round(val_acc,     4),
            "val_f1_macro": round(val_f1,      4),
            "val_f1_real":  round(float(f1_cls[0]), 4),
            "val_f1_fake":  round(float(f1_cls[1]), 4),
            "grad_norm":    round(epoch_gnorm, 4),  # ← added
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
        })
        tqdm.write(
            f" [HEAD EPOCH {epoch:02d}] "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"f1={val_f1:.4f}  acc={val_acc:.4f}  "
            f"grad_norm={epoch_gnorm:.4f}"  # ← added
        )

        if val_f1 > best_val_f1 + 0.001:
            best_val_f1  = val_f1
            best_state   = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_cnt = 0
            tqdm.write(f"   New best val_f1={best_val_f1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                tqdm.write(f" Early stopping at epoch {epoch}")
                break

    head.load_state_dict(best_state)
    head.eval()

    tqdm.write("\nFinal test evaluation...")
    test_preds, test_labs = [], []
    with torch.no_grad():
        for sem, stat, labs in loaders["test"]:
            sem, stat = sem.to(device), stat.to(device)
            test_preds.extend(head(sem, stat).argmax(-1).cpu().tolist())
            test_labs.extend(labs.tolist())

    tqdm.write("\n" + classification_report(
        test_labs, test_preds, target_names=["Real", "Fake"]
    ))

    torch.save({
        "head_state":   best_state,
        "stat_mean":    stat_raw[0].mean(axis=0).tolist(),
        "stat_std":     (stat_raw[0].std(axis=0) + 1e-8).tolist(),
        "semantic_dim": 768,
        "stat_dim":     3,
        "hidden":       256,
        "best_val_f1":  best_val_f1,
    }, f"{PHASE2_OUT}/fusion_head.pt")

    tqdm.write(f"\n Phase 2 done — saved to {PHASE2_OUT}/fusion_head.pt")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tqdm.write("Loading data...")
    df = pd.read_csv(DATA_PATH)
    tqdm.write(
        f"Total: {len(df)} | Real: {sum(df['label']==0)} | Fake: {sum(df['label']==1)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)

    dataset = Dataset.from_pandas(df).map(tokenize, batched=True)

    split1   = dataset.train_test_split(test_size=0.2, seed=SEED)
    split2   = split1["test"].train_test_split(test_size=0.5, seed=SEED)
    train_ds = split1["train"]
    val_ds   = split2["train"]
    test_ds  = split2["test"]

    tqdm.write(
        f"Split → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    splits = {
        "train": {"texts": train_ds["text"], "labels": train_ds["label"]},
        "val":   {"texts": val_ds["text"],   "labels": val_ds["label"]},
        "test":  {"texts": test_ds["text"],  "labels": test_ds["label"]},
    }

    best_path = phase1(train_ds, val_ds, tokenizer)
    phase2(best_path, tokenizer, splits)

    tqdm.write("\n" + "=" * 70)
    tqdm.write("ALL DONE")
    tqdm.write(f"  Phase 1 step logs  : {PHASE1_OUT}/step_logs.csv")
    tqdm.write(f"  Phase 1 epoch logs : {PHASE1_OUT}/epoch_logs.csv")
    tqdm.write(f"  Phase 1 model      : {PHASE1_OUT}/best/")
    tqdm.write(f"  Phase 2 head logs  : {PHASE2_OUT}/fusion_head_logs.csv")
    tqdm.write(f"  Phase 2 model      : {PHASE2_OUT}/fusion_head.pt")
    tqdm.write("=" * 70)


if __name__ == "__main__":
    main()