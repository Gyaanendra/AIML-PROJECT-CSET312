# phase2_gpu.py
# Phase 2 — Fusion head  (GPU-accelerated GPT-2 stat features)
#
# Key changes vs train.py Phase 2:
#   • GPT-2 moved to CUDA — stat features drop from ~14 s/it → <1 s/it
#   • GPT-2 runs in bf16 on Ampere (same as RoBERTa phase)
#   • Batched GPT-2 inference (batch_size=128) instead of 1 sample at a time
#   • Sentence variance still computed on CPU (pure numpy — already fast)
#   • Everything else (fusion head, loaders, logging) unchanged from train.py
#
# Usage:
#   python phase2_gpu.py
#
# Expects:
#   ./phase1_roberta_fulltune/best/   — saved from Phase 1
#   dataset_cleaned.csv               — same CSV used in Phase 1

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
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ═══════════════════════════════════════════════════════════════════
# CONFIG  — must match Phase 1 exactly
# ═══════════════════════════════════════════════════════════════════
DATA_PATH    = "dataset_cleaned.csv"
PHASE1_BEST  = "./phase1_roberta_fulltune/best"
PHASE2_OUT   = "./phase2_fusion_head"
MODEL_NAME   = "roberta-base"
GPT2_NAME    = "gpt2"
MAX_LENGTH   = 512
PHASE2_EPOCHS= 20
PHASE2_LR    = 3e-4
SEED         = 42

# ── GPU batch sizes ──────────────────────────────────────────────
# A6000 48 GB: GPT-2 (small, 117M) fits easily alongside frozen RoBERTa
# Use large batches so GPU stays saturated
GPT2_BATCH       = 64   # GPT-2 stat-feature extraction  (was 64 on CPU)
CLS_BATCH        = 128   # RoBERTa CLS extraction          (unchanged)
FUSION_BATCH     = 64   # Fusion head training            (unchanged)


# ═══════════════════════════════════════════════════════════════════
# CSV LOGGER  (identical to train.py)
# ═══════════════════════════════════════════════════════════════════
class CSVLogger:
    def __init__(self, path, fields):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=fields).to_csv(self.path, index=False)

    def write(self, row):
        pd.DataFrame([row]).to_csv(self.path, mode="a", header=False, index=False)


# ═══════════════════════════════════════════════════════════════════
# FUSION HEAD  (identical to train.py)
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
# GPT-2  —  loaded on GPU with bf16
# ═══════════════════════════════════════════════════════════════════
def load_gpt2(device):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tqdm.write(f"Loading GPT-2 on {device} (bf16)...")
    tok = GPT2TokenizerFast.from_pretrained(GPT2_NAME)
    tok.pad_token = tok.eos_token
    # torch_dtype=torch.bfloat16 halves VRAM and speeds up Ampere significantly
    mod = GPT2LMHeadModel.from_pretrained(
        GPT2_NAME,
        torch_dtype=torch.bfloat16,
    ).to(device)
    mod.eval()
    return tok, mod


# ═══════════════════════════════════════════════════════════════════
# STAT FEATURES  —  GPU-batched GPT-2 inference
#
# Key difference from train.py:
#   • Tokenise a whole batch at once (left-padded via padding_side='left')
#   • Forward pass the batch on GPU
#   • Per-sample CE loss extracted from per-token loss manually
#     (because model(input_ids, labels=input_ids).loss averages over the
#      *whole* batch — we need per-sample values)
#   • Sentence variance still computed per-sample on CPU (cheap)
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_stat_features_gpu(texts, gpt2_tok, gpt2_mod, device, batch_size=256):
    import nltk
    for corpus in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)
    from nltk.tokenize import sent_tokenize

    # GPT-2 tokenizer: pad on the LEFT so loss on real tokens is contiguous
    gpt2_tok.padding_side = "left"

    all_feats = []
    texts = list(texts)

    for i in tqdm(range(0, len(texts), batch_size), desc="  stat features (GPU)", leave=False):
        chunk = [str(t) if t is not None else "" for t in texts[i : i + batch_size]]

        # ── sentence length variance (CPU, cheap) ─────────────────
        slvs = []
        for text in chunk:
            sents   = sent_tokenize(text) if text.strip() else [""]
            lengths = [len(s.split()) for s in sents]
            slvs.append(float(np.var(lengths)) if len(lengths) > 1 else 0.0)

        # ── GPT-2 per-sample cross-entropy loss (GPU) ─────────────
        enc = gpt2_tok(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,          # batch padding (left-side)
        )
        input_ids      = enc["input_ids"].to(device)          # [B, T]
        attention_mask = enc["attention_mask"].to(device)     # [B, T]

        # Forward pass — get per-token logits
        outputs = gpt2_mod(input_ids=input_ids)
        logits  = outputs.logits                              # [B, T, vocab]

        # Shift: predict token t+1 from token t
        shift_logits = logits[:, :-1, :].contiguous()        # [B, T-1, vocab]
        shift_labels = input_ids[:, 1:].contiguous()         # [B, T-1]
        shift_mask   = attention_mask[:, 1:].contiguous()    # [B, T-1]  (left-pad zeros)

        # Per-token CE loss
        loss_fn   = nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())                           # [B, T-1]

        # Mask padding tokens and compute per-sample mean loss
        token_loss = token_loss * shift_mask.float()
        n_real     = shift_mask.float().sum(dim=1).clamp(min=1)  # [B]
        ce_losses  = (token_loss.sum(dim=1) / n_real).cpu().float().numpy()  # [B]

        for ce, slv in zip(ce_losses, slvs):
            # Handle empty / ultra-short texts
            if np.isnan(ce) or np.isinf(ce):
                ce = 0.0
            all_feats.append([float(ce), slv, -float(ce)])

    return np.array(all_feats, dtype=np.float32)


def normalise_stat_features(train_f, val_f, test_f):
    mean = train_f.mean(axis=0)
    std  = train_f.std(axis=0) + 1e-8
    return (train_f - mean) / std, (val_f - mean) / std, (test_f - mean) / std


# ═══════════════════════════════════════════════════════════════════
# CLS EMBEDDING EXTRACTION  (identical to train.py)
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_cls_embeddings(model, tokenizer, texts, batch_size=128, device="cuda"):
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
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Device: {device}")

    # ── Load & split data (same seed as Phase 1) ──────────────────
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

    splits = {
        "train": {"texts": train_ds["text"], "labels": train_ds["label"]},
        "val":   {"texts": val_ds["text"],   "labels": val_ds["label"]},
        "test":  {"texts": test_ds["text"],  "labels": test_ds["label"]},
    }
    split_names = ["train", "val", "test"]

    # ── Load frozen RoBERTa from Phase 1 best ─────────────────────
    tqdm.write(f"\nLoading frozen RoBERTa from {PHASE1_BEST}...")
    roberta = AutoModelForSequenceClassification.from_pretrained(
        PHASE1_BEST, num_labels=2, device_map="auto",
    )
    roberta.eval()
    for p in roberta.parameters():
        p.requires_grad = False

    # ── Extract CLS embeddings ─────────────────────────────────────
    tqdm.write("\nExtracting CLS embeddings (frozen RoBERTa)...")
    cls_vecs = []
    for name in split_names:
        texts = splits[name]["texts"]
        tqdm.write(f"  {name} ({len(texts)} samples)...")
        cls_vecs.append(
            extract_cls_embeddings(roberta, tokenizer, texts,
                                   batch_size=CLS_BATCH, device=device)
        )

    del roberta
    gc.collect()
    torch.cuda.empty_cache()
    tqdm.write("RoBERTa unloaded.")

    # ── GPT-2 stat features ON GPU ────────────────────────────────
    gpt2_tok, gpt2_mod = load_gpt2(device)

    tqdm.write("\nComputing statistical features (GPU)...")
    stat_raw = []
    for name in split_names:
        texts = splits[name]["texts"]
        tqdm.write(f"  {name} ({len(texts)} samples)...")
        stat_raw.append(
            compute_stat_features_gpu(texts, gpt2_tok, gpt2_mod,
                                      device=device, batch_size=GPT2_BATCH)
        )

    del gpt2_mod, gpt2_tok
    gc.collect()
    torch.cuda.empty_cache()
    tqdm.write("GPT-2 unloaded.")

    # ── Normalise stat features ────────────────────────────────────
    stat_norm_tr, stat_norm_val, stat_norm_te = normalise_stat_features(*stat_raw)
    stat_norm = [stat_norm_tr, stat_norm_val, stat_norm_te]

    # ── Build TensorDatasets ───────────────────────────────────────
    datasets_dict = {}
    for i, name in enumerate(split_names):
        sem  = cls_vecs[i]
        stat = torch.tensor(stat_norm[i], dtype=torch.float32)
        labs = torch.tensor(list(splits[name]["labels"]), dtype=torch.long)
        datasets_dict[name] = TensorDataset(sem, stat, labs)

    loaders = {
        n: DataLoader(
            datasets_dict[n],
            batch_size=FUSION_BATCH,
            shuffle=(n == "train"),
            num_workers=0,
            pin_memory=False,
        )
        for n in split_names
    }

    # ── Fusion head training ───────────────────────────────────────
    head = FusionHead(semantic_dim=768, stat_dim=3, hidden=256, dropout=0.3).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=PHASE2_LR, weight_decay=0.01)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PHASE2_EPOCHS)
    crit = nn.CrossEntropyLoss()

    Path(PHASE2_OUT).mkdir(exist_ok=True)
    head_log = CSVLogger(
        f"{PHASE2_OUT}/fusion_head_logs.csv",
        ["epoch", "train_loss", "val_loss", "val_accuracy",
         "val_f1_macro", "val_f1_real", "val_f1_fake", "grad_norm", "timestamp"],
    )

    best_val_f1  = -1.0
    best_state   = None
    patience_cnt = 0
    patience     = 5

    tqdm.write("\nTraining fusion head...")
    for epoch in range(1, PHASE2_EPOCHS + 1):
        head.train()
        train_loss  = 0.0
        total_gnorm = 0.0
        num_batches = 0
        for sem, stat, labs in loaders["train"]:
            sem, stat, labs = sem.to(device), stat.to(device), labs.to(device)
            opt.zero_grad()
            loss = crit(head(sem, stat), labs)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            total_gnorm += gnorm.item()
            num_batches += 1
            opt.step()
            train_loss += loss.item()
        train_loss  /= len(loaders["train"])
        epoch_gnorm  = total_gnorm / max(num_batches, 1)
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
            "grad_norm":    round(epoch_gnorm, 4),
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
        })
        tqdm.write(
            f" [EPOCH {epoch:02d}] "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"f1={val_f1:.4f}  acc={val_acc:.4f}  "
            f"grad_norm={epoch_gnorm:.4f}"
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

    tqdm.write("\n" + classification_report(test_labs, test_preds,
                                            target_names=["Real", "Fake"]))

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
    tqdm.write(f"  Logs : {PHASE2_OUT}/fusion_head_logs.csv")


if __name__ == "__main__":
    main()