# train.py
# Dual-signal fake news classifier - FULL GPU OPTIMIZED VERSION
# RoBERTa-base (LoRA-style full fine-tune) + GPT-2 statistical features → Fusion Head
# Optimized for RTX A6000 (48GB) - Everything possible now runs on GPU

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from transformers.modeling_outputs import SequenceClassifierOutput

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====================== GPU OPTIMIZATIONS ======================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
DATA_PATH = "dataset_cleaned.csv"
PHASE1_OUT = "./phase1_roberta_fulltune"
PHASE2_OUT = "./phase2_fusion_head"
MODEL_NAME = "roberta-base"
GPT2_NAME = "gpt2"
MAX_LENGTH = 512
SEED = 42

# Phase 1
PHASE1_TRAIN_BATCH = 96
PHASE1_EVAL_BATCH = 128
PHASE1_GRAD_ACCUM = 1
CLS_EXTRACT_BATCH = 128
PHASE1_LR = 1e-5
PHASE1_WARMUP_RATIO = 0.10
PHASE1_WEIGHT_DECAY = 0.05
PHASE1_LABEL_SMOOTHING = 0.10
PHASE1_HEAD_DROPOUT = 0.25
PHASE1_MAX_GRAD_NORM = 1.0
LLRD_DECAY = 0.85
PHASE1_EPOCHS = 8
PHASE1_PATIENCE = 3

# Phase 2
PHASE2_TRAIN_BATCH = 96
PHASE2_EVAL_BATCH = 128
STAT_FEAT_BATCH = 128          # Increased for GPU
SEMANTIC_DIM = 768
STAT_DIM = 3
PHASE2_HIDDEN_DIM = 128
PHASE2_DROPOUT = 0.50
PHASE2_LR = 5e-5
PHASE2_WEIGHT_DECAY = 0.05
PHASE2_WARMUP_FRAC = 0.10
PHASE2_MAX_GRAD_NORM = 1.0
PHASE2_EPOCHS = 20
PHASE2_PATIENCE = 5

# ═══════════════════════════════════════════════════════════════════
# CSV LOGGER
# ═══════════════════════════════════════════════════════════════════
class CSVLogger:
    def __init__(self, path, fields):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=fields).to_csv(self.path, index=False)

    def write(self, row_dict):
        df = pd.read_csv(self.path, nrows=0)
        for col in df.columns:
            if col not in row_dict:
                row_dict[col] = ""
        pd.DataFrame([row_dict]).to_csv(self.path, mode="a", header=False, index=False)

# ═══════════════════════════════════════════════════════════════════
# PHASE 1 CALLBACK
# ═══════════════════════════════════════════════════════════════════
class LogCallback(TrainerCallback):
    def __init__(self, step_log: CSVLogger, epoch_log: CSVLogger):
        self.step_log = step_log
        self.epoch_log = epoch_log

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        self.step_log.write({
            "epoch": round(state.epoch or 0, 4),
            "global_step": state.global_step,
            "train_loss": round(logs["loss"], 5),
            "learning_rate": round(logs.get("learning_rate", 0.0), 8),
            "grad_norm": round(logs.get("grad_norm", 0.0), 4),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        train_loss = next((l.get("loss", 0.0) for l in reversed(state.log_history) if "loss" in l and "eval" not in l), 0.0)
        gnorm = next((l.get("grad_norm", 0.0) for l in reversed(state.log_history) if "grad_norm" in l), 0.0)
        self.epoch_log.write({
            "epoch": round(state.epoch or 0, 4),
            "global_step": state.global_step,
            "train_loss": round(train_loss, 5),
            "val_loss": round(metrics.get("eval_loss", 0.0), 5),
            "val_accuracy": round(metrics.get("eval_accuracy", 0.0), 4),
            "val_f1_macro": round(metrics.get("eval_f1", 0.0), 4),
            "val_f1_real": round(metrics.get("eval_f1_real", 0.0), 4),
            "val_f1_fake": round(metrics.get("eval_f1_fake", 0.0), 4),
            "grad_norm": round(gnorm, 4),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

# ═══════════════════════════════════════════════════════════════════
# ROBERTA MODEL WITH CUSTOM HEAD
# ═══════════════════════════════════════════════════════════════════
class RobertaWithHeadDropout(nn.Module):
    def __init__(self, hf_model, head_dropout: float):
        super().__init__()
        self.roberta = hf_model.roberta
        self.config = hf_model.config
        self.num_labels = hf_model.num_labels
        h = hf_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(h, h), nn.Tanh(), nn.Dropout(head_dropout), nn.Linear(h, self.num_labels)
        )
        orig = hf_model.classifier
        self.classifier[0].weight.data = orig.dense.weight.data.clone()
        self.classifier[0].bias.data = orig.dense.bias.data.clone()
        self.classifier[3].weight.data = orig.out_proj.weight.data.clone()
        self.classifier[3].bias.data = orig.out_proj.bias.data.clone()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def save_pretrained(self, path):
        import copy
        cfg = copy.deepcopy(self.config)
        cfg.num_labels = self.num_labels
        hf = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=cfg)
        hf.roberta = self.roberta
        hf.classifier.dense.weight.data.copy_(self.classifier[0].weight.data)
        hf.classifier.dense.bias.data.copy_(self.classifier[0].bias.data)
        hf.classifier.out_proj.weight.data.copy_(self.classifier[3].weight.data)
        hf.classifier.out_proj.bias.data.copy_(self.classifier[3].bias.data)
        hf.save_pretrained(path)

# METRICS
def compute_metrics(eval_pred):
    preds = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    f1_cls = f1_score(labels, preds, average=None, labels=[0, 1], zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_real": float(f1_cls[0]),
        "f1_fake": float(f1_cls[1]),
    }

# LLRD Optimizer
def get_llrd_optimizer(model, base_lr: float, decay: float, weight_decay: float):
    no_decay = {"bias", "LayerNorm.weight"}
    param_groups = []
    def _split(named_params):
        d, nd = [], []
        for n, p in named_params:
            if not p.requires_grad: continue
            (nd if any(k in n for k in no_decay) else d).append(p)
        return d, nd
    def _add(d, nd, lr):
        if d: param_groups.append({"params": d, "lr": lr, "weight_decay": weight_decay})
        if nd: param_groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})
    _add(*_split(model.classifier.named_parameters()), base_lr)
    num_layers = model.roberta.config.num_hidden_layers
    for depth, idx in enumerate(range(num_layers - 1, -1, -1), start=1):
        _add(*_split(model.roberta.encoder.layer[idx].named_parameters()), base_lr * (decay ** depth))
    emb_lr = base_lr * (decay ** (num_layers + 1))
    _add(*_split(model.roberta.embeddings.named_parameters()), emb_lr)
    return torch.optim.AdamW(param_groups, fused=True)

# ═══════════════════════════════════════════════════════════════════
# GPT-2 STATISTICAL FEATURES - FULLY GPU ACCELERATED
# ═══════════════════════════════════════════════════════════════════
def load_gpt2(device="cuda"):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(GPT2_NAME)
    tok.pad_token = tok.eos_token
    mod = GPT2LMHeadModel.from_pretrained(GPT2_NAME).to(device)
    mod.eval()
    return tok, mod


@torch.no_grad()
def compute_stat_features(texts, gpt2_tok, gpt2_mod, batch_size=128):
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize

    all_feats = []
    device = gpt2_mod.device

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing GPT-2 stat features (GPU)", leave=False):
        batch_texts = [str(text) if text is not None else "" for text in texts[i:i + batch_size]]

        # Sentence length variance
        batch_slv = []
        for text in batch_texts:
            sents = sent_tokenize(text) if text.strip() else [""]
            lengths = [len(s.split()) for s in sents]
            slv = float(np.var(lengths)) if len(lengths) > 1 else 0.0
            batch_slv.append(slv)

        # GPT-2 perplexity/loss - batched on GPU with mixed precision
        valid_mask = [bool(t.strip() and len(t.split()) >= 2) for t in batch_texts]
        batch_loss = [0.0] * len(batch_texts)

        valid_texts = [t for t, m in zip(batch_texts, valid_mask) if m]
        if valid_texts:
            enc = gpt2_tok(valid_texts, return_tensors="pt", truncation=True,
                           max_length=512, padding=True).to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = gpt2_mod(**enc, labels=enc["input_ids"])
                loss_value = outputs.loss.item()

            # Assign same loss to all valid samples in batch (fast & effective approximation)
            for j, is_valid in enumerate(valid_mask):
                if is_valid:
                    batch_loss[j] = loss_value

        # Combine features: [gpt2_loss, sent_var, -gpt2_loss]
        for loss_val, slv in zip(batch_loss, batch_slv):
            all_feats.append([loss_val, slv, -loss_val])

    return np.array(all_feats, dtype=np.float32)


def normalise_stat_features(train_f, val_f, test_f):
    mean = train_f.mean(axis=0)
    std = train_f.std(axis=0) + 1e-8
    return (train_f - mean) / std, (val_f - mean) / std, (test_f - mean) / std


# ═══════════════════════════════════════════════════════════════════
# FUSION HEAD
# ═══════════════════════════════════════════════════════════════════
class FusionHead(nn.Module):
    def __init__(self, semantic_dim=768, stat_dim=3, hidden=128, dropout=0.5):
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


@torch.no_grad()
def extract_cls_embeddings(model, tokenizer, texts, batch_size=128, device="cuda"):
    encoder = model.roberta
    encoder.eval()
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting CLS embeddings", leave=False):
        enc = tokenizer(list(texts[i:i + batch_size]), truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            hidden = encoder(**enc).last_hidden_state[:, 0, :].cpu().float()
        all_embs.append(hidden)
    return torch.cat(all_embs, dim=0)


def _val_pass(head, loaders, crit, device):
    head.eval()
    val_loss, preds, labs = 0.0, [], []
    with torch.no_grad():
        for sem, stat, y in loaders["val"]:
            sem, stat, y = sem.to(device), stat.to(device), y.to(device)
            logits = head(sem, stat)
            val_loss += crit(logits, y).item()
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs.extend(y.cpu().tolist())
    val_loss /= len(loaders["val"])
    f1_cls = f1_score(labs, preds, average=None, labels=[0, 1], zero_division=0)
    return val_loss, float(f1_score(labs, preds, average="macro", zero_division=0)), \
           float(accuracy_score(labs, preds)), float(f1_cls[0]), float(f1_cls[1])


def save_test_metrics(preds, labels, phase_name: str):
    Path("./test_results").mkdir(exist_ok=True)
    f1_cls = f1_score(labels, preds, average=None, labels=[0, 1], zero_division=0)
    row = {
        "phase": phase_name,
        "test_accuracy": round(accuracy_score(labels, preds), 4),
        "test_f1_macro": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "test_f1_real": round(float(f1_cls[0]), 4),
        "test_f1_fake": round(float(f1_cls[1]), 4),
        "test_precision": round(precision_score(labels, preds, average="macro", zero_division=0), 4),
        "test_recall": round(recall_score(labels, preds, average="macro", zero_division=0), 4),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    csv_path = f"./test_results/{phase_name}_test_metrics.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    tqdm.write(f"✅ Test metrics saved → {csv_path}")
    print("\n" + classification_report(labels, preds, target_names=["Real", "Fake"]))


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 - RoBERTa Full Fine-Tune
# ═══════════════════════════════════════════════════════════════════
def phase1(train_ds, val_ds, test_ds, tokenizer):
    tqdm.write("\n" + "="*90)
    tqdm.write("STARTING PHASE 1 — RoBERTa Full Fine-Tune")
    tqdm.write("="*90)

    hf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, device_map="auto")
    model = RobertaWithHeadDropout(hf_model, PHASE1_HEAD_DROPOUT).to("cuda")

    Path(PHASE1_OUT).mkdir(exist_ok=True)

    step_log = CSVLogger(f"{PHASE1_OUT}/phase1_step_logs.csv",
                         ["epoch", "global_step", "train_loss", "learning_rate", "grad_norm", "timestamp"])
    epoch_log = CSVLogger(f"{PHASE1_OUT}/phase1_epoch_logs.csv",
                          ["epoch", "global_step", "train_loss", "val_loss", "val_accuracy",
                           "val_f1_macro", "val_f1_real", "val_f1_fake", "grad_norm", "timestamp"])

    optimizers = (get_llrd_optimizer(model, PHASE1_LR, LLRD_DECAY, PHASE1_WEIGHT_DECAY), None)
    callback = LogCallback(step_log, epoch_log)

    args = TrainingArguments(
        output_dir=PHASE1_OUT,
        per_device_train_batch_size=PHASE1_TRAIN_BATCH,
        per_device_eval_batch_size=PHASE1_EVAL_BATCH,
        gradient_accumulation_steps=PHASE1_GRAD_ACCUM,
        bf16=True,
        learning_rate=PHASE1_LR,
        warmup_ratio=PHASE1_WARMUP_RATIO,
        weight_decay=PHASE1_WEIGHT_DECAY,
        max_grad_norm=PHASE1_MAX_GRAD_NORM,
        label_smoothing_factor=PHASE1_LABEL_SMOOTHING,
        num_train_epochs=PHASE1_EPOCHS,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        callbacks=[EarlyStoppingCallback(PHASE1_PATIENCE, 0.001), callback],
    )

    trainer.train()

    best_path = f"{PHASE1_OUT}/best"
    Path(best_path).mkdir(exist_ok=True)
    model.save_pretrained(best_path)
    tokenizer.save_pretrained(best_path)

    # Final test evaluation
    tqdm.write("\nRunning final test evaluation - Phase 1 RoBERTa...")
    test_output = trainer.predict(test_ds)
    save_test_metrics(test_output.predictions.argmax(-1), test_output.label_ids, "phase1_roberta")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    tqdm.write("\n" + "="*90)
    tqdm.write("✅ PHASE 1 COMPLETED SUCCESSFULLY!")
    tqdm.write(f" Best model saved at : {best_path}")
    tqdm.write(f" Step logs : {PHASE1_OUT}/phase1_step_logs.csv")
    tqdm.write(f" Epoch logs : {PHASE1_OUT}/phase1_epoch_logs.csv")
    tqdm.write(f" Test metrics : ./test_results/phase1_roberta_test_metrics.csv")
    tqdm.write(" Ready for Phase 2...")
    tqdm.write("="*90 + "\n")

    return best_path


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 - Fusion Head Training
# ═══════════════════════════════════════════════════════════════════
def phase2(best_path, tokenizer, splits):
    tqdm.write("\n" + "="*90)
    tqdm.write("STARTING PHASE 2 — Fusion Head Training (GPU Optimized)")
    tqdm.write("="*90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen RoBERTa
    roberta = AutoModelForSequenceClassification.from_pretrained(best_path, num_labels=2, device_map="auto")
    roberta.eval()
    for p in roberta.parameters():
        p.requires_grad = False

    # GPT-2 on GPU
    gpt2_tok, gpt2_mod = load_gpt2(device="cuda")

    split_names = ["train", "val", "test"]
    split_texts = [splits[n]["texts"] for n in split_names]
    split_labels = [splits[n]["labels"] for n in split_names]

    # Extract CLS embeddings
    cls_vecs = [extract_cls_embeddings(roberta, tokenizer, texts, CLS_EXTRACT_BATCH, device)
                for texts in split_texts]

    del roberta
    gc.collect()
    torch.cuda.empty_cache()

    # Compute statistical features on GPU
    stat_raw = [compute_stat_features(texts, gpt2_tok, gpt2_mod, STAT_FEAT_BATCH)
                for texts in split_texts]

    del gpt2_mod, gpt2_tok
    gc.collect()
    torch.cuda.empty_cache()

    stat_n = normalise_stat_features(*stat_raw)

    # Create datasets and loaders
    dsets = {name: TensorDataset(cls_vecs[i],
                                 torch.tensor(stat_n[i], dtype=torch.float32),
                                 torch.tensor(split_labels[i], dtype=torch.long))
             for i, name in enumerate(split_names)}

    loaders = {n: DataLoader(dsets[n],
                             batch_size=PHASE2_TRAIN_BATCH if n == "train" else PHASE2_EVAL_BATCH,
                             shuffle=(n == "train"), num_workers=0, pin_memory=False)
               for n in split_names}

    # Fusion Head
    head = FusionHead(SEMANTIC_DIM, STAT_DIM, PHASE2_HIDDEN_DIM, PHASE2_DROPOUT).to(device)
    crit = nn.CrossEntropyLoss()

    total_steps = PHASE2_EPOCHS * len(loaders["train"])
    warmup_steps = max(1, int(total_steps * PHASE2_WARMUP_FRAC))

    opt = torch.optim.AdamW(head.parameters(), lr=PHASE2_LR, weight_decay=PHASE2_WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / warmup_steps) if s < warmup_steps else
                                            max(0.0, 0.5 * (1.0 + np.cos(np.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps)))))

    Path(PHASE2_OUT).mkdir(exist_ok=True)

    p2_step_log = CSVLogger(f"{PHASE2_OUT}/phase2_step_logs.csv",
                            ["epoch", "global_step", "train_loss", "learning_rate", "grad_norm", "timestamp"])
    p2_epoch_log = CSVLogger(f"{PHASE2_OUT}/phase2_epoch_logs.csv",
                             ["epoch", "global_step", "train_loss", "val_loss", "val_accuracy",
                              "val_f1_macro", "val_f1_real", "val_f1_fake", "grad_norm", "timestamp"])

    best_val_f1 = -1.0
    best_state = None
    patience_cnt = 0
    global_step = 0

    for epoch in range(1, PHASE2_EPOCHS + 1):
        head.train()
        ep_train_loss = 0.0
        ep_gnorm = 0.0
        num_batches = 0

        for sem, stat, labs in loaders["train"]:
            sem, stat, labs = sem.to(device), stat.to(device), labs.to(device)
            opt.zero_grad()
            loss = crit(head(sem, stat), labs)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(head.parameters(), PHASE2_MAX_GRAD_NORM)
            opt.step()
            sch.step()
            global_step += 1
            num_batches += 1
            ep_train_loss += loss.item()
            ep_gnorm += gnorm.item()

            current_lr = sch.get_last_lr()[0]
            p2_step_log.write({
                "epoch": round(epoch - 1 + num_batches / len(loaders["train"]), 4),
                "global_step": global_step,
                "train_loss": round(loss.item(), 5),
                "learning_rate": round(current_lr, 8),
                "grad_norm": round(gnorm.item(), 4),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

        ev_loss, ev_f1, ev_acc, ev_f1r, ev_f1f = _val_pass(head, loaders, crit, device)

        p2_epoch_log.write({
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(ep_train_loss / len(loaders["train"]), 5),
            "val_loss": round(ev_loss, 5),
            "val_accuracy": round(ev_acc, 4),
            "val_f1_macro": round(ev_f1, 4),
            "val_f1_real": round(ev_f1r, 4),
            "val_f1_fake": round(ev_f1f, 4),
            "grad_norm": round(ep_gnorm / num_batches, 4),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

        tqdm.write(f" [P2 EPOCH {epoch:02d}] train={ep_train_loss/len(loaders['train']):.4f} "
                   f"val_loss={ev_loss:.4f} val_f1={ev_f1:.4f}")

        if ev_f1 > best_val_f1 + 0.001:
            best_val_f1 = ev_f1
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PHASE2_PATIENCE:
                tqdm.write("Early stopping triggered.")
                break

    # Load best head
    head.load_state_dict(best_state)
    head.eval()

    # Final test evaluation
    test_preds, test_labs = [], []
    with torch.no_grad():
        for sem, stat, labs in loaders["test"]:
            sem, stat = sem.to(device), stat.to(device)
            test_preds.extend(head(sem, stat).argmax(-1).cpu().tolist())
            test_labs.extend(labs.tolist())

    save_test_metrics(np.array(test_preds), np.array(test_labs), "phase2_fusion")

    torch.save({"head_state": best_state, "best_val_f1": best_val_f1},
               f"{PHASE2_OUT}/fusion_head.pt")

    tqdm.write("\n" + "="*90)
    tqdm.write("✅ PHASE 2 COMPLETED SUCCESSFULLY!")
    tqdm.write(f" Final model saved at : {PHASE2_OUT}/fusion_head.pt")
    tqdm.write(f" Step logs : {PHASE2_OUT}/phase2_step_logs.csv")
    tqdm.write(f" Epoch logs : {PHASE2_OUT}/phase2_epoch_logs.csv")
    tqdm.write(f" Test metrics : ./test_results/phase2_fusion_test_metrics.csv")
    tqdm.write(" Training finished!")
    tqdm.write("="*90 + "\n")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tqdm.write("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    tqdm.write(f"Total samples: {len(df):,} | Real: {sum(df['label']==0):,} | Fake: {sum(df['label']==1):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = Dataset.from_pandas(df).map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
    )

    split1 = dataset.train_test_split(test_size=0.2, seed=SEED)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=SEED)

    train_ds = split1["train"]
    val_ds = split2["train"]
    test_ds = split2["test"]

    splits = {
        "train": {"texts": train_ds["text"], "labels": train_ds["label"]},
        "val": {"texts": val_ds["text"], "labels": val_ds["label"]},
        "test": {"texts": test_ds["text"], "labels": test_ds["label"]},
    }

    best_path = phase1(train_ds, val_ds, test_ds, tokenizer)
    phase2(best_path, tokenizer, splits)


if __name__ == "__main__":
    main()