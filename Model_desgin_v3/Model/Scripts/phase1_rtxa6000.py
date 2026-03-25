# train.py
# Single-phase fake news classifier using RoBERTa-base with LoRA
# Includes full logging (step_logs.csv + epoch_logs.csv)

import os, warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
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
# CONFIG — edit these paths, nothing else needs changing
# ═══════════════════════════════════════════════════════════════════
DATA_PATH       = "/hhd/aiml_cset312/final/dataset_cleaned.csv"
MODEL_OUT       = "./roberta_lora_final"       # Final model and logs saved here
MODEL_NAME      = "roberta-base"
MAX_LENGTH      = 512
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.1
EPOCHS          = 8
LR              = 2e-5
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
# TRAINER CALLBACK (Full Logging)
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
# TRAINING FUNCTION (Phase 1 Only with Full Logging)
# ═══════════════════════════════════════════════════════════════════
def train_model():
    tqdm.write("\n" + "="*70)
    tqdm.write("TRAINING RoBERTa-base with LoRA (Phase 1 Only)")
    tqdm.write("="*70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tqdm.write(f"Loading {MODEL_NAME} ...")
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

    Path(MODEL_OUT).mkdir(exist_ok=True)

    # Initialize loggers
    step_log  = CSVLogger(f"{MODEL_OUT}/step_logs.csv",
                          ["epoch","global_step","loss","learning_rate","timestamp"])
    
    epoch_log = CSVLogger(f"{MODEL_OUT}/epoch_logs.csv",
                          ["epoch","train_loss","val_loss","val_accuracy",
                           "val_f1_macro","val_f1_real","val_f1_fake","timestamp"])

    args = TrainingArguments(
        output_dir=MODEL_OUT,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        fp16=True,
        learning_rate=LR,
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

    # Save best model + tokenizer
    best_path = f"{MODEL_OUT}/best"
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)

    tqdm.write(f"\nTraining completed — Best model saved to {best_path}")

    # Final test evaluation
    tqdm.write("\nRunning final test evaluation...")
    predictions = trainer.predict(test_ds)
    preds = predictions.predictions.argmax(-1)
    
    tqdm.write("\n" + classification_report(test_ds["label"], preds,
                                            target_names=["Real", "Fake"]))

    tqdm.write("\n" + "="*70)
    tqdm.write("TRAINING DONE SUCCESSFULLY")
    tqdm.write(f"  Output folder : {MODEL_OUT}/")
    tqdm.write(f"  Best model    : {best_path}/")
    tqdm.write(f"  Step logs     : {MODEL_OUT}/step_logs.csv")
    tqdm.write(f"  Epoch logs    : {MODEL_OUT}/epoch_logs.csv")
    tqdm.write("="*70)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    tqdm.write("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    tqdm.write(f"Total samples: {len(df)} | Real: {sum(df['label']==0)} | Fake: {sum(df['label']==1)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)

    dataset = Dataset.from_pandas(df).map(tokenize, batched=True)

    # Create splits
    split1   = dataset.train_test_split(test_size=0.2, seed=SEED)
    split2   = split1["test"].train_test_split(test_size=0.5, seed=SEED)

    global train_ds, val_ds, test_ds
    train_ds = split1["train"]
    val_ds   = split2["train"]
    test_ds  = split2["test"]

    tqdm.write(f"Splits → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Start training (Phase 1 only)
    train_model()


if __name__ == "__main__":
    main()