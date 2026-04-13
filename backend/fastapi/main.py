import re
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import asyncio
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
)
import nltk

nltk.download("punkt", quiet=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
PHASE1_DIR = r"H:\CSET312_project\v3\models\phase1_roberta_fulltune\best"
PHASE2_HEAD = r"H:\CSET312_project\v3\models\phase2_fusion_head\fusion_head.pt"

app = FastAPI(title="Fake News Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ── Models ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(PHASE1_DIR, use_fast=False)
roberta = AutoModelForSequenceClassification.from_pretrained(
    PHASE1_DIR, num_labels=2, output_attentions=True
).to(device)
roberta.eval()


class FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768 + 3, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, semantic, stat):
        return self.net(torch.cat([semantic, stat], dim=-1))


fusion_head = None
if Path(PHASE2_HEAD).exists():
    ckpt = torch.load(PHASE2_HEAD, map_location=device, weights_only=False)
    fusion_head = FusionHead().to(device)
    fusion_head.load_state_dict(ckpt["head_state"])
    fusion_head.eval()

gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_tok.pad_token = gpt2_tok.eos_token
gpt2_mod = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_mod.eval()

# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_word(word: str) -> str:
    word = word.replace("\u00e2\u0080\u0099", "'")
    word = word.replace("\u00e2\u0080\u009c", '"')
    word = word.replace("\u00e2\u0080\u009d", '"')
    word = word.replace("\u010c\u201c", '"')
    word = word.replace("\u010c\u2122", "'")
    word = word.replace("\u00e2\u0122\u017e", '"')
    word = word.replace("\u00e2\u0122\u00be", '"')
    word = word.replace("\u00e2\u0122\u0080", "...")
    word = word.replace("\u00e2\u0122\u0081", "-")
    word = word.replace("\u00c4\u0141", "g")
    word = word.replace("\u00c4\u00b1", "i")
    word = re.sub(r"[^\x00-\x7F]+", "", word)
    word = re.sub(r" +", " ", word).strip()
    return word


def score_to_tier(score: float) -> int:
    if score > 0.92:
        return 4
    elif score > 0.78:
        return 3
    elif score > 0.62:
        return 2
    elif score > 0.48:
        return 1
    return 0


def build_token_highlights(attentions, enc):
    input_ids = enc["input_ids"][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    real_len = sum(enc["attention_mask"][0].cpu().tolist())

    seq_len = attentions[0][0].shape[-1]
    rollout = torch.eye(seq_len).to(attentions[0][0].device)
    for layer_attn in attentions:
        avg_heads = layer_attn[0].mean(dim=0)
        aug = avg_heads + torch.eye(seq_len).to(avg_heads.device)
        aug = aug / aug.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(aug, rollout)

    token_importance = rollout[0].cpu().numpy()
    tokens = tokens[1:real_len - 1]
    scores = token_importance[1:real_len - 1]

    ranks = np.argsort(np.argsort(scores))
    scores = ranks / (len(ranks) - 1 + 1e-8)

    result = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        score = float(scores[i])
        word = tok.replace("\u0120", "").replace("\u2581", "")

        while (
            i + 1 < len(tokens)
            and not tokens[i + 1].startswith("\u0120")
            and not tokens[i + 1].startswith("\u2581")
        ):
            i += 1
            word += tokens[i].replace("\u0120", "").replace("\u2581", "")
            score = max(score, float(scores[i]))

        word = clean_word(word)
        if word:
            result.append({
                "word": word,
                "tier": score_to_tier(score),
                "leading_space": tok.startswith("\u0120") or tok.startswith("\u2581"),
            })
        i += 1

    return result


@torch.no_grad()
def compute_stat_features(text: str):
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(text)
    sent_var = float(np.var([len(s.split()) for s in sents])) if len(sents) > 1 else 0.0
    loss = 0.0
    if len(text.split()) >= 2:
        enc = gpt2_tok(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(device)
        loss = gpt2_mod(**enc, labels=enc.input_ids).loss.item()
    return np.array([[loss, sent_var, -loss]], dtype=np.float32)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class AnalyzeRequest(BaseModel):
    text: str
    model: str = "phase1"


@app.post("/analyze")
@torch.no_grad()
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    enc = tokenizer(
        text, truncation=True, max_length=512, padding=True, return_tensors="pt"
    ).to(device)

    if req.model == "phase2" and fusion_head is not None:
        roberta_out = roberta.roberta(**enc, output_attentions=True)
        cls_emb = roberta_out.last_hidden_state[:, 0, :].cpu()
        attentions = roberta_out.attentions
        stat_raw = compute_stat_features(text)
        stat_norm = (stat_raw - stat_raw.mean(0)) / (stat_raw.std(0) + 1e-8)
        logits = fusion_head(
            cls_emb.to(device),
            torch.tensor(stat_norm, dtype=torch.float32).to(device),
        )
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    else:
        out = roberta(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        attentions = out.attentions

    label = "FAKE" if probs.argmax() == 1 else "REAL"
    tokens = build_token_highlights(attentions, enc)

    return {
        "label": label,
        "prob_real": round(float(probs[0]), 4),
        "prob_fake": round(float(probs[1]), 4),
        "tokens": tokens,
    }


@app.post("/analyze_stream")
def analyze_stream(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    with torch.no_grad():
        enc = tokenizer(
            text, truncation=True, max_length=512, padding=True, return_tensors="pt"
        ).to(device)

        if req.model == "phase2" and fusion_head is not None:
            roberta_out = roberta.roberta(**enc, output_attentions=True)
            cls_emb = roberta_out.last_hidden_state[:, 0, :].cpu()
            attentions = roberta_out.attentions
            stat_raw = compute_stat_features(text)
            stat_norm = (stat_raw - stat_raw.mean(0)) / (stat_raw.std(0) + 1e-8)
            logits = fusion_head(
                cls_emb.to(device),
                torch.tensor(stat_norm, dtype=torch.float32).to(device),
            )
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        else:
            out = roberta(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
            attentions = out.attentions

        label = "FAKE" if probs.argmax() == 1 else "REAL"
        tokens = build_token_highlights(attentions, enc)

    async def event_generator():
        verdict_data = {
            "type": "verdict",
            "label": label,
            "prob_real": round(float(probs[0]), 4),
            "prob_fake": round(float(probs[1]), 4),
        }
        yield json.dumps(verdict_data) + "\n\n"
        
        for tok in tokens:
            await asyncio.sleep(0.04)
            tok_data = {
                "type": "token",
                "token": tok
            }
            yield json.dumps(tok_data) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}