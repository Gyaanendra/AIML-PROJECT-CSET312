import re
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import asyncio
import os
import requests
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from duckduckgo_search import DDGS
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
)
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

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

app.mount("/static", StaticFiles(directory="static"), name="static")
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

# @app.get("/", response_class=HTMLResponse)
@app.get("/")
async def index(request: Request):
    # return templates.TemplateResponse("index.html", {"request": request})
    return "Hello project is running"


class AnalyzeRequest(BaseModel):
    text: str
    model: str = "phase2"
    # 'model' = ML only | 'agent' = LLM agent only | 'both' = full pipeline
    agent_mode: str = "both"


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
async def analyze_stream(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    def run_local_ml():
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
            return label, probs, tokens

    def search_web_claims(claims):
        try:
            gnews_key = os.getenv("GNEWS_API_KEY")
            query = urllib.parse.quote(claims.strip())
            url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max=3&apikey={gnews_key}"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                results = ""
                for article in data.get("articles", []):
                    results += f"Title: {article['title']}\nSnippet: {article['description']}\n\n"
                return results if results else "No strict verifications found."
            return f"Search API returned {res.status_code}"
        except Exception as e:
            return f"Search failed: {str(e)}"

    agent_mode = req.agent_mode  # 'model' | 'agent' | 'both'

    async def event_generator():
        # ── MODE: model only ────────────────────────────────────────────────────
        if agent_mode == "model":
            yield json.dumps({"type": "status", "message": "Cross-referencing with Custom RoBERTa Backbone..."}) + "\n\n"
            label, probs, tokens_result = await asyncio.to_thread(run_local_ml)
            verdict_data = {
                "type": "verdict",
                "label": label,
                "prob_real": round(float(probs[0]), 4),
                "prob_fake": round(float(probs[1]), 4),
            }
            yield json.dumps(verdict_data) + "\n\n"
            for tok in tokens_result:
                await asyncio.sleep(0.01)
                yield json.dumps({"type": "token", "token": tok}) + "\n\n"
            return

        # ── Shared LLM client (used by 'agent' and 'both') ──────────────────────
        client = ChatNVIDIA(
            model="google/gemma-3n-e2b-it",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.8,
            top_p=0.9,
            max_completion_tokens=2048,
        )

        yield json.dumps({"type": "status", "message": "Extracting search parameters..."}) + "\n\n"
        extract_prompt = f"Extract 2 to 4 crucial keywords identifying the core event from this text to use as a search query. Output ONLY the keywords separated by spaces. DO NOT use bullet points or extra text:\n\n{text[:1500]}"
        claims_res = await asyncio.to_thread(client.invoke, [{"role": "user", "content": extract_prompt}])
        claims = claims_res.content

        yield json.dumps({"type": "status", "message": "Verifying facts against live web data..."}) + "\n\n"
        web_context = await asyncio.to_thread(search_web_claims, claims)

        # ── MODE: both — also run ML scoring ───────────────────────────────────
        label, probs, tokens_result = None, None, []
        if agent_mode == "both":
            yield json.dumps({"type": "status", "message": "Cross-referencing with Custom RoBERTa Backbone..."}) + "\n\n"
            label, probs, tokens_result = await asyncio.to_thread(run_local_ml)

        yield json.dumps({"type": "status", "message": "Synthesizing Threat Intelligence..."}) + "\n\n"

        # Emit verdict + token highlights only when ML ran
        if agent_mode == "both" and probs is not None:
            verdict_data = {
                "type": "verdict",
                "label": label,
                "prob_real": round(float(probs[0]), 4),
                "prob_fake": round(float(probs[1]), 4),
            }
            yield json.dumps(verdict_data) + "\n\n"
            for tok in tokens_result:
                await asyncio.sleep(0.01)
                yield json.dumps({"type": "token", "token": tok}) + "\n\n"

        # Build the LLM final prompt
        web_has_hits = (
            "No strict verifications found" not in web_context
            and "Search failed" not in web_context
        )

        if agent_mode == "both" and probs is not None:
            ml_conf = max(probs[0], probs[1])
            ml_section = f"""## ML STRUCTURAL ANALYSIS (40% weight)
- Prediction: {label} | Confidence: {ml_conf:.2%}
- Note: Trained on pre-2024 data — moderate caution for highly recent breaking events.

## REASONING RULES
1. Web evidence CONFIRMS claims → lean REAL, use ML as a confidence booster.
2. Web evidence CONTRADICTS claims → lean FAKE regardless of ML output.
3. Web evidence is ABSENT but ML confidence is HIGH (>85%) → lean toward ML verdict, note temporal lag.
4. Both signals agree → high-confidence verdict in that direction.
5. Signals conflict with low confidence → INCONCLUSIVE."""
            weight_note = "- Live Web Verification (GNews): 60% weight — PRIMARY signal.\n- Custom Trained ML Model (RoBERTa + Fusion Head): 40% weight — SECONDARY signal."
        else:
            ml_section = "## NOTE\nNo ML model output available — base your verdict solely on the live web evidence."
            weight_note = "- Live Web Verification (GNews): 100% weight — sole signal for this analysis."

        final_prompt = f"""You are VIGIL-AI, an advanced threat intelligence system determining if a news article is REAL or FAKE.

## SCORING WEIGHTS
{weight_note}

## TARGET TEXT
"{text[:900]}"

## LIVE WEB EVIDENCE
{"CORROBORATING ARTICLES FOUND:" if web_has_hits else "NO CORROBORATING EVIDENCE FOUND in live news sources."}
{web_context}

{ml_section}

Write 2-3 sentences of weighted analysis, then conclude:
**VERDICT: [REAL / FAKE / INCONCLUSIVE]** — one decisive bottom-line sentence.
"""

        async for chunk in client.astream([{"role": "user", "content": final_prompt}]):
            if hasattr(chunk, "additional_kwargs") and "reasoning_content" in chunk.additional_kwargs:
                reasoning = chunk.additional_kwargs["reasoning_content"]
                if reasoning:
                    yield json.dumps({"type": "llm_chunk", "reasoning": reasoning}) + "\n\n"
            if chunk.content:
                yield json.dumps({"type": "llm_chunk", "content": chunk.content}) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}