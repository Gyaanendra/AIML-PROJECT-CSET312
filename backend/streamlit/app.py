import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2TokenizerFast, GPT2LMHeadModel
import nltk

nltk.download('punkt', quiet=True)

# ------------------- Config & Theme -------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .real { color: #4ade80; font-size: 2.2rem; font-weight: bold; }
    .fake { color: #ff4d4d; font-size: 2.2rem; font-weight: bold; }

    /* Attention highlight tiers */
    .attn-s4 {
        background: #0fa0d8; color: #fff;
        border-radius: 3px; padding: 1px 4px;
    }
    .attn-s3 {
        background: #38bfe8; color: #fff;
        border-radius: 3px; padding: 1px 4px;
    }
    .attn-s2 {
        background: #7dd8f5; color: #0a4a5c;
        border-radius: 3px; padding: 1px 4px;
    }
    .attn-s1 {
        background: #bdedfb; color: #0a4a5c;
        border-radius: 3px; padding: 1px 4px;
    }

    /* Legend swatches */
    .legend-row {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 0.75rem;
        font-size: 13px;
        color: #aaa;
    }
    .swatch {
        display: inline-block;
        width: 20px; height: 14px;
        border-radius: 3px;
    }

    /* Token display box */
    .token-display {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        font-size: 15px;
        line-height: 2;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detector")
st.caption("RoBERTa + GPT-2 Fusion Head | Attention Token Highlighting")

# ------------------- Model Paths (Change if needed) -------------------
PHASE1_DIR = r"H:\CSET312_project\v3\models\phase1_roberta_fulltune\best"
PHASE2_HEAD = r"H:\CSET312_project\v3\models\phase2_fusion_head\fusion_head.pt"

# ------------------- Load Models -------------------
@st.cache_resource
def load_models():
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
        fusion_head.load_state_dict(ckpt['head_state'])
        fusion_head.eval()

    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_mod = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_mod.eval()

    return tokenizer, roberta, fusion_head, gpt2_tok, gpt2_mod, device

tokenizer, roberta, fusion_head, gpt2_tok, gpt2_mod, device = load_models()

# ------------------- Prediction Functions -------------------
@torch.no_grad()
def predict_phase1(text):
    enc = tokenizer(text, truncation=True, max_length=512, padding=True, return_tensors='pt').to(device)
    out = roberta(**enc)
    probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
    attentions = out.attentions  # tuple of (batch, heads, seq, seq)
    return {
        "label": "FAKE" if probs.argmax() == 1 else "REAL",
        "prob_real": float(probs[0]),
        "prob_fake": float(probs[1]),
        "attentions": attentions,
        "enc": enc
    }

@torch.no_grad()
def compute_stat_features(text):
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(text)
    sent_var = float(np.var([len(s.split()) for s in sents])) if len(sents) > 1 else 0.0

    valid = len(text.split()) >= 2
    loss = 0.0
    if valid:
        enc = gpt2_tok(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
        loss = gpt2_mod(**enc, labels=enc.input_ids).loss.item()

    return np.array([[loss, sent_var, -loss]], dtype=np.float32)

@torch.no_grad()
def predict_phase2(text):
    if fusion_head is None:
        return {"label": "ERROR", "prob_real": 0.0, "prob_fake": 0.0, "attentions": None, "enc": None}

    enc = tokenizer(text, truncation=True, max_length=512, padding=True, return_tensors='pt').to(device)
    roberta_out = roberta.roberta(**enc, output_attentions=True)
    cls_emb = roberta_out.last_hidden_state[:, 0, :].cpu()
    attentions = roberta_out.attentions

    stat_raw = compute_stat_features(text)
    stat_norm = (stat_raw - stat_raw.mean(0)) / (stat_raw.std(0) + 1e-8)

    logits = fusion_head(cls_emb.to(device), torch.tensor(stat_norm, dtype=torch.float32).to(device))
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return {
        "label": "FAKE" if probs.argmax() == 1 else "REAL",
        "prob_real": float(probs[0]),
        "prob_fake": float(probs[1]),
        "attentions": attentions,
        "enc": enc
    }

# ------------------- Attention-Based Token Highlighting -------------------
import re

def clean_word(word):
    """Remove encoding artifacts from individual tokens."""
    word = word.replace("\u00e2\u0080\u0099", "'")
    word = word.replace("\u00e2\u0080\u009c", '"')
    word = word.replace("\u00e2\u0080\u009d", '"')
    word = word.replace("\u010c\u201c", '"')
    word = word.replace("\u010c\u2122", "'")
    word = word.replace("\u00e2\u0122\u017e", '"')
    word = word.replace("\u00e2\u0122\u00be", '"')
    word = word.replace("\u00e2\u0122\u0080", '...')
    word = word.replace("\u00e2\u0122\u0081", '-')
    word = word.replace("\u00c4\u0141", "g")
    word = word.replace("\u00c4\u00b1", "i")
    # Remove any remaining non-ASCII garbage
    word = re.sub(r'[^\x00-\x7F]+', '', word)
    word = re.sub(r' +', ' ', word).strip()
    return word


def score_to_class(score):
    """Only highlight genuinely high-scoring tokens."""
    if score > 0.92:
        return "attn-s4"
    elif score > 0.78:
        return "attn-s3"
    elif score > 0.62:
        return "attn-s2"
    elif score > 0.48:
        return "attn-s1"
    return ""


def highlight_tokens_attention(text, attentions, enc):
    """
    Produce HTML with attention rollout + rank-normalized highlighting.
    Guarantees visual spread across all tokens, with clean word rendering.
    """
    if attentions is None:
        return f"<div class='token-display'>{text}</div>"

    input_ids = enc['input_ids'][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    attention_mask = enc['attention_mask'][0].cpu().tolist()
    real_len = sum(attention_mask)

    # ── Attention Rollout ──────────────────────────────────────────────
    seq_len = attentions[0][0].shape[-1]
    rollout = torch.eye(seq_len).to(attentions[0][0].device)

    for layer_attn in attentions:
        avg_heads = layer_attn[0].mean(dim=0)          # (seq, seq)
        aug = avg_heads + torch.eye(seq_len).to(avg_heads.device)
        aug = aug / aug.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(aug, rollout)

    token_importance = rollout[0].cpu().numpy()        # CLS row

    # Drop CLS (idx 0) and SEP (idx real_len-1)
    tokens = tokens[1:real_len - 1]
    scores = token_importance[1:real_len - 1]

    # ── Rank-based normalization ───────────────────────────────────────
    ranks = np.argsort(np.argsort(scores))
    scores = ranks / (len(ranks) - 1 + 1e-8)

    # ── Merge subword tokens & build HTML ─────────────────────────────
    html_parts = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        score = float(scores[i])

        word = tok.replace("Ġ", "").replace("▁", "")
        while i + 1 < len(tokens) and not tokens[i + 1].startswith("Ġ") and not tokens[i + 1].startswith("▁"):
            i += 1
            word += tokens[i].replace("Ġ", "").replace("▁", "")
            score = max(score, float(scores[i]))

        # Clean encoding artifacts
        word = clean_word(word)

        # Skip empty tokens after cleaning
        if not word:
            i += 1
            continue

        leading_space = " " if tok.startswith("Ġ") or tok.startswith("▁") else ""
        css_class = score_to_class(score)

        if css_class:
            html_parts.append(f"{leading_space}<span class='{css_class}'>{word}</span>")
        else:
            html_parts.append(f"{leading_space}{word}")

        i += 1

    body = "".join(html_parts).strip()

    legend_html = """
    <div class="legend-row">
        <span>Strong Attribution</span>
        <span class="swatch" style="background:#0fa0d8;"></span>
        <span class="swatch" style="background:#38bfe8;"></span>
        <span class="swatch" style="background:#7dd8f5;"></span>
        <span class="swatch" style="background:#bdedfb;"></span>
        <span>Weak Attribution</span>
    </div>
    """

    return f"<div class='token-display'>{body}</div>{legend_html}"

# ------------------- Main UI -------------------
st.subheader("🔍 Real-Time Fake News Detection")

text_input = st.text_area(
    "Enter news text / article",
    height=180,
    value="US Centcom says Hormuz blockade will begin Monday and will only apply to Iranian ports; China will be most affected..."
)

col1, col2 = st.columns([1, 2])
with col1:
    model_choice = st.radio("Model", ["Phase 1 (RoBERTa)", "Phase 2 (Fusion Head)"], horizontal=True)

with col2:
    if st.button("🚀 Analyze", type="primary", use_container_width=True):
        if text_input.strip():
            with st.spinner("Analyzing with your trained models..."):
                if model_choice.startswith("Phase 1"):
                    result = predict_phase1(text_input)
                else:
                    result = predict_phase2(text_input)

                # ── Result label ──
                st.markdown(
                    f"<p class='{result['label'].lower()}'>"
                    f"{result['label']} — {max(result['prob_real'], result['prob_fake']):.1%}"
                    f"</p>",
                    unsafe_allow_html=True
                )

                col_a, col_b = st.columns(2)
                col_a.metric("P(REAL)", f"{result['prob_real']:.1%}")
                col_b.metric("P(FAKE)", f"{result['prob_fake']:.1%}")

                # ── Attention Token Highlighting ──
                st.subheader("🔤 Attention Token Highlighting")
                highlighted_html = highlight_tokens_attention(
                    text_input,
                    result.get("attentions"),
                    result.get("enc")
                )
                st.markdown(highlighted_html, unsafe_allow_html=True)

        else:
            st.warning("Please enter some text to analyze.")

# ------------------- Sidebar Info -------------------
st.sidebar.header("About")
st.sidebar.info("""
This app uses your actual trained models:
- **Phase 1**: Fine-tuned RoBERTa
- **Phase 2**: Fusion Head (RoBERTa + GPT-2 stats)

Token highlighting uses **real attention weights** from the last RoBERTa layer (CLS row, averaged across heads).

Color intensity = attention strength:
- 🔵 Dark teal → strong attribution
- 🔵 Light teal → weak attribution
""")

st.sidebar.caption("Monotone Dark Theme • Attention-Based Highlighting")