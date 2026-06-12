import os
import time
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LABEL_NAMES = {0: "positive ✅", 1: "negative ❌"}

SAMPLE_TEXTS = [
    "Отличный врач, всё объяснил понятно и без спешки. Очень доволен приёмом!",
    "Ужасная очередь, грубый персонал, ждал три часа и так и не попал к врачу.",
    "Поликлиника как поликлиника. Записался через портал, приняли без проблем.",
]


# ── Model definitions (must mirror work.ipynb) ─────────────────────────────

def simple_tokenize(text: str) -> list[str]:
    return text.lower().split()


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_classes=2, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, (h, _) = self.lstm(emb)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))


# ── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_resource
def load_logreg():
    path = os.path.join(MODELS_DIR, "logreg_tfidf.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_lstm():
    vocab_path = os.path.join(MODELS_DIR, "lstm_vocab.pkl")
    model_path = os.path.join(MODELS_DIR, "lstm_model.pt")
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        return None, None
    with open(vocab_path, "rb") as f:
        word2idx = pickle.load(f)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = LSTMClassifier(
        ckpt["vocab_size"], ckpt["embed_dim"], ckpt["hidden_dim"],
        ckpt["num_classes"], ckpt["num_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, word2idx


@st.cache_resource
def load_bert():
    bert_dir = os.path.join(MODELS_DIR, "rubert_finetuned")
    if not os.path.exists(bert_dir):
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
    model.eval()
    return model, tokenizer


# ── Inference helpers ───────────────────────────────────────────────────────

def predict_logreg(bundle, text: str):
    tfidf, model = bundle["tfidf"], bundle["model"]
    t0 = time.perf_counter()
    vec = tfidf.transform([text])
    proba = model.predict_proba(vec)[0]
    label = int(model.predict(vec)[0])
    elapsed = (time.perf_counter() - t0) * 1000
    return label, float(proba[label]), elapsed


MAX_LEN_LSTM = 256


def predict_lstm(model, word2idx, text: str):
    tokens = simple_tokenize(text)[:MAX_LEN_LSTM]
    ids = [word2idx.get(t, 1) for t in tokens]
    ids += [0] * (MAX_LEN_LSTM - len(ids))
    x = torch.tensor([ids], dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
    elapsed = (time.perf_counter() - t0) * 1000
    proba = torch.softmax(logits, dim=1)[0].numpy()
    label = int(logits.argmax(dim=1).item())
    return label, float(proba[label]), elapsed


def predict_bert(model, tokenizer, text: str, output_attentions=False):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=output_attentions)
    elapsed = (time.perf_counter() - t0) * 1000
    proba = torch.softmax(outputs.logits, dim=1)[0].numpy()
    label = int(outputs.logits.argmax(dim=1).item())
    return label, float(proba[label]), elapsed, outputs, inputs


# ── UI ──────────────────────────────────────────────────────────────────────

st.title("🏥 Классификация отзывов на медучреждения")
st.markdown(
    "Сравнение трёх подходов к анализу тональности: "
    "**TF-IDF + LogReg** · **LSTM** · **rubert-tiny2**"
)

# Load models
logreg_bundle = load_logreg()
lstm_model, word2idx = load_lstm()
bert_model, bert_tokenizer = load_bert()

missing = [
    name for name, ok in [
        ("TF-IDF + LogReg", logreg_bundle is not None),
        ("LSTM", lstm_model is not None),
        ("rubert-tiny2", bert_model is not None),
    ] if not ok
]
if missing:
    st.warning(
        f"Модели не найдены: **{', '.join(missing)}**. "
        "Сначала запустите ноутбук `page1_reviews/work.ipynb` целиком."
    )

# ── Metrics table ────────────────────────────────────────────────────────────
metrics_path = os.path.join(MODELS_DIR, "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    st.subheader("Результаты обучения")
    m_df = pd.DataFrame([
        {
            "Модель": metrics["logreg"]["name"],
            "F1-macro": round(metrics["logreg"]["f1_macro"], 4),
            "Инференс (мс/сэмпл)": round(metrics["logreg"]["inf_time_ms"], 3),
        },
        {
            "Модель": metrics["lstm"]["name"],
            "F1-macro": round(metrics["lstm"]["f1_macro"], 4),
            "Инференс (мс/сэмпл)": round(metrics["lstm"]["inf_time_ms"], 3),
        },
        {
            "Модель": metrics["bert"]["name"],
            "F1-macro": round(metrics["bert"]["f1_macro"], 4),
            "Инференс (мс/сэмпл)": round(metrics["bert"]["inf_time_ms"], 3),
        },
    ])
    st.dataframe(m_df, use_container_width=True, hide_index=True)

    comparison_img = os.path.join(MODELS_DIR, "comparison.png")
    if os.path.exists(comparison_img):
        st.image(comparison_img, use_container_width=True)

st.divider()

# ── Text input ───────────────────────────────────────────────────────────────
st.subheader("Попробовать на своём тексте")

sample_choice = st.selectbox(
    "Вставить пример:", ["— введите свой текст —"] + SAMPLE_TEXTS, index=0
)
default_text = "" if sample_choice.startswith("—") else sample_choice
user_text = st.text_area("Текст отзыва:", value=default_text, height=120)

predict_clicked = st.button("Предсказать", type="primary", disabled=not user_text.strip())

if predict_clicked and user_text.strip():
    results = []

    if logreg_bundle:
        label, conf, ms = predict_logreg(logreg_bundle, user_text)
        results.append({
            "Модель": "TF-IDF + LogReg",
            "Тональность": LABEL_NAMES[label],
            "Уверенность": f"{conf:.1%}",
            "Время (мс)": f"{ms:.1f}",
        })

    if lstm_model is not None:
        label, conf, ms = predict_lstm(lstm_model, word2idx, user_text)
        results.append({
            "Модель": "LSTM",
            "Тональность": LABEL_NAMES[label],
            "Уверенность": f"{conf:.1%}",
            "Время (мс)": f"{ms:.1f}",
        })

    if bert_model is not None:
        label, conf, ms, _, _ = predict_bert(bert_model, bert_tokenizer, user_text)
        results.append({
            "Модель": "rubert-tiny2",
            "Тональность": LABEL_NAMES[label],
            "Уверенность": f"{conf:.1%}",
            "Время (мс)": f"{ms:.1f}",
        })

    if results:
        st.subheader("Результаты")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
