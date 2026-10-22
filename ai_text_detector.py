"""Trained AI-vs-human text detector.
Model: word + character TF-IDF features + Logistic Regression.
The detector is a separate signal from factual REAL/FAKE verification.
"""
from __future__ import annotations
import re
from pathlib import Path
import joblib
from scipy.sparse import hstack

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "ai_text_model.joblib"
MIN_WORDS = 25

def analyze_ai_text(text: str) -> dict:
    raw = text or ""
    n = len(re.findall(r"[A-Za-z']+", raw))
    if n < MIN_WORDS:
        return {"available": False, "reason": f"At least {MIN_WORDS} words are recommended for AI-text screening."}
    if not MODEL_PATH.exists():
        return {"available": False, "reason": "AI-text model is not trained yet. Run train_ai_text.py first."}
    bundle = joblib.load(MODEL_PATH)
    word = bundle["word_vectorizer"].transform([raw])
    char = bundle["char_vectorizer"].transform([raw])
    X = hstack([word, char])
    p_ai = float(bundle["model"].predict_proba(X)[0][1])
    label = "AI-GENERATED" if p_ai >= 0.5 else "HUMAN-WRITTEN"
    return {
        "available": True,
        "label": label,
        "ai_probability": p_ai if label == "AI-GENERATED" else 1-p_ai,
        "word_count": n,
        "sentence_count": len(re.findall(r"[.!?]+", raw)) or 1,
        "warning": "This is a probabilistic classifier; AI authorship cannot be proven from writing style alone.",
    }
