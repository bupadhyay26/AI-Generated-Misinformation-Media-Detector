"""
detector.py  -  the "brain" of the project. Three small parts:

  1. check_input_type() : is the text a personal message, or a claim/news?  (simple rules)
  2. ml_style_score()   : TF-IDF + Logistic Regression writing-style score (long text only)
  3. make_verdict()     : turns the evidence found by the APIs into one final answer
"""
import re
import joblib
from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "model.joblib"
METRICS_PATH = BASE / "model" / "metrics.json"
MIN_WORDS_FOR_ML = 40  # model is trained on full articles, so it is NOT used on short statements

# ---------------------------------------------------------------- text cleaning
# (same function is used in train.py and here, so training and prediction match)
STOP = set(ENGLISH_STOP_WORDS) | {"reuters", "getty", "image", "images", "video", "photo", "featured", "via", "wire"}
DATELINE = re.compile(r"^\s*[A-Za-z0-9][A-Za-z0-9 .,'/-]{0,80}?\([A-Za-z .&]{2,20}\)\s*[-\u2013\u2014]\s*")


def clean_text(text):
    text = DATELINE.sub("", text or "", count=1)  # removes the "WASHINGTON (Reuters) - " shortcut
    words = re.findall(r"[a-z]{2,}", text.lower())
    return " ".join(w for w in words if w not in STOP)


def keywords(text, n=5):
    """Content words for search queries, ranked most-distinctive-first:
    1) tokens containing a digit (flight numbers, case numbers, years - e.g. 'FZ1073')
       are maximally specific to one story, so they always come first when present.
    2) other capitalized words (likely names/places, e.g. 'Yogi Adityanath', 'Flydubai')
    3) everything else, in sentence order.
    Without digit-priority, a claim naming a flight number could still end up querying
    only generic words like 'uae attorney general' if the number happened to appear
    later in the sentence than other capitalized words - missing the one term that
    actually pins the story down."""
    seen, numeric, proper, other = set(), [], [], []
    for w in re.findall(r"[A-Za-z0-9]{3,}", text or ""):
        low = w.lower()
        if low in STOP or low in seen:
            continue
        seen.add(low)
        if any(ch.isdigit() for ch in w):
            numeric.append(low)
        elif w[0].isupper():
            proper.append(low)
        else:
            other.append(low)
    return (numeric + proper + other)[:n]


# ---------------------------------------------------------------- 1. input type
PERSONAL_PATTERNS = [re.compile(p, re.I) for p in [
    r"^\s*(hi|hello|hey|dear|respected|good\s+(morning|afternoon|evening)|thanks|thank\s+you)\b",
    r"\b(i\s+am|i'm|i\s+will|i'll|i\s+can't|i\s+cannot|i\s+won't|i\s+could\s+not)\b.{0,40}\b(unable|not\s+able|late|absent|sick|leave|attend|come|coming|submit|join|busy)\b",
    r"\b(unable|not\s+able)\s+to\s+(attend|come|join|submit)\b",
    r"\bmy\s+(class|lecture|assignment|project|exam|lab|teacher|sir|professor)\b",
    r"\b(please|kindly|can\s+you|could\s+you)\b.{0,30}\b(send|share|give|help|tell|grant|check)\b",
    r"\b(leave|permission)\s+(for|application)\b",
    r"^\s*(how\s+are\s+you|what\s+are\s+you\s+doing|where\s+are\s+you)\b",
    r"\b(mai|main|mujhe|mere)\b.{0,30}\b(nahi|nhi)\b",
]]


def check_input_type(text):
    """Returns personal/news using trained classifier when available.
    Fallback regex is used only before the classifier is trained."""
    raw = (text or "").strip()
    if not raw:
        return "personal"
    model_path = Path(__file__).parent / "model" / "input_type_model.joblib"
    if model_path.exists() and len(raw.split()) >= 4:
        try:
            model = joblib.load(model_path)
            label = int(model.predict([raw])[0])
            return "claim" if label == 1 else "personal"
        except Exception:
            pass
    if len(raw.split()) <= 60 and any(p.search(raw) for p in PERSONAL_PATTERNS):
        return "personal"
    return "claim"


# ---------------------------------------------------------------- claim type + place
EARTHQUAKE = re.compile(r"\b(earthquakes?|quakes?|tremors?|seismic)\b", re.I)
HEAVY_RAIN = re.compile(r"\b(heavy\s+rain\w*|rainfall|floods?|flooding|waterlogg\w*|cloudburst|downpour)\b", re.I)
CLOSURE = re.compile(r"\b(closed|close|shut|holiday|suspended|curfew|declared)\b", re.I)
PLACE_SKIP = {"the", "a", "an", "this", "that", "these", "those", "case", "fact", "order", "due", "all", "our",
              "my", "your", "his", "her", "their", "its", "some", "any", "last", "next", "today",
              "tomorrow", "yesterday"}


def detect_event(text):
    if EARTHQUAKE.search(text):
        return "earthquake"
    if HEAVY_RAIN.search(text):
        return "rain"
    return "general"


def place_candidates(text):
    """Words that follow in/at/near/around, e.g. 'emergency in Lucknow due...' -> ['Lucknow']."""
    out = []
    for m in re.finditer(r"\b(?:in|at|near|around|across)\s+([A-Za-z]{3,})(?:\s+([A-Za-z]{3,}))?", text):
        first, second = m.group(1), m.group(2)
        if first.lower() in PLACE_SKIP:
            continue
        if second and second[0].isupper() and second.lower() not in PLACE_SKIP:
            out.append(f"{first} {second}")  # e.g. "New Delhi"
        if first.lower() != "new":  # "new" alone is not a place
            out.append(first)
    return out


# ---------------------------------------------------------------- 2. ML style score
@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)  # {"vectorizer": ..., "model": ...}


def ml_style_score(text):
    """Writing-style score. Returns None for short text (the model is unreliable there)."""
    if len((text or "").split()) < MIN_WORDS_FOR_ML:
        return None
    bundle = load_model()
    X = bundle["vectorizer"].transform([clean_text(text)])
    p_real = float(bundle["model"].predict_proba(X)[0][1])  # class 1 = REAL
    row = X.tocoo()
    terms = bundle["vectorizer"].get_feature_names_out()
    contrib = sorted(zip(row.col, row.data * bundle["model"].coef_[0][row.col]), key=lambda t: -abs(t[1]))[:6]
    return {
        "label": "REAL" if p_real >= 0.5 else "FAKE",
        "prob_real": p_real,
        "confidence": max(p_real, 1 - p_real),
        "top_terms": [(terms[c], float(v)) for c, v in contrib],  # + pushes to REAL, - pushes to FAKE
    }


# ---------------------------------------------------------------- 3. final verdict
def make_verdict(kind, evidence, ml=None):
    """
    Factual verdict rules:
    - Personal message -> NOT NEWS
    - Relevant fact-check evidence can establish REAL or FAKE
    - News coverage alone is not proof
    - Missing, mixed or weak evidence -> UNVERIFIED
    """

    if kind == "personal":
        return {
            "label": "NOT NEWS",
            "reason": "This appears to be a personal message, not a news claim."
        }

    fact_checks = [
        e for e in evidence
        if e.get("source") == "Google Fact Check API"
        and e.get("status") in ("supports", "contradicts")
    ]

    positive = [e for e in fact_checks if e["status"] == "supports"]
    negative = [e for e in fact_checks if e["status"] == "contradicts"]

    # Conflicting fact-check results must not produce a firm verdict
    if positive and negative:
        return {
            "label": "UNVERIFIED",
            "reason": "Relevant fact-check results conflict. Further verification is needed."
        }

    if negative:
        details = "; ".join(e.get("detail", "") for e in negative)
        return {
            "label": "FAKE",
            "reason": "A matching fact-check source contradicts this claim: " + details
        }

    if positive:
        details = "; ".join(e.get("detail", "") for e in positive)
        return {
            "label": "REAL",
            "reason": "A matching fact-check source supports this claim: " + details
        }

    # News articles, writing style and AI authorship do not prove truth.
    supporting_sources = [
        e for e in evidence
        if e.get("status") == "supports"
    ]

    if supporting_sources:
        names = ", ".join(
            sorted({e.get("source", "Unknown source") for e in supporting_sources})
        )
        reason = (
            f"Some evidence was found from {names}, "
            "but it does not independently establish the full claim."
        )
    else:
        reason = (
            "No sufficiently strong, matching fact-check evidence was found. "
            "The claim cannot safely be labelled REAL or FAKE."
        )

    return {
        "label": "UNVERIFIED",
        "reason": reason
    }