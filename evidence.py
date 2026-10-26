"""
evidence.py  -  talks to the 4 free APIs and runs the full pipeline (analyze).

Every check returns one dict:  {"source", "status", "detail", "url"}
status = supports | contradicts | partial | no_data | error | skipped
Nothing here ever crashes: a network/API problem just becomes status "error".

  USGS Earthquake API   -> real earthquake data      (no key needed)
  Open-Meteo API        -> real rainfall data        (no key needed)
  Google Fact Check API -> claims already fact-checked by newsrooms (free key)
  NewsAPI               -> do real news outlets report it?          (free key)
"""
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from detector import (CLOSURE, check_input_type, detect_event, keywords, make_verdict,
                      ml_style_score, place_candidates)

TIMEOUT = 8
HEAVY_RAIN_MM = 64.5      # IMD "heavy rain" starts at 64.5 mm in 24 hours
QUAKE_RADIUS_KM = 300
QUAKE_MIN_MAG = 4.0       # USGS catalogue is reliable for M4+ worldwide
KEYS_FILE = Path(__file__).parent / "api_keys.txt"


def _ev(source, status, detail, url=""):
    return {"source": source, "status": status, "detail": detail, "url": url}


def load_keys():
    """Load API keys from local file, Streamlit Secrets, or environment variables."""
    keys = {}

    # 1. Read local api_keys.txt, if available
    if KEYS_FILE.exists():
        for line in KEYS_FILE.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            if "=" in line and not line.strip().startswith("#"):
                name, value = line.split("=", 1)
                if value.strip():
                    keys[name.strip()] = value.strip()

    # 2. Read Streamlit Community Cloud Secrets
    try:
        import streamlit as st

        for name in ("NEWSAPI_KEY", "GOOGLE_FACTCHECK_KEY"):
            if not keys.get(name):
                try:
                    value = st.secrets.get(name, "")
                    if value:
                        keys[name] = str(value).strip()
                except Exception:
                    pass
    except ImportError:
        pass

    # 3. Read environment variables as a fallback
    for name in ("NEWSAPI_KEY", "GOOGLE_FACTCHECK_KEY"):
        if not keys.get(name) and os.environ.get(name):
            keys[name] = os.environ[name].strip()

    return keys
# ---------------------------------------------------------------- place -> coordinates
def find_place(text):
    """Open-Meteo geocoding: 'lucknow' -> ((26.84, 80.94, 'Lucknow, India'), None).
    Returns (None, reason) if the place is not found or the geocoder cannot be reached."""
    for name in place_candidates(text):
        try:
            r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": name, "count": 1, "language": "en"}, timeout=TIMEOUT)
            results = r.json().get("results")
            if results:
                p = results[0]
                return (p["latitude"], p["longitude"], f"{p['name']}, {p.get('country', '')}".strip(", ")), None
        except Exception as exc:
            return None, f"could not look up the place name ({exc})"
    return None, "no place name found in the text, so real-time data was not checked"


# ---------------------------------------------------------------- API 1: USGS earthquakes
def check_earthquake(lat, lon, label):
    src = "USGS Earthquake API"
    start = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params={
            "format": "geojson", "starttime": start, "latitude": lat, "longitude": lon,
            "maxradiuskm": QUAKE_RADIUS_KM, "minmagnitude": QUAKE_MIN_MAG,
            "orderby": "magnitude", "limit": 5}, timeout=TIMEOUT)
        r.raise_for_status()
        quakes = r.json().get("features", [])
    except Exception as exc:
        return _ev(src, "error", f"Could not reach USGS ({exc})")
    if quakes:
        p = quakes[0]["properties"]
        return _ev(src, "supports", f"earthquake M{p.get('mag')} recorded near {p.get('place')} "
                   f"within {QUAKE_RADIUS_KM} km of {label} in the last 48 hours", p.get("url", ""))
    return _ev(src, "contradicts", f"no earthquake of M{QUAKE_MIN_MAG}+ recorded within {QUAKE_RADIUS_KM} km "
               f"of {label} in the last 48 hours")


# ---------------------------------------------------------------- API 2: Open-Meteo rainfall
def check_rain(lat, lon, label, closure=False):
    src = "Open-Meteo Weather API"
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "daily": "precipitation_sum",
            "past_days": 1, "forecast_days": 2, "timezone": "auto"}, timeout=TIMEOUT)
        r.raise_for_status()
        daily = r.json()["daily"]
        rain = [x or 0 for x in daily["precipitation_sum"]]
    except Exception as exc:
        return _ev(src, "error", f"Could not reach Open-Meteo ({exc})")
    peak = max(rain)
    span = f"yesterday to tomorrow for {label}"
    if peak >= HEAVY_RAIN_MM and closure:
        return _ev(src, "partial", f"heavy rain confirmed ({peak:.0f} mm/day, {span}), but the closure/holiday "
                   "order itself is not verified by weather data")
    if peak >= HEAVY_RAIN_MM:
        return _ev(src, "supports", f"heavy rain confirmed ({peak:.0f} mm/day, {span})")
    return _ev(src, "contradicts", f"peak rainfall only {peak:.0f} mm/day ({span}); heavy rain starts at {HEAVY_RAIN_MM} mm/day")


# ---------------------------------------------------------------- API 3: Google Fact Check
NEG = ("false", "fake", "misleading", "mislead", "incorrect", "hoax", "pants on fire", "fabricated", "baseless",
       "not true", "untrue", "no evidence", "scam", "satire", "wrong", "unfounded")
POS = ("true", "correct", "accurate", "genuine", "authentic", "real")


def _rating_kind(rating):
    r = rating.lower()
    if any(w in r for w in NEG):
        return "neg"
    if any(w in r for w in POS) and "half" not in r:
        return "pos"
    return "mix"


def check_fact_check(text, key):
    src = "Google Fact Check API"
    if not key:
        return _ev(src, "skipped", "no GOOGLE_FACTCHECK_KEY in api_keys.txt")
    try:
        r = requests.get("https://factchecktools.googleapis.com/v1alpha1/claims:search",
                         params={"query": text[:200], "key": key, "languageCode": "en", "pageSize": 5}, timeout=TIMEOUT)
        if r.status_code in (400, 403):
            return _ev(src, "error", "key rejected - check the key and that 'Fact Check Tools API' is enabled")
        r.raise_for_status()
        claims = r.json().get("claims", [])
    except Exception as exc:
        return _ev(src, "error", f"Could not reach Google Fact Check ({exc})")

    words = set(keywords(text, 12))
    found = []  # (kind, rating, publisher, url)
    for c in claims:
        claim_words = set(keywords(c.get("text", ""), 30))
        if not words or len(words & claim_words) / len(words) < 0.4:
            continue  # unrelated fact-check, ignore
        for rev in c.get("claimReview", [])[:1]:
            found.append((_rating_kind(rev.get("textualRating", "")), rev.get("textualRating", "?"),
                          rev.get("publisher", {}).get("name", "?"), rev.get("url", "")))
    if not found:
        return _ev(src, "no_data", "no matching fact-check found")
    neg = [f for f in found if f[0] == "neg"]
    pos = [f for f in found if f[0] == "pos"]
    top = (neg or pos or found)[0]
    detail = f"fact-checked by {top[2]} as '{top[1]}'"
    if neg and len(neg) >= len(pos):
        return _ev(src, "contradicts", detail, top[3])
    if pos and len(pos) > len(neg):
        return _ev(src, "supports", detail, top[3])
    return _ev(src, "partial", detail + " (mixed rating)", top[3])


# ---------------------------------------------------------------- API 4: NewsAPI
def check_news(text, key):
    """
    News coverage is a discovery signal, not proof of truth.
    This function never marks a claim as verified REAL.
    """

    src = "NewsAPI"

    if not key:
        return _ev(
            src, "skipped",
            "No NEWSAPI_KEY configured."
        )

    words = keywords(text, 5)

    if not words:
        return _ev(src, "no_data", "Not enough keywords to search.")

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": " AND ".join(words),
                "searchIn": "title,description",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 5,
                "apiKey": key
            },
            timeout=TIMEOUT
        )

        if response.status_code == 401:
            return _ev(src, "error", "NewsAPI key was rejected.")

        if response.status_code == 429:
            return _ev(src, "error", "NewsAPI request limit reached.")

        if response.status_code == 426:
            return _ev(src, "error", "NewsAPI plan restriction.")

        response.raise_for_status()
        articles = response.json().get("articles", [])

    except Exception as exc:
        return _ev(src, "error", f"Could not reach NewsAPI: {exc}")

    if not articles:
        return _ev(
            src, "no_data",
            "No matching articles found. Absence of coverage does not prove a claim is false."
        )

    claim_words = set(words)
    relevant = []

    for article in articles:
        article_text = (
            f"{article.get('title', '')} "
            f"{article.get('description', '')}"
        )

        article_words = set(keywords(article_text, 20))

        overlap = (
            len(claim_words & article_words) / len(claim_words)
            if claim_words else 0
        )

        if overlap >= 0.4:
            relevant.append(article)

    if not relevant:
        return _ev(
            src, "no_data",
            "Search results were not sufficiently relevant to the claim."
        )

    outlets = sorted({
        (article.get("source") or {}).get("name", "Unknown")
        for article in relevant
    })

    first = relevant[0]

    detail = (
        f"Found {len(relevant)} potentially relevant article(s) "
        f"from {', '.join(outlets[:3])}. Coverage alone does not verify truth."
    )

    return _ev(
        src, "partial", detail, first.get("url", "")
    )
    # Query with only the top 2-3 most distinctive keywords (proper nouns first, from
    # keywords()) rather than ANDing all 5: real coverage of the same story often uses
    # different wording for some of the claim's words (e.g. "axe" vs "crash axe",
    # "co-pilot" vs "pilot"), so requiring all 5 exact words silently excludes major
    # outlets. The broader net here is fine because the relevance filter below (which
    # checks against all 5 original words) still enforces precision afterwards.
    query_words = words[:2] if len(words) >= 2 else words
    try:
        r = requests.get("https://newsapi.org/v2/everything", params={
            "q": " AND ".join(query_words), "searchIn": "title,description", "language": "en",
            "sortBy": "relevancy", "pageSize": 20, "apiKey": key}, timeout=TIMEOUT)
        if r.status_code == 401:
            return _ev(src, "error", "key rejected by NewsAPI")
        if r.status_code == 429:
            return _ev(src, "error", "daily limit (100 requests) reached")
        if r.status_code == 426:
            return _ev(src, "error", "free key works only on localhost")
        r.raise_for_status()
        articles = r.json().get("articles", [])
    except Exception as exc:
        return _ev(src, "error", f"Could not reach NewsAPI ({exc})")
    note = " (free plan only shows articles older than 24 hours)"
    if not articles:
        return _ev(src, "no_data", "no matching news articles found" + note)

    # Relevance filter - same idea as check_fact_check(): a returned article must
    # actually share the claim's own keywords, not just satisfy the broad "AND" query
    # (NewsAPI's matching is loose, e.g. across title/description separately).
    claim_words = set(words)
    relevant = []
    for a in articles:
        art_text = f"{a.get('title', '')} {a.get('description', '')}"
        art_words = set(keywords(art_text, 20))
        overlap = len(claim_words & art_words) / len(claim_words) if claim_words else 0
        if overlap >= 0.4:
            relevant.append(a)
    if not relevant:
        return _ev(src, "no_data", "found articles, but none matched closely enough to be relevant" + note)

    outlets = sorted({(a.get("source") or {}).get("name", "?") for a in relevant})
    first = relevant[0]
    detail = f"{len(relevant)} matching article(s) from {', '.join(outlets[:3])}, e.g. '{first.get('title', '')}'"
    if len(outlets) >= 2:
        return _ev(src, "supports", detail, first.get("url", ""))
    return _ev(src, "partial", detail + " - only one outlet, not enough", first.get("url", ""))


# ---------------------------------------------------------------- full pipeline
def analyze(text, keys=None):
    keys = load_keys() if keys is None else keys
    kind = check_input_type(text)
    if kind == "personal":
        return {"kind": kind, "event": None, "evidence": [], "ml": None, "verdict": make_verdict(kind, [])}

    event = detect_event(text)
    evidence = []
    if event in ("earthquake", "rain"):
        place, problem = find_place(text)
        name = "USGS Earthquake API" if event == "earthquake" else "Open-Meteo Weather API"
        if place is None:
            evidence.append(_ev(name, "no_data", problem))
        elif event == "earthquake":
            evidence.append(check_earthquake(*place))
        else:
            evidence.append(check_rain(*place, closure=bool(CLOSURE.search(text))))
    evidence.append(check_fact_check(text, keys.get("GOOGLE_FACTCHECK_KEY")))
    evidence.append(check_news(text, keys.get("NEWSAPI_KEY")))

    try:
        ml = ml_style_score(text)
    except Exception:
        ml = None
    return {"kind": kind, "event": event, "evidence": evidence, "ml": ml, "verdict": make_verdict(kind, evidence, ml)}