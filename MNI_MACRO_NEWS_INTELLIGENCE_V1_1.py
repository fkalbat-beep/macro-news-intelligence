#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNI — Macro News Intelligence Agent v1.1
Personal-use MVP for Faisal's E/B/C/D crypto strategy family.

Purpose
-------
1) Receive structured macro/news events.
2) Measure economic surprise vs forecast/previous.
3) Convert the event into a crypto macro-impact score.
4) Observe BTC/ETH market reaction after the event.
5) Publish one normalized MACRO REGIME payload for E/B/C/D.

Important
---------
- This module is decision-support only.
- It does NOT place trades.
- It does NOT modify E/B/C/D.
- Strategies can read /regime or /strategy-context when integration is enabled.

Run locally / Railway:
    pip install fastapi uvicorn requests feedparser
    python -u MNI_MACRO_NEWS_INTELLIGENCE_V1.py

Environment variables (optional):
    PORT=8080
    MNI_POLL_SECONDS=60
    MNI_RSS_URLS=https://example.com/feed1.xml,https://example.com/feed2.xml
    MNI_STATE_PATH=/data/mni_state.json
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import hashlib
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import feedparser
except Exception:
    feedparser = None

APP_VERSION = "MNI-MACRO-NEWS-INTELLIGENCE-V1.1"
PORT = int(os.getenv("PORT", "8080"))
POLL_SECONDS = max(30, int(os.getenv("MNI_POLL_SECONDS", "60")))
STATE_PATH = Path(os.getenv("MNI_STATE_PATH", "/data/mni_state.json"))
RSS_URLS = [x.strip() for x in os.getenv("MNI_RSS_URLS", "").split(",") if x.strip()]

BINANCE_BASE = "https://api.binance.com"
TRACK_SYMBOLS = ("BTCUSDT", "ETHUSDT")
MAX_EVENTS = 300
MAX_ARTICLES = 500

# ----------------------------- policy -----------------------------

EVENT_BASE_WEIGHT = {
    "FOMC_RATE": 100,
    "FOMC_STATEMENT": 95,
    "POWELL_SPEECH": 90,
    "CPI": 92,
    "CORE_CPI": 94,
    "PCE": 90,
    "CORE_PCE": 92,
    "NFP": 88,
    "UNEMPLOYMENT": 82,
    "PPI": 78,
    "GDP": 72,
    "RETAIL_SALES": 70,
    "ISM": 68,
    "JOBLESS_CLAIMS": 62,
    "OIL_SHOCK": 85,
    "GEOPOLITICAL": 88,
    "REGULATION": 80,
    "ETF": 82,
    "CRYPTO_SYSTEMIC": 95,
    "OTHER": 45,
}

# Positive macro bias => bullish for crypto. Negative => bearish.
# This is intentionally simple in V1. The reaction layer can override it.
EVENT_DIRECTION_RULE = {
    "CPI": -1,
    "CORE_CPI": -1,
    "PCE": -1,
    "CORE_PCE": -1,
    "PPI": -1,
    "FOMC_RATE": -1,
    "FOMC_STATEMENT": 0,
    "POWELL_SPEECH": 0,
    "NFP": -1,
    "UNEMPLOYMENT": +1,
    "JOBLESS_CLAIMS": +1,
    "GDP": -1,
    "RETAIL_SALES": -1,
    "ISM": -1,
    "OIL_SHOCK": -1,
    "GEOPOLITICAL": -1,
    "REGULATION": 0,
    "ETF": 0,
    "CRYPTO_SYSTEMIC": -1,
    "OTHER": 0,
}

STRATEGY_POLICY = {
    "E": {
        "red": "ALLOW_EXCEPTIONAL_ONLY",
        "yellow": "ALLOW_SELECTIVE",
        "green": "NORMAL",
        "size_mult_red": 0.50,
        "size_mult_yellow": 0.80,
        "size_mult_green": 1.00,
    },
    "B": {
        "red": "TIGHTEN_CONFIRMATION",
        "yellow": "NORMAL_SELECTIVE",
        "green": "NORMAL",
        "size_mult_red": 0.60,
        "size_mult_yellow": 0.90,
        "size_mult_green": 1.00,
    },
    "C": {
        "red": "WAIT_FOR_1H_ABSORPTION",
        "yellow": "ALLOW_HIGH_QUALITY",
        "green": "NORMAL",
        "size_mult_red": 0.50,
        "size_mult_yellow": 0.85,
        "size_mult_green": 1.00,
    },
    "D": {
        "red": "BLOCK_NEW_LONG_UNTIL_ABSORBED",
        "yellow": "ALLOW_EXCEPTIONAL_ONLY",
        "green": "NORMAL",
        "size_mult_red": 0.00,
        "size_mult_yellow": 0.70,
        "size_mult_green": 1.00,
    },
}

NEGATIVE_WORDS = {
    "hotter", "inflation", "hawkish", "rate hike", "war", "attack", "sanction",
    "liquidation", "hack", "exploit", "ban", "lawsuit", "default", "recession",
    "tariff", "higher yields", "oil surge", "crisis", "selloff",
}
POSITIVE_WORDS = {
    "cooler", "dovish", "rate cut", "approval", "approved", "stimulus",
    "ceasefire", "inflows", "surplus liquidity", "easing", "soft landing",
    "lower yields", "disinflation", "etf approval",
}

# ----------------------------- models -----------------------------

class MacroEventIn(BaseModel):
    headline: str
    event_type: str = "OTHER"
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[str] = None
    notes: Optional[str] = None

class NewsItemIn(BaseModel):
    headline: str
    summary: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[str] = None
    assets: List[str] = Field(default_factory=list)
    event_type: Optional[str] = None

class ReactionPoint(BaseModel):
    symbol: str
    horizon_min: int
    return_pct: float

class MNIState:
    def __init__(self) -> None:
        self.events: deque = deque(maxlen=MAX_EVENTS)
        self.articles: deque = deque(maxlen=MAX_ARTICLES)
        self.last_regime: Dict[str, Any] = self._empty_regime()
        self.last_prices: Dict[str, float] = {}
        self.updated_at = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _empty_regime() -> Dict[str, Any]:
        return {
            "engine": APP_VERSION,
            "macro_regime": "YELLOW",
            "impact_score": 0,
            "direction": "NEUTRAL",
            "confidence": 0,
            "dominant_event": None,
            "reaction": {},
            "reason": "No high-impact event evaluated yet.",
            "valid_until": None,
        }

STATE = MNIState()
app = FastAPI(title="MNI Macro News Intelligence", version="1.1")

# ----------------------------- utilities -----------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def now_iso() -> str:
    return now_utc().isoformat()

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

def normalize_event_type(event_type: Optional[str], text: str = "") -> str:
    e = (event_type or "").upper().strip().replace(" ", "_")
    if e in EVENT_BASE_WEIGHT:
        return e
    t = text.lower()
    rules = [
        ("core cpi", "CORE_CPI"),
        ("cpi", "CPI"),
        ("core pce", "CORE_PCE"),
        ("pce", "PCE"),
        ("producer price", "PPI"),
        ("ppi", "PPI"),
        ("nonfarm payroll", "NFP"),
        ("payroll", "NFP"),
        ("unemployment", "UNEMPLOYMENT"),
        ("jobless claims", "JOBLESS_CLAIMS"),
        ("fomc", "FOMC_STATEMENT"),
        ("powell", "POWELL_SPEECH"),
        ("federal reserve", "FOMC_STATEMENT"),
        ("rate decision", "FOMC_RATE"),
        ("interest rate", "FOMC_RATE"),
        ("gdp", "GDP"),
        ("retail sales", "RETAIL_SALES"),
        ("ism", "ISM"),
        ("oil", "OIL_SHOCK"),
        ("war", "GEOPOLITICAL"),
        ("attack", "GEOPOLITICAL"),
        ("sanction", "GEOPOLITICAL"),
        ("etf", "ETF"),
        ("hack", "CRYPTO_SYSTEMIC"),
        ("exploit", "CRYPTO_SYSTEMIC"),
        ("liquidation", "CRYPTO_SYSTEMIC"),
        ("regulation", "REGULATION"),
        ("sec ", "REGULATION"),
    ]
    for needle, kind in rules:
        if needle in t:
            return kind
    return "OTHER"

def surprise(actual: Optional[float], forecast: Optional[float], previous: Optional[float]) -> Tuple[float, str]:
    """
    Returns normalized surprise in approximately [-3,+3].
    Prefer forecast; fallback to previous.
    """
    a, f, p = safe_float(actual), safe_float(forecast), safe_float(previous)
    baseline = f if f is not None else p
    if a is None or baseline is None:
        return 0.0, "NO_NUMERIC_SURPRISE"
    scale = max(abs(baseline), 0.1)
    raw = (a - baseline) / scale
    return clamp(raw * 5.0, -3.0, 3.0), ("VS_FORECAST" if f is not None else "VS_PREVIOUS")

def lexical_sentiment(text: str) -> Tuple[float, List[str]]:
    t = text.lower()
    pos = [w for w in POSITIVE_WORDS if w in t]
    neg = [w for w in NEGATIVE_WORDS if w in t]
    score = clamp((len(pos) - len(neg)) / 4.0, -1.0, 1.0)
    tags = [f"+{x}" for x in sorted(pos)] + [f"-{x}" for x in sorted(neg)]
    return score, tags

def macro_direction(event_type: str, surprise_score: float, text_sentiment: float) -> float:
    """
    +1 bullish crypto, -1 bearish crypto.
    Numeric macro surprise is interpreted by event type.
    Text sentiment is blended for qualitative events.
    """
    base_rule = EVENT_DIRECTION_RULE.get(event_type, 0)
    if base_rule != 0 and abs(surprise_score) > 0.01:
        directional = base_rule * surprise_score
        return clamp(directional / 1.5, -1.0, 1.0)
    if event_type in {"FOMC_STATEMENT", "POWELL_SPEECH", "REGULATION", "ETF", "OTHER"}:
        return clamp(text_sentiment, -1.0, 1.0)
    return clamp((0.7 * text_sentiment), -1.0, 1.0)

def event_impact_score(event_type: str, surprise_score: float, direction_strength: float) -> int:
    base = EVENT_BASE_WEIGHT.get(event_type, EVENT_BASE_WEIGHT["OTHER"])
    surprise_boost = min(20.0, abs(surprise_score) * 7.0)
    directional_boost = abs(direction_strength) * 8.0
    return int(round(clamp(base * 0.78 + surprise_boost + directional_boost, 0, 100)))

def duration_minutes(event_type: str, impact: int) -> int:
    if event_type in {"FOMC_RATE", "FOMC_STATEMENT", "POWELL_SPEECH", "CPI", "CORE_CPI", "PCE", "CORE_PCE"}:
        return 240 if impact >= 80 else 120
    if event_type in {"NFP", "GDP", "OIL_SHOCK", "GEOPOLITICAL", "CRYPTO_SYSTEMIC"}:
        return 240 if impact >= 75 else 90
    return 60 if impact >= 60 else 30

def market_price(symbol: str) -> float:
    r = requests.get(f"{BINANCE_BASE}/api/v3/ticker/price", params={"symbol": symbol}, timeout=8)
    r.raise_for_status()
    return float(r.json()["price"])

def klines(symbol: str, interval: str, limit: int = 120) -> List[List[Any]]:
    r = requests.get(
        f"{BINANCE_BASE}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

def return_over_minutes(symbol: str, minutes: int) -> Optional[float]:
    interval = "1m" if minutes <= 60 else "5m"
    needed = min(500, minutes + 2) if interval == "1m" else min(500, math.ceil(minutes / 5) + 2)
    try:
        rows = klines(symbol, interval, needed)
        if len(rows) < 2:
            return None
        open_px = float(rows[0][1])
        close_px = float(rows[-1][4])
        if open_px <= 0:
            return None
        return (close_px / open_px - 1.0) * 100.0
    except Exception:
        return None

def current_cross_market_reaction() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for sym in TRACK_SYMBOLS:
        result[sym] = {
            "5m": return_over_minutes(sym, 5),
            "15m": return_over_minutes(sym, 15),
            "60m": return_over_minutes(sym, 60),
        }
    return result

def reaction_bias(reaction: Dict[str, Any]) -> Tuple[float, float]:
    """
    Returns (bias [-1,+1], confidence [0,1]).
    BTC receives greater weight than ETH.
    """
    values = []
    weights = []
    mapping = [
        ("BTCUSDT", "5m", 0.15),
        ("BTCUSDT", "15m", 0.30),
        ("BTCUSDT", "60m", 0.25),
        ("ETHUSDT", "15m", 0.15),
        ("ETHUSDT", "60m", 0.15),
    ]
    for sym, h, w in mapping:
        v = reaction.get(sym, {}).get(h)
        if v is None:
            continue
        # 1.5% move ~= full directional reaction for V1 purposes.
        values.append(clamp(v / 1.5, -1.0, 1.0) * w)
        weights.append(w)
    if not weights:
        return 0.0, 0.0
    total_w = sum(weights)
    bias = sum(values) / total_w
    confidence = clamp(total_w, 0.0, 1.0)
    return clamp(bias, -1.0, 1.0), confidence

def classify_regime(combined_bias: float, impact: int) -> str:
    if impact >= 70 and combined_bias <= -0.25:
        return "RED"
    if impact >= 75 and combined_bias >= 0.25:
        return "GREEN"
    if combined_bias <= -0.60:
        return "RED"
    if combined_bias >= 0.60:
        return "GREEN"
    return "YELLOW"

def direction_name(v: float) -> str:
    if v >= 0.20:
        return "BULLISH"
    if v <= -0.20:
        return "BEARISH"
    return "MIXED_NEUTRAL"

def event_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:14]

def save_state() -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "events": list(STATE.events),
            "articles": list(STATE.articles),
            "last_regime": STATE.last_regime,
            "updated_at": now_iso(),
        }
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception as exc:
        print(f"[MNI STATE WARN] {exc}", flush=True)

def load_state() -> None:
    try:
        if not STATE_PATH.exists():
            return
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        for x in payload.get("events", [])[-MAX_EVENTS:]:
            STATE.events.append(x)
        for x in payload.get("articles", [])[-MAX_ARTICLES:]:
            STATE.articles.append(x)
        if isinstance(payload.get("last_regime"), dict):
            STATE.last_regime = payload["last_regime"]
        print(f"[MNI STATE] loaded events={len(STATE.events)} articles={len(STATE.articles)}", flush=True)
    except Exception as exc:
        print(f"[MNI STATE WARN] load failed: {exc}", flush=True)

# ----------------------------- analysis -----------------------------

def analyze_event(evt: MacroEventIn, reaction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    text = " ".join(x for x in [evt.headline, evt.notes or ""] if x)
    etype = normalize_event_type(evt.event_type, text)
    s, basis = surprise(evt.actual, evt.forecast, evt.previous)
    text_sent, tags = lexical_sentiment(text)
    prior_bias = macro_direction(etype, s, text_sent)
    impact = event_impact_score(etype, s, prior_bias)

    reaction = reaction or current_cross_market_reaction()
    rx_bias, rx_conf = reaction_bias(reaction)

    # Market reaction overrides headline/theory when sufficiently observed.
    if rx_conf >= 0.50:
        combined = clamp(prior_bias * 0.40 + rx_bias * 0.60, -1.0, 1.0)
    else:
        combined = clamp(prior_bias * 0.75 + rx_bias * 0.25, -1.0, 1.0)

    regime = classify_regime(combined, impact)
    duration = duration_minutes(etype, impact)
    valid_until_epoch = time.time() + duration * 60
    confidence = int(round(clamp(
        45 + abs(s) * 10 + abs(combined) * 25 + rx_conf * 20,
        0, 100
    )))

    payload = evt.model_dump()
    rec = {
        "id": event_id(payload),
        "engine": APP_VERSION,
        "event_type": etype,
        "headline": evt.headline,
        "actual": evt.actual,
        "forecast": evt.forecast,
        "previous": evt.previous,
        "surprise_score": round(s, 4),
        "surprise_basis": basis,
        "text_sentiment": round(text_sent, 4),
        "text_tags": tags,
        "theoretical_crypto_bias": round(prior_bias, 4),
        "reaction_bias": round(rx_bias, 4),
        "combined_bias": round(combined, 4),
        "direction": direction_name(combined),
        "impact_score": impact,
        "confidence": confidence,
        "macro_regime": regime,
        "expected_duration_min": duration,
        "reaction": reaction,
        "source": evt.source,
        "source_url": evt.source_url,
        "published_at": evt.published_at or now_iso(),
        "evaluated_at": now_iso(),
        "valid_until_epoch": valid_until_epoch,
        "valid_until": datetime.fromtimestamp(valid_until_epoch, tz=timezone.utc).isoformat(),
    }
    return rec

def rebuild_regime_from_events() -> Dict[str, Any]:
    now_ts = time.time()
    active = [
        e for e in STATE.events
        if float(e.get("valid_until_epoch", 0) or 0) >= now_ts
    ]
    if not active:
        STATE.last_regime = MNIState._empty_regime()
        return STATE.last_regime

    active.sort(key=lambda e: (e.get("impact_score", 0), e.get("confidence", 0)), reverse=True)
    dominant = active[0]

    weighted_num = 0.0
    weighted_den = 0.0
    for e in active[:10]:
        w = max(1.0, float(e.get("impact_score", 0))) * max(0.25, float(e.get("confidence", 0)) / 100.0)
        weighted_num += float(e.get("combined_bias", 0.0)) * w
        weighted_den += w
    portfolio_bias = weighted_num / weighted_den if weighted_den else 0.0
    impact = max(int(e.get("impact_score", 0)) for e in active)
    regime = classify_regime(portfolio_bias, impact)

    STATE.last_regime = {
        "engine": APP_VERSION,
        "macro_regime": regime,
        "impact_score": impact,
        "direction": direction_name(portfolio_bias),
        "confidence": int(round(sum(float(e.get("confidence", 0)) for e in active[:5]) / min(5, len(active)))),
        "dominant_event": {
            "id": dominant.get("id"),
            "event_type": dominant.get("event_type"),
            "headline": dominant.get("headline"),
            "impact_score": dominant.get("impact_score"),
            "direction": dominant.get("direction"),
        },
        "reaction": dominant.get("reaction", {}),
        "reason": f"{len(active)} active macro/news event(s); market reaction is blended with theoretical impact.",
        "valid_until": max((e.get("valid_until") for e in active if e.get("valid_until")), default=None),
        "updated_at": now_iso(),
    }
    return STATE.last_regime

def strategy_context() -> Dict[str, Any]:
    regime = rebuild_regime_from_events()
    r = regime.get("macro_regime", "YELLOW").lower()
    result = {}
    for strat, pol in STRATEGY_POLICY.items():
        result[strat] = {
            "action": pol.get(r, "NORMAL"),
            "size_multiplier": pol.get(f"size_mult_{r}", 1.0),
            "macro_regime": regime.get("macro_regime"),
            "impact_score": regime.get("impact_score"),
            "direction": regime.get("direction"),
            "confidence": regime.get("confidence"),
            "dominant_event": regime.get("dominant_event"),
        }
    return {
        "engine": APP_VERSION,
        "generated_at": now_iso(),
        "regime": regime,
        "strategies": result,
    }

# ----------------------------- RSS ingestion -----------------------------

def news_from_rss() -> int:
    if not RSS_URLS or feedparser is None:
        return 0
    existing = {a.get("id") for a in STATE.articles}
    added = 0
    for url in RSS_URLS:
        try:
            feed = feedparser.parse(url)
            for ent in feed.entries[:50]:
                headline = str(ent.get("title", "")).strip()
                if not headline:
                    continue
                summary = re.sub(r"<[^>]+>", " ", str(ent.get("summary", "")))
                source_url = str(ent.get("link", "")).strip() or None
                published = str(ent.get("published", "")).strip() or None
                raw = {
                    "headline": headline,
                    "summary": summary[:1200],
                    "source_url": source_url,
                    "published_at": published,
                }
                rid = event_id(raw)
                if rid in existing:
                    continue
                etype = normalize_event_type(None, f"{headline} {summary}")
                sent, tags = lexical_sentiment(f"{headline} {summary}")
                STATE.articles.append({
                    "id": rid,
                    "headline": headline,
                    "summary": summary[:1200],
                    "source_url": source_url,
                    "published_at": published,
                    "event_type": etype,
                    "lexical_sentiment": sent,
                    "tags": tags,
                    "ingested_at": now_iso(),
                })
                existing.add(rid)
                added += 1
        except Exception as exc:
            print(f"[MNI RSS WARN] {url}: {exc}", flush=True)
    if added:
        save_state()
    return added

# ----------------------------- API -----------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "engine": APP_VERSION,
        "status": "online",
        "purpose": "Macro/news intelligence and cross-market reaction layer for E/B/C/D.",
        "trade_execution": False,
        "events": len(STATE.events),
        "articles": len(STATE.articles),
        "macro_regime": rebuild_regime_from_events(),
    }

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "engine": APP_VERSION, "time": now_iso()}

@app.post("/event")
def post_event(evt: MacroEventIn) -> Dict[str, Any]:
    try:
        rec = analyze_event(evt)
        # de-duplicate by id
        STATE.events = deque([e for e in STATE.events if e.get("id") != rec["id"]], maxlen=MAX_EVENTS)
        STATE.events.append(rec)
        rebuild_regime_from_events()
        save_state()
        print(
            f"[MNI EVENT] {rec['event_type']} impact={rec['impact_score']} "
            f"direction={rec['direction']} regime={rec['macro_regime']} "
            f"confidence={rec['confidence']} headline={rec['headline'][:100]}",
            flush=True,
        )
        return {"event": rec, "regime": STATE.last_regime, "strategy_context": strategy_context()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/news")
def post_news(item: NewsItemIn) -> Dict[str, Any]:
    raw = item.model_dump()
    rid = event_id(raw)
    text = f"{item.headline} {item.summary or ''}"
    etype = normalize_event_type(item.event_type, text)
    sent, tags = lexical_sentiment(text)
    rec = {
        "id": rid,
        **raw,
        "event_type": etype,
        "lexical_sentiment": sent,
        "tags": tags,
        "ingested_at": now_iso(),
    }
    STATE.articles = deque([a for a in STATE.articles if a.get("id") != rid], maxlen=MAX_ARTICLES)
    STATE.articles.append(rec)
    save_state()
    return rec

@app.get("/regime")
def get_regime() -> Dict[str, Any]:
    return rebuild_regime_from_events()

@app.get("/strategy-context")
def get_strategy_context() -> Dict[str, Any]:
    return strategy_context()

@app.get("/events")
def get_events(limit: int = 50) -> List[Dict[str, Any]]:
    limit = max(1, min(200, limit))
    return list(STATE.events)[-limit:][::-1]

@app.get("/news")
def get_news(limit: int = 50) -> List[Dict[str, Any]]:
    limit = max(1, min(200, limit))
    return list(STATE.articles)[-limit:][::-1]

@app.post("/rss/poll")
def poll_rss() -> Dict[str, Any]:
    added = news_from_rss()
    return {"added": added, "rss_sources": len(RSS_URLS), "feedparser_available": feedparser is not None}

@app.get("/reaction")
def reaction() -> Dict[str, Any]:
    rx = current_cross_market_reaction()
    b, c = reaction_bias(rx)
    return {
        "time": now_iso(),
        "reaction": rx,
        "reaction_bias": b,
        "reaction_direction": direction_name(b),
        "reaction_confidence": c,
    }

# ----------------------------- background loop -----------------------------

async def background_loop() -> None:
    while True:
        try:
            if RSS_URLS:
                added = await asyncio.to_thread(news_from_rss)
                if added:
                    print(f"[MNI RSS] added={added}", flush=True)

            # Re-evaluate the latest active macro event with fresh BTC/ETH reaction.
            if STATE.events:
                latest = STATE.events[-1]
                if float(latest.get("valid_until_epoch", 0) or 0) >= time.time():
                    rx = await asyncio.to_thread(current_cross_market_reaction)
                    rx_bias, rx_conf = reaction_bias(rx)
                    theoretical = float(latest.get("theoretical_crypto_bias", 0.0))
                    combined = clamp(
                        theoretical * (0.40 if rx_conf >= 0.50 else 0.75)
                        + rx_bias * (0.60 if rx_conf >= 0.50 else 0.25),
                        -1.0, 1.0
                    )
                    latest["reaction"] = rx
                    latest["reaction_bias"] = round(rx_bias, 4)
                    latest["combined_bias"] = round(combined, 4)
                    latest["direction"] = direction_name(combined)
                    latest["macro_regime"] = classify_regime(combined, int(latest.get("impact_score", 0)))
                    latest["last_reaction_update"] = now_iso()
                    rebuild_regime_from_events()
                    save_state()

            print(
                f"[MNI CYCLE] regime={STATE.last_regime.get('macro_regime')} "
                f"impact={STATE.last_regime.get('impact_score')} "
                f"direction={STATE.last_regime.get('direction')} "
                f"events={len(STATE.events)} articles={len(STATE.articles)}",
                flush=True,
            )
        except Exception as exc:
            print(f"[MNI LOOP WARN] {type(exc).__name__}: {exc}", flush=True)

        await asyncio.sleep(POLL_SECONDS)

@app.on_event("startup")
async def startup() -> None:
    load_state()
    print(f"[MNI BUILD] {APP_VERSION}", flush=True)
    print("[MNI EXECUTION] DECISION-SUPPORT ONLY — NO TRADE EXECUTION", flush=True)
    print(f"[MNI POLICY] News -> Surprise -> Impact -> BTC/ETH Reaction -> Macro Regime -> E/B/C/D context", flush=True)
    asyncio.create_task(background_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
