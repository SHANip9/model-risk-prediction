"""
OmniSight AI — Live Zone Risk Scoring
======================================
Blends two sources:
  1. XGBoost baseline (55%) — from zone_risk_scores_latest.csv, produced
     by the weekly ML pipeline (src.risk_model.pipeline.weekly_job)
  2. Live weather (45%)     — WeatherAPI, same WEATHER_API_KEY as /route-risk

Cache: 15-minute in-process cache so 5 WeatherAPI calls are not made
       on every frontend poll.

File location: omnisight-backend/zone_risk.py
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent
_SCORES_CSV  = (
    _BACKEND_DIR
    / "model-risk-prediction"
    / "data"
    / "processed"
    / "zone_risk_scores_latest.csv"
)

ZONES: list[dict[str, Any]] = [
    {"id": "zone_1", "display_name": "Dharavi",      "city": "Mumbai", "lat": 19.0422, "lon": 72.8553},
    {"id": "zone_2", "display_name": "Kurla West",   "city": "Mumbai", "lat": 19.0728, "lon": 72.8826},
    {"id": "zone_3", "display_name": "Andheri East", "city": "Mumbai", "lat": 19.1136, "lon": 72.8697},
    {"id": "zone_4", "display_name": "Bandra Kurla", "city": "Mumbai", "lat": 19.0596, "lon": 72.8656},
    {"id": "zone_5", "display_name": "Thane West",   "city": "Thane",  "lat": 19.1852, "lon": 72.9710},
]

_GEO_FALLBACK: dict[str, float] = {
    "zone_1": 62.0,
    "zone_2": 55.0,
    "zone_3": 38.0,
    "zone_4": 58.0,
    "zone_5": 34.0,
}

_CACHE_TTL = 15 * 60
_cache: dict[str, Any] = {"data": None, "ts": 0.0}


def _load_baseline_scores() -> dict[str, float]:
    if not _SCORES_CSV.exists():
        logger.warning("zone_risk_scores_latest.csv not found — using geo fallback. Run weekly pipeline to fix.")
        return _GEO_FALLBACK.copy()
    try:
        import pandas as pd
        df = pd.read_csv(_SCORES_CSV, parse_dates=["week_start"])
        df = df.sort_values("week_start").groupby("zone_id").last().reset_index()
        result = {str(row["zone_id"]): float(row["risk_score_0_100"]) for _, row in df.iterrows()}
        logger.info("Loaded XGBoost baselines for %d zones", len(result))
        return result
    except Exception as exc:
        logger.error("Failed to read scores CSV: %s — using geo fallback", exc)
        return _GEO_FALLBACK.copy()


def _fetch_weather(lat: float, lon: float) -> dict | None:
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return None
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}&aqi=no"
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.error("WeatherAPI failed (%.4f, %.4f): %s", lat, lon, exc)
        return None


def _parse_weather(raw: dict) -> dict[str, Any]:
    c = raw.get("current", {})
    return {
        "precip_mm": float(c.get("precip_mm", 0.0)),
        "wind_kph":  float(c.get("wind_kph",  0.0)),
        "vis_km":    float(c.get("vis_km",    10.0)),
        "humidity":  float(c.get("humidity",  50.0)),
        "cloud":     float(c.get("cloud",      0.0)),
        "temp_c":    float(c.get("temp_c",    28.0)),
        "condition": c.get("condition", {}).get("text", "Clear"),
    }


_FALLBACK_WEATHER: dict[str, Any] = {
    "precip_mm": 0.0, "wind_kph": 0.0, "vis_km": 10.0,
    "humidity": 55.0, "cloud": 10.0, "temp_c": 30.0, "condition": "Clear",
}


def _weather_score(w: dict[str, Any]) -> float:
    rain  = float(w["precip_mm"])
    wind  = float(w["wind_kph"])
    vis   = float(w["vis_km"])
    cloud = float(w["cloud"])
    cond  = str(w["condition"]).lower()

    rain_s  = min(rain / 50.0, 1.0)
    wind_s  = min(wind / 80.0, 1.0)
    vis_s   = max(0.0, 1.0 - vis / 10.0)
    cloud_s = cloud / 100.0

    if any(k in cond for k in ["heavy rain", "torrential", "flood"]):
        cond_boost = 0.40
    elif any(k in cond for k in ["moderate rain", "rain", "shower", "drizzle"]):
        cond_boost = 0.22
    elif any(k in cond for k in ["thunder", "storm", "blizzard", "cyclone"]):
        cond_boost = 0.45
    elif any(k in cond for k in ["fog", "mist", "haze", "smoke"]):
        cond_boost = 0.12
    else:
        cond_boost = 0.0

    raw = rain_s * 0.38 + wind_s * 0.25 + vis_s * 0.18 + cloud_s * 0.04 + cond_boost * 0.15
    return float(np.clip(raw * 100.0, 0.0, 100.0))


def _risk_bin(s: float) -> str:
    return "high" if s >= 66 else ("medium" if s >= 33 else "low")

def _risk_label(s: float) -> str:
    return "DANGER" if s >= 75 else ("CAUTION" if s >= 50 else "SAFE")

def _risk_color(s: float) -> str:
    return "#dc2626" if s >= 75 else ("#f59e0b" if s >= 50 else "#16a34a")


def get_live_zone_scores() -> list[dict[str, Any]]:
    """
    Returns live blended scores for all 5 zones, cached 15 min.
    Blend: XGBoost baseline 55% + live WeatherAPI 45%
    """
    now = time.time()
    if _cache["data"] and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["data"]  # type: ignore[return-value]

    baselines  = _load_baseline_scores()
    scored_at  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results: list[dict[str, Any]] = []

    for zone in ZONES:
        raw = _fetch_weather(zone["lat"], zone["lon"])
        wf  = _parse_weather(raw) if raw else _FALLBACK_WEATHER.copy()
        src = "live" if raw else "fallback"

        baseline = baselines.get(zone["id"], _GEO_FALLBACK[zone["id"]])
        w_score  = _weather_score(wf)
        final    = round(float(np.clip(baseline * 0.55 + w_score * 0.45, 0.0, 100.0)), 2)

        results.append({
            "zone_id":        zone["id"],
            "display_name":   zone["display_name"],
            "city":           zone["city"],
            "center":         [zone["lat"], zone["lon"]],
            "risk_score":     final,
            "risk_bin":       _risk_bin(final),
            "risk_label":     _risk_label(final),
            "color":          _risk_color(final),
            "condition":      str(wf["condition"]),
            "precip_mm":      float(wf["precip_mm"]),
            "wind_kph":       float(wf["wind_kph"]),
            "vis_km":         float(wf["vis_km"]),
            "baseline_score": round(baseline, 2),
            "weather_score":  round(w_score, 2),
            "scored_at":      scored_at,
            "source":         src,
        })

    _cache["data"] = results
    _cache["ts"]   = time.time()
    logger.info("Zone scores refreshed — %d zones scored", len(results))
    return results


def get_heatmap_payload() -> dict[str, Any]:
    """Slim payload for the heatmap iframe and frontend map components."""
    zones = get_live_zone_scores()
    return {
        "center":              [19.0760, 72.8777],
        "radius_km":           25,
        "next_refresh_ms":     int((_cache["ts"] + _CACHE_TTL) * 1000),
        "update_interval_min": 15,
        "zones": [
            {
                "id":           z["zone_id"],
                "display_name": z["display_name"],
                "city":         z["city"],
                "center":       z["center"],
                "risk_score":   z["risk_score"],
                "risk_bin":     z["risk_bin"],
                "risk_label":   z["risk_label"],
                "color":        z["color"],
                "condition":    z["condition"],
                "precip_mm":    z["precip_mm"],
                "wind_kph":     z["wind_kph"],
            }
            for z in zones
        ],
    }
