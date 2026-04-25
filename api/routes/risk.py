from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.common.config import DATA_DIR

router = APIRouter()

RISK_SCORES_PATH = DATA_DIR / "processed" / "zone_risk_scores_latest.csv"
HEATMAP_PATH = DATA_DIR / "processed" / "zone_risk_heatmap_latest.csv"
ZONES_PATH = DATA_DIR / "zones.json"


def _load_scores() -> pd.DataFrame:
    if not RISK_SCORES_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Risk scores file not found. Run weekly scoring first "
                "(src.risk_model.inference.score_weekly)."
            ),
        )

    frame = pd.read_csv(RISK_SCORES_PATH)
    frame["week_start"] = pd.to_datetime(frame["week_start"], utc=False)
    return frame


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, pd.Timestamp):
            normalized[key] = value.isoformat()
        else:
            normalized[key] = value
    return normalized


def _load_zones() -> dict[str, Any]:
    if not ZONES_PATH.exists():
        raise HTTPException(status_code=404, detail="Zone config not found.")

    with ZONES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_heatmap() -> pd.DataFrame:
    if not HEATMAP_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Heatmap file not found. Run heatmap generation first.",
        )

    return pd.read_csv(HEATMAP_PATH)


def _get_risk_rank(score: float) -> str:
    if score >= 66:
        return "high"
    if score >= 33:
        return "medium"
    return "low"


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@router.get("/latest")
def latest_scores(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
    frame = _load_scores().sort_values("week_start", ascending=False).head(limit)
    records = frame.to_dict(orient="records")
    return [_normalize_record(record) for record in records]


@router.get("/heatmap")
def get_heatmap():
    zones = _load_zones()['zones']
    heatmap = _load_heatmap()
    result = []
    for z in zones:
        matches = heatmap[heatmap['zone_id'] == z['id']]
        if matches.empty:
            continue
        zone_row = matches.iloc[0]
        score = zone_row['risk_score_0_100']
        rank = _get_risk_rank(score)
        color = z['color_' + rank]
        result.append({
            'id': z['id'],
            'display_name': z['display_name'],
            'city': z.get('city'),
            'center': z['center'],
            'risk_score': float(score),
            'risk_rank': rank,
            'color': color
        })
    return {'zones': result}

@router.get("/risk/{lat}/{lon}")
def get_risk_at_location(lat: float, lon: float):
    zones = _load_zones()['zones']
    heatmap = _load_heatmap()
    min_dist = float('inf')
    nearest_zone = None
    for z in zones:
        dist = _haversine(lat, lon, z['center'][0], z['center'][1])
        if dist < min_dist:
            min_dist = dist
            nearest_zone = z
    if nearest_zone:
        matches = heatmap[heatmap['zone_id'] == nearest_zone['id']]
        if matches.empty:
            raise HTTPException(status_code=404, detail="No score found for nearest zone")
        zone_row = matches.iloc[0]
        score = zone_row['risk_score_0_100']
        rank = _get_risk_rank(score)
        return {
            'nearest_zone': nearest_zone['id'],
            'display_name': nearest_zone['display_name'],
            'distance_km': min_dist,
            'risk_score': float(score),
            'risk_rank': rank
        }
    raise HTTPException(status_code=404, detail="No zone nearby")

@router.get("/zones/{zone_id}")
def latest_zone_score(zone_id: str) -> dict[str, Any]:
    frame = _load_scores()
    matches = frame[frame["zone_id"].astype(str).str.lower() == zone_id.lower()]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Zone not found: {zone_id}")

    latest_row = matches.sort_values("week_start", ascending=False).iloc[0]
    return _normalize_record({k: latest_row[k] for k in matches.columns})
