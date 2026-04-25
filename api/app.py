from __future__ import annotations

from pathlib import Path
import sys

import logging

from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.staticfiles import StaticFiles
from api.routes import plans, risk

app = FastAPI(
    title="OmniSight AI - Gig Worker Risk API",
    version="1.0.0",
    description="Real-time risk heatmap for gig workers across 20 Pan-India zones.",
)

app.include_router(plans.router, prefix="/plans", tags=["plans"])
app.include_router(risk.router, prefix="/risk", tags=["risk"])

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

from zone_risk import get_live_zone_scores, get_heatmap_payload

logger = logging.getLogger(__name__)

@app.get("/zones/risk/live")
def live_zone_risk():
    try:
        zones = get_live_zone_scores()
        return {
            "zones":               zones,
            "zone_count":          len(zones),
            "update_interval_min": 10,
            "scored_at":           zones[0]["scored_at"] if zones else None,
        }
    except Exception as exc:
        logger.error("live_zone_risk failed: %s", exc)
        raise HTTPException(status_code=500, detail="Zone risk scoring failed")

@app.get("/zones/risk/heatmap")
def live_heatmap_data():
    try:
        return get_heatmap_payload()
    except Exception as exc:
        logger.error("live_heatmap_data failed: %s", exc)
        raise HTTPException(status_code=500, detail="Heatmap data generation failed")
