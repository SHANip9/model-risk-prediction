from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.staticfiles import StaticFiles
from api.routes import plans, risk

app = FastAPI(
    title="OmniSight AI - Gig Worker Risk API",
    version="1.0.0",
    description="Real-time risk heatmap for gig workers in Mumbai zones (weather/traffic/time-aware).",
)

app.include_router(plans.router, prefix="/plans", tags=["plans"])
app.include_router(risk.router, prefix="/risk", tags=["risk"])

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
