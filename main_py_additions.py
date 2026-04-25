# =============================================================================
#  ADD THIS IMPORT at the top of main.py, alongside the existing imports

# =============================================================================
#
#     from zone_risk import get_live_zone_scores, get_heatmap_payload
#
# =============================================================================
#  PASTE THESE ENDPOINTS at the bottom of main.py
#  (after the existing /route-risk endpoint)
# =============================================================================


# ---------------------------------------------------------------------------
# ZONE RISK — LIVE SCORING  (every 10 min, cached in-process)
# ---------------------------------------------------------------------------

@app.get("/zones/risk/live")
def live_zone_risk():
    """
    Returns live 0-100 risk scores for all 20 Pan-India zones
    (Mumbai, Delhi, Kolkata, Chennai — 5 zones each).

    Scores are computed from:
      - XGBoost baseline trained on historical zone data (55%)
      - Live weather from WeatherAPI (same WEATHER_API_KEY as /route-risk) (45%)

    Results are cached for 10 minutes in-process.
    Call this endpoint every 10 min from the frontend to keep the heatmap fresh.

    Response shape:
    {
      "zones": [...],
      "zone_count": 20,
      "update_interval_min": 10,
      "scored_at": "2026-04-13T..."
    }
    """
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
    """
    Slim heatmap payload consumed by the self-refreshing heatmap.html
    and the AdminDashboard / ClientDashboard map components.

    Response shape:
    {
      "center": [22.5, 80.0],       // India center
      "radius_km": 1500,            // Pan-India coverage
      "next_refresh_ms": <epoch ms when cache expires>,
      "update_interval_min": 10,
      "zones": [
        {
          "id", "display_name", "city", "center",
          "risk_score", "risk_bin", "risk_label",
          "color", "condition", "precip_mm", "wind_kph"
        }, ...  // 20 zones
      ]
    }
    """
    try:
        return get_heatmap_payload()
    except Exception as exc:
        logger.error("live_heatmap_data failed: %s", exc)
        raise HTTPException(status_code=500, detail="Heatmap data generation failed")
