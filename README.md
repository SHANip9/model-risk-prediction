<p align="center">
  <img src="docs/images/hero_banner.png" alt="OmniSight AI" width="100%"/>
</p>

<h1 align="center">OmniSight AI — Zone Risk Prediction</h1>

<p align="center">
  Real-time ML-powered risk scoring for gig worker safety across 20 Pan-India zones
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Live_Weather-0ea5e9?style=for-the-badge"/>
</p>

---

## What Is This?

A **0–100 risk score** is computed for each of the **20 monitored zones** across Mumbai, Delhi, Kolkata, and Chennai by blending two signals:

- **55 %** — XGBoost model trained weekly on historical zone data (floods, curfews, outages, traffic, geography)
- **45 %** — Real-time weather from WeatherAPI.com (rainfall, wind, visibility, storm conditions)

The blended score powers a live risk heatmap, insurance pricing tiers, and safety alerts for gig workers.

<p align="center">
  <img src="docs/images/dashboard_preview.png" alt="Dashboard Preview" width="88%"/>
</p>

---

## How It Works

<p align="center">
  <img src="docs/images/architecture_diagram.png" alt="Architecture" width="85%"/>
</p>

```mermaid
flowchart TD
    A["Raw Zone History CSV\n20 zones × 4 cities"] -->|ingest| B["Feature Engineering\n16 features + geo index"]
    B --> C["XGBoost Training\n500 trees · temporal split"]
    C --> D["LightGBM Cross-Check\nRMSE within 10%"]
    C --> E["Weekly Batch Scoring\nzone_risk_scores_latest.csv"]
    E -->|baseline 55%| F["Live Score Blender"]
    G["WeatherAPI.com\nper-zone lat/lon × 20"] -->|weather 45%| F
    F --> H["FastAPI"]
    H --> I["Heatmap · Dashboards · Pricing API"]

    style A fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style B fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
    style C fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style D fill:#1e293b,stroke:#16a34a,color:#e2e8f0
    style E fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style F fill:#1e293b,stroke:#ef4444,color:#e2e8f0
    style G fill:#1e293b,stroke:#0ea5e9,color:#e2e8f0
    style H fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
    style I fill:#1e293b,stroke:#16a34a,color:#e2e8f0
```

### Weekly Batch Pipeline

Runs every Sunday 23:00 IST. Orchestrated by `src/risk_model/pipeline/weekly_job.py`.

| Stage | What Happens | Output |
|:---|:---|:---|
| **Feature Engineering** | Ingests raw zone history for all 20 zones, adds temporal (month sin/cos) and geographic (river proximity, elevation, coastal exposure) features, derives a weighted risk target | `data/features/weekly_zone_features.csv` |
| **XGBoost Training** | Trains a 500-tree regressor on an 80/20 temporal split, evaluates RMSE / MAE / R² | `artifacts/models/risk_xgboost_bundle.joblib` |
| **LightGBM Validation** | Trains a second model on the same split; passes if RMSE is within 10% of XGBoost | `artifacts/reports/validation_lightgbm_vs_xgboost.json` |
| **Batch Scoring** | Runs the trained model over all 20 zones × weeks, clips to 0–100, assigns low/medium/high bins | `data/processed/zone_risk_scores_latest.csv` |

### Live Blending Layer

Runs inside the backend server (`zone_risk.py`), refreshes every 10 minutes.

1. Loads the latest XGBoost baseline scores from the batch CSV
2. Calls WeatherAPI.com for each zone's current conditions (rain, wind, visibility, storm keywords) — **20 API calls per refresh cycle**
3. Converts weather into a 0–100 weather risk score (rainfall 38%, wind 25%, visibility 18%, condition boost 15%, cloud 4%)
4. Blends: **`final = baseline × 0.55 + weather × 0.45`**
5. Caches the result in-process for 10 minutes

### Heatmap

`generate_zone_heatmap.py` produces a self-refreshing Folium/Leaflet HTML map with colour-coded risk circles, heat overlay, and nearest-zone links. Embedded as an iframe in the React dashboards — auto-polls the API every 10 min via injected JS.

---

## Monitored Zones (20 · Pan-India)

> [!NOTE]
> All geographic configurations and zone mappings are centrally managed in `data/zones.json` to ensure synchronization between the machine learning pipeline, the FastAPI backend, and the frontend dashboards.

### Mumbai (5 zones)

| Zone | Area | Coordinates |
|:---:|:---|:---|
| `zone_1` | Dharavi | 19.042°N, 72.855°E |
| `zone_2` | Kurla West | 19.073°N, 72.883°E |
| `zone_3` | Andheri East | 19.114°N, 72.870°E |
| `zone_4` | Bandra Kurla | 19.060°N, 72.866°E |
| `zone_5` | Thane West | 19.185°N, 72.971°E |

### Delhi (5 zones)

| Zone | Area | Coordinates |
|:---:|:---|:---|
| `zone_6` | Chandni Chowk | 28.651°N, 77.230°E |
| `zone_7` | Connaught Place | 28.632°N, 77.217°E |
| `zone_8` | Lajpat Nagar | 28.570°N, 77.240°E |
| `zone_9` | Dwarka | 28.592°N, 77.046°E |
| `zone_10` | Rohini | 28.750°N, 77.057°E |

### Kolkata / West Bengal (5 zones)

| Zone | Area | Coordinates |
|:---:|:---|:---|
| `zone_11` | Salt Lake | 22.580°N, 88.415°E |
| `zone_12` | Howrah | 22.596°N, 88.264°E |
| `zone_13` | Park Street | 22.551°N, 88.353°E |
| `zone_14` | Jadavpur | 22.499°N, 88.371°E |
| `zone_15` | Dum Dum | 22.635°N, 88.423°E |

### Chennai (5 zones)

| Zone | Area | Coordinates |
|:---:|:---|:---|
| `zone_16` | T. Nagar | 13.042°N, 80.234°E |
| `zone_17` | Adyar | 13.007°N, 80.257°E |
| `zone_18` | Anna Nagar | 13.085°N, 80.210°E |
| `zone_19` | Velachery | 12.982°N, 80.218°E |
| `zone_20` | Tambaram | 12.925°N, 80.100°E |

---

## Quick Start

```bash
pip install -r requirements.txt

# Run full pipeline (generates sample data for all 20 zones if missing)
python -m src.risk_model.pipeline.weekly_job --generate-sample-if-missing

# Start API server
export WEATHER_API_KEY="your_weatherapi_key"
uvicorn api.app:app --reload
```

API docs at `http://127.0.0.1:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/health` | Health check |
| `GET` | `/risk/latest` | Latest batch-scored risk rows |
| `GET` | `/risk/zones/{zone_id}` | Latest score for a specific zone |
| `GET` | `/risk/heatmap` | Heatmap payload (batch) |
| `GET` | `/zones/risk/live` | **Live** blended scores (ML + weather) — 20 zones |
| `GET` | `/zones/risk/heatmap` | **Live** heatmap payload — 20 zones |
| `GET` | `/plans` | Insurance pricing plans |
| `GET` | `/plans/{plan_id}` | Single plan detail |
| `GET` | `/plans/compare/table` | Plan comparison table |

---

## Fallback Strategy

Every layer has a fallback so the system never returns zero data:

| Layer | Primary | Fallback |
|:---|:---|:---|
| ML Training | XGBoost | sklearn HistGradientBoosting |
| Validation | LightGBM | sklearn RandomForest |
| Live Weather | WeatherAPI.com | Clear-weather defaults |
| Baseline Scores | Batch CSV | Hardcoded geographic baselines (20 zones) |
| Heatmap Data | Live API | Static CSV → Hardcoded values |
| Zone Config | `data/zones.json` | Hardcoded dictionaries |

---

## Tech Stack

**ML:** XGBoost · LightGBM · scikit-learn · pandas · NumPy  
**API:** FastAPI · Uvicorn · Pydantic  
**Live Data:** WeatherAPI.com  
**Viz:** Folium · Leaflet.js  
**Serialisation:** joblib · PyYAML

---

<p align="center">
  <sub>Built for DevTrails 2026 — Gig Worker Safety</sub>
</p>
