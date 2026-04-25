# OmniSight AI - Model 1 (Risk Prediction)

This project implements Model 1 for weekly zone risk scoring (0-100), plus pricing-plan API endpoints used by the product layer.

## What Is Implemented
- Feature pipeline: `src/risk_model/features/build_features.py`
- XGBoost training: `src/risk_model/training/train_xgboost.py`
- LightGBM cross-check: `src/risk_model/training/validate_lightgbm.py`
- Weekly scoring: `src/risk_model/inference/score_weekly.py`
- Orchestration job: `src/risk_model/pipeline/weekly_job.py`
- Pricing + risk API: `api/app.py`

## Plan Config Source
- `configs/pricing/plans.yaml`

## Install
```powershell
cd "F:\OMNISIGHT AI\Model Risk Prediction"
python -m pip install -r requirements.txt
```

## Run Weekly Pipeline
If raw source is missing, you can bootstrap with sample data once:
```powershell
python -m src.risk_model.pipeline.weekly_job --generate-sample-if-missing
```
If `xgboost` or `lightgbm` is unavailable, the pipeline uses safe development fallbacks and records that in the generated report JSON.

## Run API
```powershell
uvicorn api.app:app --reload
```

## Key Endpoints
- `GET /health`
- `GET /plans`
- `GET /plans/{plan_id}`
- `GET /plans/compare/table`
- `GET /risk/latest`
- `GET /risk/zones/{zone_id}`
