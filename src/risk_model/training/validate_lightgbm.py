from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.config import ARTIFACTS_DIR, DATA_DIR, resolve_path

TARGET_COLUMN = "risk_score_0_100"
IDENTITY_COLUMNS = {"zone_id", "week_start", "risk_bin"}


@dataclass(slots=True)
class ValidationConfig:
    features_path: Path = DATA_DIR / "features" / "weekly_zone_features.csv"
    xgboost_metrics_path: Path = ARTIFACTS_DIR / "reports" / "train_metrics_xgboost.json"
    report_out_path: Path = ARTIFACTS_DIR / "reports" / "validation_lightgbm_vs_xgboost.json"
    test_fraction: float = 0.2
    random_state: int = 42
    rmse_tolerance_ratio: float = 0.10


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported features format: {path.suffix}")

    frame["week_start"] = pd.to_datetime(frame["week_start"], utc=False)
    return frame


def _select_feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric if column != TARGET_COLUMN and column not in IDENTITY_COLUMNS]


def _temporal_split(frame: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_frame = frame.sort_values("week_start").reset_index(drop=True)
    split_idx = int(len(sorted_frame) * (1.0 - test_fraction))
    split_idx = min(max(split_idx, 1), len(sorted_frame) - 1)
    return sorted_frame.iloc[:split_idx], sorted_frame.iloc[split_idx:]


def _build_lightgbm_model(random_state: int):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        from sklearn.ensemble import RandomForestRegressor

        fallback = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
        )
        return fallback, "random_forest_fallback", str(exc)

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
    )
    return model, "lightgbm", None


def validate_and_report(config: ValidationConfig) -> dict[str, Any]:
    features_path = resolve_path(config.features_path)
    xgb_metrics_path = resolve_path(config.xgboost_metrics_path)
    report_out_path = resolve_path(config.report_out_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    frame = _load_frame(features_path)
    feature_columns = _select_feature_columns(frame)
    if not feature_columns:
        raise ValueError("No numeric feature columns available for validation")

    train_frame, valid_frame = _temporal_split(frame, config.test_fraction)

    model, model_backend, fallback_reason = _build_lightgbm_model(config.random_state)
    model.fit(train_frame[feature_columns], train_frame[TARGET_COLUMN])
    predictions = np.clip(model.predict(valid_frame[feature_columns]), 0.0, 100.0)

    lgb_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(valid_frame[TARGET_COLUMN], predictions))),
        "mae": float(mean_absolute_error(valid_frame[TARGET_COLUMN], predictions)),
        "r2": float(r2_score(valid_frame[TARGET_COLUMN], predictions)),
    }

    xgb_rmse = None
    if xgb_metrics_path.exists():
        xgb_payload = json.loads(xgb_metrics_path.read_text(encoding="utf-8"))
        xgb_rmse = float(xgb_payload.get("metrics", {}).get("rmse", 0.0))

    status = "no_baseline"
    rmse_gap_ratio = None
    if xgb_rmse and xgb_rmse > 0.0:
        rmse_gap_ratio = (lgb_metrics["rmse"] - xgb_rmse) / xgb_rmse
        status = "pass" if rmse_gap_ratio <= config.rmse_tolerance_ratio else "fail"

    report = {
        "model": model_backend,
        "metrics": lgb_metrics,
        "xgboost_baseline_rmse": xgb_rmse,
        "rmse_gap_ratio": rmse_gap_ratio,
        "status": status,
        "tolerance_ratio": config.rmse_tolerance_ratio,
        "feature_count": len(feature_columns),
        "validated_rows": int(len(valid_frame)),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }
    if fallback_reason:
        report["fallback_reason"] = fallback_reason

    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    report_out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> ValidationConfig:
    parser = argparse.ArgumentParser(description="Validate Model 1 with LightGBM")
    parser.add_argument("--features", default=str(DATA_DIR / "features" / "weekly_zone_features.csv"))
    parser.add_argument("--xgb-metrics", default=str(ARTIFACTS_DIR / "reports" / "train_metrics_xgboost.json"))
    parser.add_argument("--report-out", default=str(ARTIFACTS_DIR / "reports" / "validation_lightgbm_vs_xgboost.json"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rmse-tolerance-ratio", type=float, default=0.10)
    args = parser.parse_args()

    return ValidationConfig(
        features_path=Path(args.features),
        xgboost_metrics_path=Path(args.xgb_metrics),
        report_out_path=Path(args.report_out),
        test_fraction=args.test_fraction,
        random_state=args.random_state,
        rmse_tolerance_ratio=args.rmse_tolerance_ratio,
    )


if __name__ == "__main__":
    result = validate_and_report(parse_args())
    print(json.dumps(result, indent=2))
