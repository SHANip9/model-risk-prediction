from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.config import ARTIFACTS_DIR, DATA_DIR, resolve_path

TARGET_COLUMN = "risk_score_0_100"
IDENTITY_COLUMNS = {"zone_id", "week_start", "risk_bin"}


@dataclass(slots=True)
class TrainConfig:
    features_path: Path = DATA_DIR / "features" / "weekly_zone_features.csv"
    model_out_path: Path = ARTIFACTS_DIR / "models" / "risk_xgboost_bundle.joblib"
    metrics_out_path: Path = ARTIFACTS_DIR / "reports" / "train_metrics_xgboost.json"
    test_fraction: float = 0.2
    random_state: int = 42


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported features format: {path.suffix}")

    if "week_start" in frame.columns:
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


def _build_xgboost_model(random_state: int):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        from sklearn.ensemble import HistGradientBoostingRegressor

        fallback = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=500,
            random_state=random_state,
        )
        return fallback, "hist_gradient_boosting_fallback", str(exc)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        objective="reg:squarederror",
    )
    return model, "xgboost", None


def train_and_save(config: TrainConfig) -> dict[str, Any]:
    features_path = resolve_path(config.features_path)
    model_out_path = resolve_path(config.model_out_path)
    metrics_out_path = resolve_path(config.metrics_out_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    frame = _load_frame(features_path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing training target column: {TARGET_COLUMN}")

    feature_columns = _select_feature_columns(frame)
    if not feature_columns:
        raise ValueError("No numeric feature columns available for model training")

    train_frame, valid_frame = _temporal_split(frame, config.test_fraction)

    x_train = train_frame[feature_columns]
    y_train = train_frame[TARGET_COLUMN]
    x_valid = valid_frame[feature_columns]
    y_valid = valid_frame[TARGET_COLUMN]

    model, model_backend, fallback_reason = _build_xgboost_model(config.random_state)
    model.fit(x_train, y_train)

    valid_predictions = np.clip(model.predict(x_valid), 0.0, 100.0)

    rmse = float(np.sqrt(mean_squared_error(y_valid, valid_predictions)))
    mae = float(mean_absolute_error(y_valid, valid_predictions))
    r2 = float(r2_score(y_valid, valid_predictions))

    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "model_backend": model_backend,
        "trained_rows": int(len(train_frame)),
        "validated_rows": int(len(valid_frame)),
        "trained_utc": pd.Timestamp.utcnow().isoformat(),
    }
    joblib.dump(bundle, model_out_path)

    metrics = {
        "model": model_backend,
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        },
        "feature_count": len(feature_columns),
        "trained_rows": int(len(train_frame)),
        "validated_rows": int(len(valid_frame)),
        "feature_columns": feature_columns,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "artifact": str(model_out_path),
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }
    if fallback_reason:
        metrics["fallback_reason"] = fallback_reason

    metrics_out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Model 1 XGBoost risk model")
    parser.add_argument("--features", default=str(DATA_DIR / "features" / "weekly_zone_features.csv"))
    parser.add_argument("--model-out", default=str(ARTIFACTS_DIR / "models" / "risk_xgboost_bundle.joblib"))
    parser.add_argument("--metrics-out", default=str(ARTIFACTS_DIR / "reports" / "train_metrics_xgboost.json"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    return TrainConfig(
        features_path=Path(args.features),
        model_out_path=Path(args.model_out),
        metrics_out_path=Path(args.metrics_out),
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    report = train_and_save(parse_args())
    print(json.dumps(report, indent=2))
