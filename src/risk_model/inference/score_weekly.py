from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.common.config import ARTIFACTS_DIR, DATA_DIR, resolve_path


@dataclass(slots=True)
class ScoreConfig:
    features_path: Path = DATA_DIR / "features" / "weekly_zone_features.csv"
    model_bundle_path: Path = ARTIFACTS_DIR / "models" / "risk_xgboost_bundle.joblib"
    output_path: Path = DATA_DIR / "processed" / "zone_risk_scores_latest.csv"


def score_weekly(config: ScoreConfig) -> dict[str, Any]:
    features_path = resolve_path(config.features_path)
    model_bundle_path = resolve_path(config.model_bundle_path)
    output_path = resolve_path(config.output_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not model_bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_bundle_path}")

    frame = pd.read_csv(features_path)
    frame["week_start"] = pd.to_datetime(frame["week_start"], utc=False)

    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    missing_columns = sorted(set(feature_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Scoring frame missing feature columns: {', '.join(missing_columns)}")

    predictions = np.clip(model.predict(frame[feature_columns]), 0.0, 100.0)

    scored = frame[["zone_id", "week_start"]].copy()
    scored["risk_score_0_100"] = predictions
    scored["risk_bin"] = pd.cut(
        scored["risk_score_0_100"],
        bins=[-0.1, 33.0, 66.0, 100.0],
        labels=["low", "medium", "high"],
    ).astype(str)
    scored["scored_utc"] = pd.Timestamp.utcnow().isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    summary = {
        "rows": int(len(scored)),
        "zones": int(scored["zone_id"].nunique()),
        "output_path": str(output_path),
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    return summary


def parse_args() -> ScoreConfig:
    parser = argparse.ArgumentParser(description="Score weekly zone risk from trained model")
    parser.add_argument("--features", default=str(DATA_DIR / "features" / "weekly_zone_features.csv"))
    parser.add_argument("--model-bundle", default=str(ARTIFACTS_DIR / "models" / "risk_xgboost_bundle.joblib"))
    parser.add_argument("--output", default=str(DATA_DIR / "processed" / "zone_risk_scores_latest.csv"))
    args = parser.parse_args()

    return ScoreConfig(
        features_path=Path(args.features),
        model_bundle_path=Path(args.model_bundle),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    print(json.dumps(score_weekly(parse_args()), indent=2))
