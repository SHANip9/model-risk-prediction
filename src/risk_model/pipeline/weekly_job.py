from __future__ import annotations

import argparse
import json

from src.risk_model.features.build_features import FeatureBuildConfig, run_feature_pipeline
from src.risk_model.inference.score_weekly import ScoreConfig, score_weekly
from src.risk_model.training.train_xgboost import TrainConfig, train_and_save
from src.risk_model.training.validate_lightgbm import ValidationConfig, validate_and_report


def run_weekly_pipeline(generate_sample_if_missing: bool = False) -> dict:
    feature_result = run_feature_pipeline(
        FeatureBuildConfig(generate_sample_if_missing=generate_sample_if_missing)
    )
    train_result = train_and_save(TrainConfig())
    validation_result = validate_and_report(ValidationConfig())
    scoring_result = score_weekly(ScoreConfig())

    return {
        "feature_pipeline": feature_result,
        "xgboost_training": train_result,
        "lightgbm_validation": validation_result,
        "weekly_scoring": scoring_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full weekly Model 1 pipeline")
    parser.add_argument("--generate-sample-if-missing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    payload = run_weekly_pipeline(generate_sample_if_missing=cli_args.generate_sample_if_missing)
    print(json.dumps(payload, indent=2))
