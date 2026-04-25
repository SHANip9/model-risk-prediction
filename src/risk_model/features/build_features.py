from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import DATA_DIR, resolve_path

BASE_FEATURE_COLUMNS = [
    "flood_event_count_24m",
    "avg_event_rainfall_mm",
    "max_event_rainfall_mm",
    "curfew_count_qtr",
    "avg_curfew_hours",
    "power_outage_count_month",
    "avg_power_outage_hours",
    "traffic_disruption_hours_week",
    "disruption_days_month",
    "proximity_river_km",
    "coastal_exposure_index",
    "elevation_m",
    "adjacent_zone_risk_prev_week",
]

IDENTITY_COLUMNS = ["zone_id", "week_start"]
TARGET_COLUMN = "risk_score_0_100"
RISK_BIN_COLUMN = "risk_bin"

SCORE_WEIGHTS = {
    "flood_event_count_24m": 0.14,
    "avg_event_rainfall_mm": 0.14,
    "max_event_rainfall_mm": 0.08,
    "curfew_count_qtr": 0.08,
    "avg_curfew_hours": 0.06,
    "power_outage_count_month": 0.08,
    "avg_power_outage_hours": 0.06,
    "traffic_disruption_hours_week": 0.10,
    "disruption_days_month": 0.10,
    "coastal_exposure_index": 0.07,
    "adjacent_zone_risk_prev_week": 0.09,
}


@dataclass(slots=True)
class FeatureBuildConfig:
    input_path: Path = DATA_DIR / "raw" / "zones" / "zone_history.csv"
    output_path: Path = DATA_DIR / "features" / "weekly_zone_features.csv"
    metadata_path: Path = DATA_DIR / "features" / "weekly_zone_features_meta.json"
    generate_sample_if_missing: bool = False
    sample_zone_count: int = 20
    sample_weeks: int = 112


def _min_max_scale(series: pd.Series) -> pd.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    if np.isclose(max_value, min_value):
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - min_value) / (max_value - min_value)


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported source format: {path.suffix}")


def _ensure_required_columns(frame: pd.DataFrame) -> None:
    required = set(IDENTITY_COLUMNS + BASE_FEATURE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    typed = frame.copy()
    typed["week_start"] = pd.to_datetime(typed["week_start"], utc=False)

    for column in BASE_FEATURE_COLUMNS:
        typed[column] = pd.to_numeric(typed[column], errors="coerce")

    typed = typed.dropna(subset=BASE_FEATURE_COLUMNS + ["week_start", "zone_id"])
    typed["zone_id"] = typed["zone_id"].astype(str)
    return typed


def _add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    month_index = enriched["week_start"].dt.month
    enriched["month_sin"] = np.sin(2.0 * np.pi * month_index / 12.0)
    enriched["month_cos"] = np.cos(2.0 * np.pi * month_index / 12.0)
    return enriched


def _add_geographic_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    river_risk = 1.0 / (enriched["proximity_river_km"] + 1.0)
    elevation_risk = 1.0 - _min_max_scale(enriched["elevation_m"])
    coastal_risk = _min_max_scale(enriched["coastal_exposure_index"])
    enriched["geo_risk_index"] = (river_risk * 0.45) + (elevation_risk * 0.25) + (coastal_risk * 0.30)
    return enriched


def _derive_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    enriched = frame.copy()
    if "historical_risk_score" in enriched.columns:
        enriched[TARGET_COLUMN] = pd.to_numeric(enriched["historical_risk_score"], errors="coerce")
        source = "historical_risk_score"
    else:
        weighted_score = np.zeros(len(enriched), dtype=float)
        for column, weight in SCORE_WEIGHTS.items():
            weighted_score += _min_max_scale(enriched[column]).to_numpy() * weight

        weighted_score += _min_max_scale(enriched["geo_risk_index"]).to_numpy() * 0.10
        enriched[TARGET_COLUMN] = weighted_score * 100.0
        source = "heuristic_weighted_proxy"

    enriched[TARGET_COLUMN] = enriched[TARGET_COLUMN].clip(0.0, 100.0)
    enriched[RISK_BIN_COLUMN] = pd.cut(
        enriched[TARGET_COLUMN],
        bins=[-0.1, 33.0, 66.0, 100.0],
        labels=["low", "medium", "high"],
    ).astype(str)

    return enriched, source


def build_feature_table(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    _ensure_required_columns(raw_frame)
    prepared = _coerce_types(raw_frame)
    prepared = _add_temporal_features(prepared)
    prepared = _add_geographic_features(prepared)
    prepared, target_source = _derive_target(prepared)

    ordered_columns = IDENTITY_COLUMNS + BASE_FEATURE_COLUMNS + ["month_sin", "month_cos", "geo_risk_index", TARGET_COLUMN, RISK_BIN_COLUMN]
    prepared = prepared.sort_values(["zone_id", "week_start"]).reset_index(drop=True)
    return prepared[ordered_columns], target_source


def generate_sample_source(path: Path, zone_count: int = 5, weeks: int = 112) -> Path:
    rng = np.random.default_rng(42)
    start = pd.Timestamp.utcnow().normalize() - pd.Timedelta(weeks=weeks)

    rows: list[dict[str, Any]] = []
    for zone_number in range(1, zone_count + 1):
        zone_id = f"zone_{zone_number}"
        base_rain = rng.uniform(25.0, 120.0)
        coastal_exposure = rng.uniform(0.1, 1.0)
        elevation = rng.uniform(5.0, 250.0)
        river_distance = rng.uniform(0.3, 12.0)

        adjacent_risk = rng.uniform(20.0, 70.0)
        for week_idx in range(weeks):
            week_start = start + pd.Timedelta(weeks=week_idx)
            month_factor = 1.0 + 0.7 * np.sin((week_start.month / 12.0) * 2.0 * np.pi)

            avg_rain = max(0.0, rng.normal(base_rain * month_factor, 10.0))
            max_rain = avg_rain + abs(rng.normal(20.0, 8.0))
            flood_events = rng.poisson(max(avg_rain / 90.0, 0.2))
            curfew_count = rng.poisson(0.4)
            avg_curfew_hours = max(0.0, rng.normal(2.2, 1.0))
            outage_count = rng.poisson(1.1 + month_factor * 0.5)
            outage_hours = max(0.2, rng.normal(3.0, 1.1))
            traffic_hours = max(0.5, rng.normal(6.0 + month_factor * 2.5, 1.5))
            disruption_days = max(0, int(rng.normal(4.0 + month_factor * 1.5, 2.0)))

            risk_signal = (
                (avg_rain / 180.0)
                + (flood_events * 0.08)
                + (traffic_hours / 15.0)
                + (disruption_days / 20.0)
                + (coastal_exposure * 0.2)
                + (adjacent_risk / 200.0)
            )
            historical_risk_score = float(np.clip(risk_signal * 55.0, 5.0, 98.0))

            rows.append(
                {
                    "zone_id": zone_id,
                    "week_start": week_start.strftime("%Y-%m-%d"),
                    "flood_event_count_24m": int(flood_events),
                    "avg_event_rainfall_mm": float(round(avg_rain, 3)),
                    "max_event_rainfall_mm": float(round(max_rain, 3)),
                    "curfew_count_qtr": int(curfew_count),
                    "avg_curfew_hours": float(round(avg_curfew_hours, 3)),
                    "power_outage_count_month": int(outage_count),
                    "avg_power_outage_hours": float(round(outage_hours, 3)),
                    "traffic_disruption_hours_week": float(round(traffic_hours, 3)),
                    "disruption_days_month": int(disruption_days),
                    "proximity_river_km": float(round(river_distance, 3)),
                    "coastal_exposure_index": float(round(coastal_exposure, 3)),
                    "elevation_m": float(round(elevation, 3)),
                    "adjacent_zone_risk_prev_week": float(round(adjacent_risk, 3)),
                    "historical_risk_score": historical_risk_score,
                }
            )

            adjacent_risk = float(np.clip((adjacent_risk * 0.8) + historical_risk_score * 0.2 + rng.normal(0.0, 2.0), 0.0, 100.0))

    sample = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    return path


def run_feature_pipeline(config: FeatureBuildConfig) -> dict[str, Any]:
    source_path = resolve_path(config.input_path)
    output_path = resolve_path(config.output_path)
    metadata_path = resolve_path(config.metadata_path)

    if not source_path.exists():
        if not config.generate_sample_if_missing:
            raise FileNotFoundError(
                f"Feature source not found: {source_path}. Add source data or run with --generate-sample-if-missing."
            )
        source_path = generate_sample_source(source_path, config.sample_zone_count, config.sample_weeks)

    frame = _read_frame(source_path)
    feature_table, target_source = build_feature_table(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    feature_table.to_csv(output_path, index=False)

    metadata = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "rows": int(len(feature_table)),
        "zones": int(feature_table["zone_id"].nunique()),
        "features": int(len(BASE_FEATURE_COLUMNS) + 3),
        "target_source": target_source,
        "output_path": str(output_path),
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> FeatureBuildConfig:
    parser = argparse.ArgumentParser(description="Build weekly risk model features")
    parser.add_argument("--input", default=str(DATA_DIR / "raw" / "zones" / "zone_history.csv"))
    parser.add_argument("--output", default=str(DATA_DIR / "features" / "weekly_zone_features.csv"))
    parser.add_argument("--metadata", default=str(DATA_DIR / "features" / "weekly_zone_features_meta.json"))
    parser.add_argument("--generate-sample-if-missing", action="store_true")
    parser.add_argument("--sample-zone-count", type=int, default=20)
    parser.add_argument("--sample-weeks", type=int, default=112)
    args = parser.parse_args()

    return FeatureBuildConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        metadata_path=Path(args.metadata),
        generate_sample_if_missing=args.generate_sample_if_missing,
        sample_zone_count=args.sample_zone_count,
        sample_weeks=args.sample_weeks,
    )


if __name__ == "__main__":
    summary = run_feature_pipeline(parse_args())
    print(json.dumps(summary, indent=2))
