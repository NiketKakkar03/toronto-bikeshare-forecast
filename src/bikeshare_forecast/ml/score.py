"""Batch scoring for persisted station-demand model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import polars as pl
from numpy.typing import NDArray

from bikeshare_forecast.ml.common import read_json, sha256_file, write_json
from bikeshare_forecast.ml.dataset import FEATURE_COLUMNS
from bikeshare_forecast.ml.evaluate import _load_event_probability, predict_value


def _score_ridge(
    frame: pl.DataFrame, model_dir: Path, target: str, horizon: int
) -> NDArray[np.float64]:
    model = read_json(model_dir / f"ridge-{target}-{horizon}m.json")
    return cast(
        NDArray[np.float64],
        np.asarray([predict_value(model, row) for row in frame.to_dicts()], dtype=float),
    )


def _score_histgb(
    frame: pl.DataFrame, model_dir: Path, target: str, horizon: int
) -> NDArray[np.float64]:
    path = model_dir / f"histgb-{target}-{horizon}m.joblib"
    artifact = joblib.load(path)
    features = frame.select(FEATURE_COLUMNS).to_numpy()
    return cast(NDArray[np.float64], np.maximum(0.0, artifact["model"].predict(features)))


def score_batch(
    dataset_dir: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    split: str = "test",
    model_family: str = "hist_gradient_boosting",
) -> Path:
    """Score a dataset split to Parquet and write a reproducibility manifest."""
    dataset_path = dataset_dir / "dataset.parquet"
    manifest = read_json(dataset_dir / "dataset-manifest.json")
    if manifest["dataset_sha256"] != sha256_file(dataset_path):
        raise ValueError("dataset hash does not match its manifest")
    if split not in {"train", "validation", "test", "all"}:
        raise ValueError("split must be train, validation, test, or all")
    if model_family not in {"hist_gradient_boosting", "ridge"}:
        raise ValueError("model_family must be hist_gradient_boosting or ridge")
    frame = pl.read_parquet(dataset_path)
    if split != "all":
        frame = frame.filter(pl.col("split") == split)
    if frame.is_empty():
        raise ValueError(f"dataset split is empty: {split}")
    output = frame.select("station_id", "station_name", "feature_time", "split")
    scored_files: list[dict[str, str]] = []
    for horizon_value in manifest["config"]["horizons_minutes"]:
        horizon = int(horizon_value)
        for target in ("departures", "arrivals"):
            if model_family == "hist_gradient_boosting":
                counts = _score_histgb(frame, model_dir, target, horizon)
                count_path = model_dir / f"histgb-{target}-{horizon}m.joblib"
                features = frame.select(FEATURE_COLUMNS).to_numpy()
                probabilities = _load_event_probability(model_dir, target, horizon, features)
                event_path = model_dir / f"histgb-{target}-event-{horizon}m.joblib"
                if not event_path.exists():
                    event_path = event_path.with_suffix(".json")
                scored_files.extend(
                    [
                        {"path": count_path.name, "sha256": sha256_file(count_path)},
                        {"path": event_path.name, "sha256": sha256_file(event_path)},
                    ]
                )
                output = output.with_columns(
                    pl.Series(f"predicted_{target}_{horizon}m", counts),
                    pl.Series(f"probability_{target}_event_{horizon}m", probabilities),
                )
            else:
                counts = _score_ridge(frame, model_dir, target, horizon)
                count_path = model_dir / f"ridge-{target}-{horizon}m.json"
                scored_files.append({"path": count_path.name, "sha256": sha256_file(count_path)})
                output = output.with_columns(pl.Series(f"predicted_{target}_{horizon}m", counts))
        output = output.with_columns(
            (
                pl.col(f"predicted_arrivals_{horizon}m")
                - pl.col(f"predicted_departures_{horizon}m")
            ).alias(f"predicted_net_flow_{horizon}m")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "predictions.parquet"
    output.write_parquet(path, compression="zstd")
    write_json(
        output_dir / "prediction-manifest.json",
        {
            "schema_version": 1,
            "forecast_kind": "station_demand_batch_predictions",
            "dataset_sha256": manifest["dataset_sha256"],
            "dataset_rows_scored": output.height,
            "split": split,
            "model_family": model_family,
            "models": scored_files,
            "predictions_path": path.name,
            "predictions_sha256": sha256_file(path),
        },
    )
    return path
