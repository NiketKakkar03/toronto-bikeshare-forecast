"""Deterministic ridge-regression station-demand baselines."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from bikeshare_forecast.ml.common import read_json, sha256_file, write_json
from bikeshare_forecast.ml.dataset import FEATURE_COLUMNS


def _fit_ridge(
    values: list[list[float]], labels: list[float]
) -> tuple[list[float], float, list[float], list[float]]:
    count, dimension = len(values), len(values[0])
    means = [sum(row[i] for row in values) / count for i in range(dimension)]
    scales = [
        max((sum((row[i] - means[i]) ** 2 for row in values) / count) ** 0.5, 1e-12)
        for i in range(dimension)
    ]
    standardized = [[(row[i] - means[i]) / scales[i] for i in range(dimension)] for row in values]
    weights, intercept = [0.0] * dimension, sum(labels) / count
    for _ in range(1000):
        errors = [
            intercept + sum(w * x for w, x in zip(weights, row, strict=True)) - label
            for row, label in zip(standardized, labels, strict=True)
        ]
        intercept -= 0.03 * sum(errors) / count
        for i in range(dimension):
            gradient = (
                sum(error * row[i] for error, row in zip(errors, standardized, strict=True)) / count
            )
            weights[i] -= 0.03 * (gradient + 1e-4 * weights[i])
    return weights, intercept, means, scales


def train_models(dataset_dir: Path, output_dir: Path) -> Path:
    """Train departure and arrival count models for each horizon."""
    dataset_path = dataset_dir / "dataset.parquet"
    manifest = read_json(dataset_dir / "dataset-manifest.json")
    expected_hash = manifest.get("dataset_sha256")
    if expected_hash != sha256_file(dataset_path):
        raise ValueError("dataset hash does not match its manifest")
    frame = pl.read_parquet(dataset_path).filter(pl.col("split") == "train")
    if frame.is_empty():
        raise ValueError("training split is empty")
    values = [[float(value) for value in row] for row in frame.select(FEATURE_COLUMNS).iter_rows()]
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for horizon_value in manifest["config"]["horizons_minutes"]:
        horizon = int(horizon_value)
        for target in ("departures", "arrivals"):
            labels = [float(value) for value in frame[f"target_{target}_{horizon}m"].to_list()]
            weights, intercept, means, scales = _fit_ridge(values, labels)
            model = {
                "schema_version": 2,
                "model_type": "deterministic_ridge_regression",
                "target": target,
                "horizon_minutes": horizon,
                "feature_columns": list(FEATURE_COLUMNS),
                "standardization_means": means,
                "standardization_scales": scales,
                "weights": weights,
                "intercept": intercept,
                "training_rows": frame.height,
                "dataset_sha256": expected_hash,
            }
            path = output_dir / f"ridge-{target}-{horizon}m.json"
            write_json(path, model)
            files.append({"path": path.name, "sha256": sha256_file(path)})
    path = output_dir / "model-manifest.json"
    write_json(
        path,
        {
            "schema_version": 2,
            "forecast_kind": "station_demand",
            "dataset_sha256": expected_hash,
            "models": files,
        },
    )
    return path
