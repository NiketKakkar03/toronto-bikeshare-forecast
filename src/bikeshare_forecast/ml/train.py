"""Deterministic logistic classification baseline training."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from bikeshare_forecast.ml.common import read_json, sha256_file, write_json
from bikeshare_forecast.ml.dataset import FEATURE_COLUMNS


def _matrix(frame: pl.DataFrame, means: list[float]) -> list[list[float]]:
    rows: list[list[float]] = []
    for row in frame.select(FEATURE_COLUMNS).iter_rows():
        rows.append(
            [means[index] if value is None else float(value) for index, value in enumerate(row)]
        )
    return rows


def _fit_logistic(
    values: list[list[float]], labels: list[int], *, iterations: int = 800, rate: float = 0.08
) -> tuple[list[float], float, list[float], list[float]]:
    """Fit standardized L2 logistic regression with fixed batch gradient descent."""
    count = len(values)
    dimension = len(values[0])
    means = [sum(row[column] for row in values) / count for column in range(dimension)]
    scales = []
    for column, mean in enumerate(means):
        variance = sum((row[column] - mean) ** 2 for row in values) / count
        scales.append(max(math.sqrt(variance), 1.0e-12))
    standardized = [
        [(row[column] - means[column]) / scales[column] for column in range(dimension)]
        for row in values
    ]
    weights = [0.0] * dimension
    prevalence = min(max(sum(labels) / count, 1.0e-6), 1 - 1.0e-6)
    intercept = math.log(prevalence / (1 - prevalence))
    regularization = 1.0e-4
    for _ in range(iterations):
        gradient = [0.0] * dimension
        intercept_gradient = 0.0
        for row, label in zip(standardized, labels, strict=True):
            score = max(
                min(intercept + sum(w * x for w, x in zip(weights, row, strict=True)), 35), -35
            )
            error = 1 / (1 + math.exp(-score)) - label
            intercept_gradient += error
            for column in range(dimension):
                gradient[column] += error * row[column]
        intercept -= rate * intercept_gradient / count
        for column in range(dimension):
            weights[column] -= rate * (gradient[column] / count + regularization * weights[column])
    return weights, intercept, means, scales


def train_models(dataset_dir: Path, output_dir: Path) -> Path:
    """Train one binary unavailability classifier per configured horizon."""
    dataset_path = dataset_dir / "dataset.parquet"
    manifest = read_json(dataset_dir / "dataset-manifest.json")
    expected_hash = manifest.get("dataset_sha256")
    if expected_hash != sha256_file(dataset_path):
        raise ValueError("dataset hash does not match its manifest")
    frame = pl.read_parquet(dataset_path).filter(pl.col("split") == "train")
    if frame.is_empty():
        raise ValueError("training split is empty")
    horizons = manifest["config"]["horizons_minutes"]
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for horizon_value in horizons:
        horizon = int(horizon_value)
        label_column = f"target_unavailable_{horizon}m"
        raw_rows = frame.select(FEATURE_COLUMNS).to_dicts()
        imputation_means = [
            sum(float(row[column]) for row in raw_rows if row[column] is not None)
            / sum(row[column] is not None for row in raw_rows)
            for column in FEATURE_COLUMNS
        ]
        values = _matrix(frame, imputation_means)
        labels = [int(value) for value in frame[label_column].to_list()]
        weights, intercept, means, scales = _fit_logistic(values, labels)
        model = {
            "schema_version": 1,
            "model_type": "deterministic_batch_logistic_regression",
            "horizon_minutes": horizon,
            "feature_columns": list(FEATURE_COLUMNS),
            "imputation_means": imputation_means,
            "standardization_means": means,
            "standardization_scales": scales,
            "weights": weights,
            "intercept": intercept,
            "decision_threshold": 0.5,
            "training_rows": frame.height,
            "positive_rows": sum(labels),
            "dataset_sha256": expected_hash,
        }
        path = output_dir / f"logistic-{horizon}m.json"
        write_json(path, model)
        files.append({"path": path.name, "sha256": sha256_file(path)})
    model_manifest = {
        "schema_version": 1,
        "dataset_sha256": expected_hash,
        "models": files,
    }
    path = output_dir / "model-manifest.json"
    write_json(path, model_manifest)
    return path
