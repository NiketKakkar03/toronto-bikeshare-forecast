"""Evaluation for classification and inventory baselines."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from bikeshare_forecast.ml.common import read_json, sha256_file, write_json


def _probability(model: dict[str, Any], row: dict[str, object]) -> float:
    values = []
    for index, column in enumerate(model["feature_columns"]):
        raw = row[column]
        if raw is not None and not isinstance(raw, int | float):
            raise ValueError(f"non-numeric feature {column}: {raw!r}")
        value = model["imputation_means"][index] if raw is None else float(raw)
        values.append(
            (value - model["standardization_means"][index]) / model["standardization_scales"][index]
        )
    score = model["intercept"] + sum(
        weight * value for weight, value in zip(model["weights"], values, strict=True)
    )
    return 1 / (1 + math.exp(-max(min(score, 35), -35)))


def _classification(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    predictions = [int(value >= 0.5) for value in probabilities]
    true_positive = sum(a == b == 1 for a, b in zip(labels, predictions, strict=True))
    false_positive = sum(a == 0 and b == 1 for a, b in zip(labels, predictions, strict=True))
    false_negative = sum(a == 1 and b == 0 for a, b in zip(labels, predictions, strict=True))
    accuracy = sum(a == b for a, b in zip(labels, predictions, strict=True)) / len(labels)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    clipped = [min(max(value, 1.0e-15), 1 - 1.0e-15) for value in probabilities]
    log_loss = -sum(
        label * math.log(value) + (1 - label) * math.log(1 - value)
        for label, value in zip(labels, clipped, strict=True)
    ) / len(labels)
    positives = [value for label, value in zip(labels, probabilities, strict=True) if label == 1]
    negatives = [value for label, value in zip(labels, probabilities, strict=True) if label == 0]
    auc = (
        sum(
            (positive > negative) + 0.5 * (positive == negative)
            for positive in positives
            for negative in negatives
        )
        / (len(positives) * len(negatives))
        if positives and negatives
        else 0.5
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": sum((a - b) ** 2 for a, b in zip(labels, probabilities, strict=True))
        / len(labels),
        "log_loss": log_loss,
        "roc_auc": auc,
    }


def _regression(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    errors = [prediction - target for target, prediction in zip(actual, predicted, strict=True)]
    return {
        "mae": sum(abs(value) for value in errors) / len(errors),
        "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
    }


def evaluate_run(dataset_dir: Path, model_dir: Path, output_dir: Path) -> Path:
    """Evaluate persisted models and inventory baselines on the held-out test split."""
    dataset_path = dataset_dir / "dataset.parquet"
    dataset_manifest = read_json(dataset_dir / "dataset-manifest.json")
    if dataset_manifest["dataset_sha256"] != sha256_file(dataset_path):
        raise ValueError("dataset hash does not match its manifest")
    frame = pl.read_parquet(dataset_path)
    test = frame.filter(pl.col("split") == "test")
    if test.is_empty():
        raise ValueError("test split is empty")
    history: dict[tuple[str, datetime], float] = {
        (str(row["station_id"]), row["feature_time"]): float(row["bikes_available"])
        for row in frame.to_dicts()
    }
    results: dict[str, object] = {}
    model_hashes: list[dict[str, str]] = []
    for horizon_value in dataset_manifest["config"]["horizons_minutes"]:
        horizon = int(horizon_value)
        model_path = model_dir / f"logistic-{horizon}m.json"
        model = read_json(model_path)
        if model["dataset_sha256"] != dataset_manifest["dataset_sha256"]:
            raise ValueError("model was trained from a different dataset")
        rows = test.to_dicts()
        labels = [int(row[f"target_unavailable_{horizon}m"]) for row in rows]
        probabilities = [_probability(model, row) for row in rows]
        actual = [float(row[f"target_bikes_{horizon}m"]) for row in rows]
        persistence = [float(row["bikes_available"]) for row in rows]
        seasonal = [
            history.get(
                (str(row["station_id"]), row["feature_time"] - timedelta(days=7)),
                float(row["bikes_available"]),
            )
            for row in rows
        ]
        results[f"{horizon}m"] = {
            "classification": {"logistic": _classification(labels, probabilities)},
            "inventory_regression": {
                "persistence": _regression(actual, persistence),
                "seasonal_7d": _regression(actual, seasonal),
            },
        }
        model_hashes.append({"path": model_path.name, "sha256": sha256_file(model_path)})
    report = {
        "schema_version": 1,
        "evaluation_split": "test",
        "rows": test.height,
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "models": model_hashes,
        "seasonal_fallback": "persistence when no exact station value exists seven days earlier",
        "metrics": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation.json"
    write_json(path, report)
    return path
