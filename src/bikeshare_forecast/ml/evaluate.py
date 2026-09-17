"""Evaluation for classification and inventory baselines."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
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
    ranked = sorted(zip(probabilities, labels, strict=True), reverse=True)
    positive_count = sum(labels)
    seen_positive = 0
    average_precision = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            seen_positive += 1
            average_precision += seen_positive / rank
    average_precision = average_precision / positive_count if positive_count else 0.0
    calibration_error = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        members = [
            (label, probability)
            for label, probability in zip(labels, probabilities, strict=True)
            if lower <= probability < upper or (bin_index == 9 and probability == 1)
        ]
        if members:
            observed = sum(label for label, _ in members) / len(members)
            predicted = sum(probability for _, probability in members) / len(members)
            calibration_error += len(members) / len(labels) * abs(observed - predicted)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": sum((a - b) ** 2 for a, b in zip(labels, probabilities, strict=True))
        / len(labels),
        "log_loss": log_loss,
        "roc_auc": auc,
        "precision_recall_auc": average_precision,
        "expected_calibration_error": calibration_error,
    }


def _regression(
    actual: Sequence[float], predicted: Sequence[float], persistence_mae: float
) -> dict[str, float]:
    errors = [prediction - target for target, prediction in zip(actual, predicted, strict=True)]
    mae = sum(abs(value) for value in errors) / len(errors)
    return {
        "mae": mae,
        "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "mase_vs_persistence": mae / max(persistence_mae, 1.0e-12),
    }


def _seasonal_means(frame: pl.DataFrame, value_column: str) -> dict[tuple[str, int, int], float]:
    """Fit station/weekday/15-minute averages using training rows only."""
    totals: dict[tuple[str, int, int], tuple[float, int]] = {}
    for row in frame.to_dicts():
        timestamp = row["feature_time"]
        if not isinstance(timestamp, datetime):
            raise ValueError("feature_time must be a datetime")
        key = (
            str(row["station_id"]),
            timestamp.weekday(),
            timestamp.hour * 4 + timestamp.minute // 15,
        )
        total, count = totals.get(key, (0.0, 0))
        totals[key] = (total + float(row[value_column]), count + 1)
    return {key: total / count for key, (total, count) in totals.items()}


def _seasonal_predictions(
    rows: list[dict[str, object]], means: dict[tuple[str, int, int], float], fallback: str
) -> list[float]:
    values = []
    for row in rows:
        timestamp = row["feature_time"]
        if not isinstance(timestamp, datetime):
            raise ValueError("feature_time must be a datetime")
        key = (
            str(row["station_id"]),
            timestamp.weekday(),
            timestamp.hour * 4 + timestamp.minute // 15,
        )
        fallback_value = row[fallback]
        if not isinstance(fallback_value, int | float):
            raise ValueError(f"fallback {fallback} must be numeric")
        values.append(means.get(key, float(fallback_value)))
    return values


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
    train = frame.filter(pl.col("split") == "train")
    bike_seasonal = _seasonal_means(train, "bikes_available")
    dock_seasonal = _seasonal_means(train, "docks_available")
    results: dict[str, object] = {}
    model_hashes: list[dict[str, str]] = []
    for horizon_value in dataset_manifest["config"]["horizons_minutes"]:
        horizon = int(horizon_value)
        rows = test.to_dicts()
        classifications: dict[str, Any] = {}
        for target, label_column, current_column in (
            ("empty", f"target_unavailable_{horizon}m", "bikes_available"),
            ("full", f"target_full_{horizon}m", "docks_available"),
        ):
            model_path = model_dir / f"logistic-{target}-{horizon}m.json"
            model = read_json(model_path)
            if model["dataset_sha256"] != dataset_manifest["dataset_sha256"]:
                raise ValueError("model was trained from a different dataset")
            labels = [int(row[label_column]) for row in rows]
            probabilities = [_probability(model, row) for row in rows]
            persistence_probabilities = [float(int(row[current_column] == 0)) for row in rows]
            classifications[target] = {
                "logistic": _classification(labels, probabilities),
                "persistence": _classification(labels, persistence_probabilities),
            }
            model_hashes.append({"path": model_path.name, "sha256": sha256_file(model_path)})
        actual_bikes = [float(row[f"target_bikes_{horizon}m"]) for row in rows]
        actual_docks = [float(row[f"target_docks_{horizon}m"]) for row in rows]
        persistence_bikes = [float(row["bikes_available"]) for row in rows]
        persistence_docks = [float(row["docks_available"]) for row in rows]
        seasonal_bikes = _seasonal_predictions(rows, bike_seasonal, "bikes_available")
        seasonal_docks = _seasonal_predictions(rows, dock_seasonal, "docks_available")
        bike_persistence_mae = sum(
            abs(actual - predicted)
            for actual, predicted in zip(actual_bikes, persistence_bikes, strict=True)
        ) / len(rows)
        dock_persistence_mae = sum(
            abs(actual - predicted)
            for actual, predicted in zip(actual_docks, persistence_docks, strict=True)
        ) / len(rows)
        classifications["empty"]["seasonal"] = _classification(
            [int(row[f"target_unavailable_{horizon}m"]) for row in rows],
            [float(int(value <= 0)) for value in seasonal_bikes],
        )
        classifications["full"]["seasonal"] = _classification(
            [int(row[f"target_full_{horizon}m"]) for row in rows],
            [float(int(value <= 0)) for value in seasonal_docks],
        )
        results[f"{horizon}m"] = {
            "classification": classifications,
            "inventory_regression": {
                "bikes": {
                    "persistence": _regression(
                        actual_bikes, persistence_bikes, bike_persistence_mae
                    ),
                    "seasonal": _regression(actual_bikes, seasonal_bikes, bike_persistence_mae),
                },
                "docks": {
                    "persistence": _regression(
                        actual_docks, persistence_docks, dock_persistence_mae
                    ),
                    "seasonal": _regression(actual_docks, seasonal_docks, dock_persistence_mae),
                },
            },
        }
    report = {
        "schema_version": 1,
        "evaluation_split": "test",
        "rows": test.height,
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "models": model_hashes,
        "seasonal_rule": "training-only station/weekday/15-minute mean; persistence fallback",
        "metrics": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation.json"
    write_json(path, report)
    return path
