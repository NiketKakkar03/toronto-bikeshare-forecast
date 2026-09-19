"""Held-out evaluation for station-demand models and historical baselines."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from bikeshare_forecast.ml.common import read_json, sha256_file, write_json


def predict_value(model: dict[str, Any], row: dict[str, object]) -> float:
    """Apply a persisted linear demand model, clipping impossible negative counts."""
    values: list[float] = []
    for index, column in enumerate(model["feature_columns"]):
        raw = row[column]
        if not isinstance(raw, int | float):
            raise ValueError(f"non-numeric feature {column}: {raw!r}")
        values.append(
            (float(raw) - model["standardization_means"][index])
            / model["standardization_scales"][index]
        )
    prediction = model["intercept"] + sum(
        weight * value for weight, value in zip(model["weights"], values, strict=True)
    )
    return max(0.0, float(prediction))


def _regression(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    errors = [prediction - target for target, prediction in zip(actual, predicted, strict=True)]
    return {
        "mae": sum(abs(value) for value in errors) / len(errors),
        "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
    }


def evaluate_run(dataset_dir: Path, model_dir: Path, output_dir: Path) -> Path:
    """Evaluate demand models and a recent-demand baseline on the test split."""
    dataset_path = dataset_dir / "dataset.parquet"
    manifest = read_json(dataset_dir / "dataset-manifest.json")
    if manifest["dataset_sha256"] != sha256_file(dataset_path):
        raise ValueError("dataset hash does not match its manifest")
    test = pl.read_parquet(dataset_path).filter(pl.col("split") == "test")
    if test.is_empty():
        raise ValueError("test split is empty")
    rows = test.to_dicts()
    results: dict[str, object] = {}
    model_hashes: list[dict[str, str]] = []
    for horizon_value in manifest["config"]["horizons_minutes"]:
        horizon = int(horizon_value)
        metrics: dict[str, object] = {}
        predictions: dict[str, list[float]] = {}
        for target in ("departures", "arrivals"):
            model_path = model_dir / f"ridge-{target}-{horizon}m.json"
            model = read_json(model_path)
            if model["dataset_sha256"] != manifest["dataset_sha256"]:
                raise ValueError("model was trained from a different dataset")
            actual = [float(row[f"target_{target}_{horizon}m"]) for row in rows]
            predicted = [predict_value(model, row) for row in rows]
            predictions[target] = predicted
            baseline = [float(row[f"{target}_lag_15m"]) * (horizon / 15) for row in rows]
            metrics[target] = {
                "ridge": _regression(actual, predicted),
                "recent_rate": _regression(actual, baseline),
            }
            model_hashes.append({"path": model_path.name, "sha256": sha256_file(model_path)})
        actual_net = [float(row[f"target_net_flow_{horizon}m"]) for row in rows]
        predicted_net = [
            a - d for a, d in zip(predictions["arrivals"], predictions["departures"], strict=True)
        ]
        metrics["net_flow"] = {"derived_ridge": _regression(actual_net, predicted_net)}
        results[f"{horizon}m"] = metrics
    report = {
        "schema_version": 2,
        "forecast_kind": "station_demand",
        "evaluation_split": "test",
        "rows": test.height,
        "dataset_sha256": manifest["dataset_sha256"],
        "models": model_hashes,
        "metrics": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation.json"
    write_json(path, report)
    return path
