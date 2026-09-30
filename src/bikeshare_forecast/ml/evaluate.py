# ruff: noqa: E501
"""Streaming held-out evaluation for station-demand models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import joblib  # type: ignore[import-untyped]
import numpy as np
import polars as pl
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from bikeshare_forecast.ml.common import read_json, sha256_file, write_json
from bikeshare_forecast.ml.dataset import FEATURE_COLUMNS


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


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _prediction_sql(model: dict[str, Any]) -> str:
    pieces = [str(float(model["intercept"]))]
    for index, column in enumerate(model["feature_columns"]):
        mean = float(model["standardization_means"][index])
        scale = float(model["standardization_scales"][index])
        weight = float(model["weights"][index])
        pieces.append(f"({weight})*((CAST({column} AS DOUBLE)-({mean}))/({scale}))")
    return f"greatest(0.0, {' + '.join(pieces)})"


def _metrics(
    connection: duckdb.DuckDBPyConnection, source: str, actual: str, predicted: str
) -> dict[str, float]:
    values = connection.execute(
        f"SELECT avg(abs(({predicted})-({actual}))), sqrt(avg(pow(({predicted})-({actual}),2))) FROM {source} WHERE split='test'"
    ).fetchone()
    if values is None:
        raise ValueError("evaluation metrics are unavailable")
    return {"mae": float(values[0]), "rmse": float(values[1])}


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def _classification_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    predicted = probability >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", zero_division=0
    )
    report: dict[str, object] = {
        "positive_rate": float(np.mean(actual)),
        "mean_probability": float(np.mean(probability)),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "brier_score": float(brier_score_loss(actual, probability)),
    }
    if len(np.unique(actual)) == 2:
        report["roc_auc"] = float(roc_auc_score(actual, probability))
        report["pr_auc"] = float(average_precision_score(actual, probability))
    else:
        report["roc_auc"] = None
        report["pr_auc"] = None
        report["degraded_reason"] = "test labels contain a single class"
    return report


def _load_event_probability(
    model_dir: Path, target: str, horizon: int, features: np.ndarray
) -> np.ndarray:
    path = model_dir / f"histgb-{target}-event-{horizon}m.joblib"
    if path.exists():
        artifact = joblib.load(path)
        probabilities = artifact["model"].predict_proba(features)[:, 1]
        return np.asarray(probabilities, dtype=float)
    constant = read_json(path.with_suffix(".json"))
    return np.full(features.shape[0], float(constant["probability"]), dtype=float)


def _boosted_metrics(
    frame: pl.DataFrame, model_dir: Path, target: str, horizon: int
) -> dict[str, object]:
    features = frame.select(FEATURE_COLUMNS).to_numpy()
    actual_counts = frame[f"target_{target}_{horizon}m"].to_numpy()
    regressor_path = model_dir / f"histgb-{target}-{horizon}m.joblib"
    regressor = joblib.load(regressor_path)
    predicted_counts = np.maximum(0.0, regressor["model"].predict(features))
    probability = _load_event_probability(model_dir, target, horizon, features)
    return {
        "regression": _regression_metrics(actual_counts, predicted_counts),
        "demand_event": _classification_metrics(actual_counts > 0, probability),
    }


def _prediction_interval_metrics(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    actual: str,
    predicted: str,
) -> dict[str, dict[str, float]]:
    intervals: dict[str, dict[str, float]] = {}
    for coverage in (0.8, 0.9):
        width = connection.execute(
            f"SELECT quantile_cont(abs(({predicted})-({actual})), {coverage}) "
            f"FROM {source} WHERE split='validation'"
        ).fetchone()
        if width is None or width[0] is None:
            raise ValueError("validation split is empty; cannot calibrate prediction intervals")
        half_width = float(width[0])
        test = connection.execute(
            f"""SELECT
                avg((({actual}) BETWEEN ({predicted})-({half_width})
                    AND ({predicted})+({half_width}))::INTEGER),
                avg(2.0*({half_width}))
                FROM {source} WHERE split='test'"""
        ).fetchone()
        if test is None:
            raise ValueError("test interval metrics are unavailable")
        intervals[f"p{int(coverage * 100)}"] = {
            "nominal_coverage": coverage,
            "empirical_coverage": float(test[0]),
            "mean_width": float(test[1]),
            "half_width": half_width,
        }
    return intervals


def evaluate_run(dataset_dir: Path, model_dir: Path, output_dir: Path) -> Path:
    """Evaluate demand models and recent-rate baselines without loading all rows."""
    dataset_path = dataset_dir / "dataset.parquet"
    manifest = read_json(dataset_dir / "dataset-manifest.json")
    if manifest["dataset_sha256"] != sha256_file(dataset_path):
        raise ValueError("dataset hash does not match its manifest")
    source = f"read_parquet({_sql_path(dataset_path)})"
    results: dict[str, object] = {}
    model_hashes: list[dict[str, str]] = []
    test_frame = pl.read_parquet(dataset_path).filter(pl.col("split") == "test")
    with duckdb.connect() as connection:
        rows_value = connection.execute(
            f"SELECT count(*) FROM {source} WHERE split='test'"
        ).fetchone()
        if rows_value is None or int(rows_value[0]) == 0:
            raise ValueError("test split is empty")
        test_rows = int(rows_value[0])
        for horizon_value in manifest["config"]["horizons_minutes"]:
            horizon = int(horizon_value)
            metrics: dict[str, object] = {}
            predictions: dict[str, str] = {}
            for target in ("departures", "arrivals"):
                model_path = model_dir / f"ridge-{target}-{horizon}m.json"
                model = read_json(model_path)
                if model["dataset_sha256"] != manifest["dataset_sha256"]:
                    raise ValueError("model was trained from a different dataset")
                prediction = _prediction_sql(model)
                predictions[target] = prediction
                actual = f"target_{target}_{horizon}m"
                baseline = f"{target}_lag_15m*({horizon}/15.0)"
                metrics[target] = {
                    "ridge": _metrics(connection, source, actual, prediction),
                    "hist_gradient_boosting": _boosted_metrics(
                        test_frame, model_dir, target, horizon
                    ),
                    "recent_rate": _metrics(connection, source, actual, baseline),
                    "prediction_intervals": _prediction_interval_metrics(
                        connection, source, actual, prediction
                    ),
                }
                model_hashes.append({"path": model_path.name, "sha256": sha256_file(model_path)})
                for suffix in (
                    f"histgb-{target}-{horizon}m.joblib",
                    f"histgb-{target}-event-{horizon}m.joblib",
                ):
                    artifact_path = model_dir / suffix
                    if not artifact_path.exists():
                        artifact_path = artifact_path.with_suffix(".json")
                    model_hashes.append(
                        {"path": artifact_path.name, "sha256": sha256_file(artifact_path)}
                    )
            actual_net = f"target_net_flow_{horizon}m"
            predicted_net = f"({predictions['arrivals']})-({predictions['departures']})"
            metrics["net_flow"] = {
                "derived_ridge": _metrics(connection, source, actual_net, predicted_net),
                "prediction_intervals": _prediction_interval_metrics(
                    connection, source, actual_net, predicted_net
                ),
            }
            results[f"{horizon}m"] = metrics
    report = {
        "schema_version": 2,
        "forecast_kind": "station_demand",
        "evaluation_split": "test",
        "rows": test_rows,
        "dataset_sha256": manifest["dataset_sha256"],
        "models": model_hashes,
        "metrics": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation.json"
    write_json(path, report)
    return path
