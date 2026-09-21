# ruff: noqa: E501
"""Streaming held-out evaluation for station-demand models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

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
                    "recent_rate": _metrics(connection, source, actual, baseline),
                    "prediction_intervals": _prediction_interval_metrics(
                        connection, source, actual, prediction
                    ),
                }
                model_hashes.append({"path": model_path.name, "sha256": sha256_file(model_path)})
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
