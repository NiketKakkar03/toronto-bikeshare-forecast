# ruff: noqa: E501
"""Scalable, station-level diagnostics for held-out demand forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from bikeshare_forecast.ml.common import read_json, sha256_file, write_json
from bikeshare_forecast.ml.evaluate import _prediction_sql


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _records(connection: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, object]]:
    result = connection.execute(query)
    columns = [item[0] for item in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def _metric_select(group_columns: str = "") -> str:
    prefix = f"{group_columns}, " if group_columns else ""
    return f"""{prefix}count(*) row_count,
        avg(abs(prediction-actual)) mae,
        sqrt(avg(pow(prediction-actual, 2))) rmse,
        avg(prediction-actual) bias,
        avg(abs(baseline-actual)) baseline_mae,
        sqrt(avg(pow(baseline-actual, 2))) baseline_rmse"""


def _target_diagnostics(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    prediction: str,
    target: str,
    horizon: int,
    worst_station_limit: int,
) -> dict[str, object]:
    actual = f"target_{target}_{horizon}m"
    baseline = f"{target}_lag_15m*({horizon}/15.0)"
    connection.execute(f"""CREATE OR REPLACE TEMP VIEW diagnostic_rows AS
        SELECT station_id, station_name, feature_time,
               ({actual})::DOUBLE actual, ({prediction})::DOUBLE prediction,
               ({baseline})::DOUBLE baseline,
               target_departures_{horizon}m + target_arrivals_{horizon}m total_demand
        FROM {source} WHERE split='test'""")
    connection.execute("""CREATE OR REPLACE TEMP TABLE station_activity AS
        SELECT station_id, sum(total_demand) observed_demand,
               CASE ntile(3) OVER (ORDER BY sum(total_demand), station_id)
                   WHEN 1 THEN 'quiet' WHEN 2 THEN 'medium' ELSE 'busy' END activity_segment
        FROM diagnostic_rows GROUP BY station_id""")
    stations = _records(
        connection,
        f"""SELECT {_metric_select("d.station_id, any_value(d.station_name) station_name, any_value(a.activity_segment) activity_segment, any_value(a.observed_demand) observed_demand")}
            FROM diagnostic_rows d JOIN station_activity a USING (station_id)
            GROUP BY d.station_id ORDER BY mae DESC, d.station_id""",
    )
    segments = _records(
        connection,
        f"""SELECT {_metric_select("a.activity_segment")}, count(DISTINCT d.station_id) stations
            FROM diagnostic_rows d JOIN station_activity a USING (station_id)
            GROUP BY a.activity_segment
            ORDER BY CASE a.activity_segment WHEN 'quiet' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END""",
    )
    hours = _records(
        connection,
        f"""SELECT {_metric_select("extract(hour FROM timezone('America/Toronto', feature_time))::INTEGER hour_of_day")}
            FROM diagnostic_rows GROUP BY hour_of_day ORDER BY hour_of_day""",
    )
    weekdays = _records(
        connection,
        f"""SELECT {_metric_select("extract(dow FROM timezone('America/Toronto', feature_time))::INTEGER weekday")}
            FROM diagnostic_rows GROUP BY weekday ORDER BY weekday""",
    )
    rush_hours = _records(
        connection,
        f"""SELECT {_metric_select("CASE WHEN extract(dow FROM timezone('America/Toronto', feature_time)) BETWEEN 1 AND 5 AND (extract(hour FROM timezone('America/Toronto', feature_time)) BETWEEN 7 AND 9 OR extract(hour FROM timezone('America/Toronto', feature_time)) BETWEEN 16 AND 18) THEN 'rush_hour' ELSE 'other' END period")}
            FROM diagnostic_rows GROUP BY period ORDER BY period""",
    )
    beat = connection.execute("""WITH station_scores AS (
        SELECT station_id, avg(abs(prediction-actual)) mae,
               avg(abs(baseline-actual)) baseline_mae
        FROM diagnostic_rows GROUP BY station_id)
        SELECT count(*) FILTER (mae < baseline_mae), count(*),
               100.0 * count(*) FILTER (mae < baseline_mae) / count(*)
        FROM station_scores""").fetchone()
    if beat is None:
        raise ValueError("station comparison metrics are unavailable")
    return {
        "stations": stations,
        "activity_segments": segments,
        "hour_of_day": hours,
        "weekday": weekdays,
        "rush_hour": rush_hours,
        "stations_beating_baseline": int(beat[0]),
        "station_count": int(beat[1]),
        "stations_beating_baseline_percent": float(beat[2]),
        "worst_stations": stations[:worst_station_limit],
    }


def diagnostics_run(
    dataset_dir: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    worst_station_limit: int = 10,
) -> Path:
    """Write segmented test diagnostics using DuckDB aggregation throughout."""
    if worst_station_limit < 1:
        raise ValueError("worst_station_limit must be positive")
    dataset_path = dataset_dir / "dataset.parquet"
    manifest = read_json(dataset_dir / "dataset-manifest.json")
    if manifest["dataset_sha256"] != sha256_file(dataset_path):
        raise ValueError("dataset hash does not match its manifest")
    source = f"read_parquet({_sql_path(dataset_path)})"
    results: dict[str, object] = {}
    models: list[dict[str, str]] = []
    with duckdb.connect() as connection:
        test_rows = connection.execute(
            f"SELECT count(*) FROM {source} WHERE split='test'"
        ).fetchone()
        if test_rows is None or int(test_rows[0]) == 0:
            raise ValueError("test split is empty")
        for horizon_value in manifest["config"]["horizons_minutes"]:
            horizon = int(horizon_value)
            horizon_results: dict[str, object] = {}
            for target in ("departures", "arrivals"):
                model_path = model_dir / f"ridge-{target}-{horizon}m.json"
                model: dict[str, Any] = read_json(model_path)
                if model["dataset_sha256"] != manifest["dataset_sha256"]:
                    raise ValueError("model was trained from a different dataset")
                horizon_results[target] = _target_diagnostics(
                    connection,
                    source,
                    _prediction_sql(model),
                    target,
                    horizon,
                    worst_station_limit,
                )
                models.append({"path": model_path.name, "sha256": sha256_file(model_path)})
            results[f"{horizon}m"] = horizon_results
    report = {
        "schema_version": 1,
        "forecast_kind": "station_demand_diagnostics",
        "evaluation_split": "test",
        "rows": int(test_rows[0]),
        "dataset_sha256": manifest["dataset_sha256"],
        "activity_segment_rule": "tertiles of station total observed departures plus arrivals in the test split",
        "rush_hour_rule": "weekdays 07:00-09:59 and 16:00-18:59 America/Toronto",
        "models": models,
        "diagnostics": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "diagnostics.json"
    write_json(path, report)
    return path
