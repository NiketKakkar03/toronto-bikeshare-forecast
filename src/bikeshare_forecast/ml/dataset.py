# ruff: noqa: E501
"""Scalable point-in-time station-demand dataset construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from bikeshare_forecast.ml.common import sha256_file, write_json

HORIZONS_MINUTES = (15, 30, 60)
FEATURE_COLUMNS = (
    "minute_sin",
    "minute_cos",
    "weekday_sin",
    "weekday_cos",
    "departures_lag_15m",
    "arrivals_lag_15m",
    "departures_lag_60m",
    "arrivals_lag_60m",
)


@dataclass(frozen=True)
class DatasetConfig:
    """Demand windows, inactive sampling, and chronological split policy."""

    horizons_minutes: tuple[int, ...] = HORIZONS_MINUTES
    interval_minutes: int = 15
    inactive_sample_rate: int = 20
    train_fraction: float = 0.6
    validation_fraction: float = 0.2


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _sql(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _trip_files(root: Path) -> list[Path]:
    files = sorted((root / "ridership" / "parquet").glob("*.parquet"))
    if not files:
        raise ValueError(f"no historical ridership parquet files found in {root}")
    return files


def build_dataset(
    historical_dir: Path, output_dir: Path, config: DatasetConfig | None = None
) -> Path:
    """Stream trip aggregation and sampled station-time features into Parquet."""
    policy = config or DatasetConfig()
    if policy.interval_minutes <= 0 or any(
        h % policy.interval_minutes for h in policy.horizons_minutes
    ):
        raise ValueError("interval must be positive and divide every horizon")
    if policy.inactive_sample_rate < 1:
        raise ValueError("inactive_sample_rate must be positive")
    if not 0 < policy.train_fraction < 1 or not 0 < policy.validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if policy.train_fraction + policy.validation_fraction >= 1:
        raise ValueError("split fractions must leave a test partition")
    files = _trip_files(historical_dir)
    sources = ", ".join(_sql_path(path) for path in files)
    interval = policy.interval_minutes
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.parquet"
    temporary = dataset_path.with_suffix(".parquet.tmp")
    with duckdb.connect() as connection:
        connection.execute(f"CREATE TEMP VIEW trips AS SELECT * FROM read_parquet([{sources}])")
        connection.execute(f"""CREATE TEMP TABLE counts AS
            SELECT station_id, time_bucket(INTERVAL '{interval} minutes', event_time) feature_time,
                   sum(departures)::INTEGER departures, sum(arrivals)::INTEGER arrivals,
                   any_value(station_name) station_name
            FROM (
                SELECT start_station_id station_id, started_at event_time,
                       start_station_name station_name, 1 departures, 0 arrivals FROM trips
                UNION ALL
                SELECT end_station_id, ended_at, end_station_name, 0, 1 FROM trips
            ) GROUP BY station_id, feature_time""")
        bounds = connection.execute(
            "SELECT min(feature_time), max(feature_time) FROM counts"
        ).fetchone()
        if bounds is None:
            raise ValueError("historical ridership contains no event-time bounds")
        first, last = bounds
        if first is None or last is None:
            raise ValueError("historical ridership contains no usable event times")
        train_cutoff = first + (last - first) * policy.train_fraction
        validation_cutoff = first + (last - first) * (
            policy.train_fraction + policy.validation_fraction
        )
        joins: list[str] = []
        targets: list[str] = []
        for horizon in policy.horizons_minutes:
            pieces = horizon // interval
            for offset in range(pieces):
                alias = f"h{horizon}_{offset}"
                joins.append(
                    f"LEFT JOIN counts {alias} ON {alias}.station_id=g.station_id AND {alias}.feature_time=g.feature_time + INTERVAL '{offset * interval} minutes'"
                )
            departures = " + ".join(f"coalesce(h{horizon}_{i}.departures,0)" for i in range(pieces))
            arrivals = " + ".join(f"coalesce(h{horizon}_{i}.arrivals,0)" for i in range(pieces))
            targets.extend(
                (
                    f"g.feature_time + INTERVAL '{horizon} minutes' AS target_time_{horizon}m",
                    f"({departures})::INTEGER AS target_departures_{horizon}m",
                    f"({arrivals})::INTEGER AS target_arrivals_{horizon}m",
                    f"(({arrivals}) - ({departures}))::INTEGER AS target_net_flow_{horizon}m",
                )
            )
        query = f"""COPY (
            WITH stations AS (SELECT station_id, any_value(station_name) station_name FROM counts GROUP BY station_id),
            grid AS (
                SELECT s.station_id, s.station_name, gs feature_time
                FROM stations s, generate_series(cast({_sql(first)} AS TIMESTAMPTZ), cast({_sql(last)} AS TIMESTAMPTZ) - INTERVAL '{max(policy.horizons_minutes)} minutes', INTERVAL '{interval} minutes') t(gs)
            )
            SELECT g.station_id, g.station_name, g.feature_time,
                sin(2*pi()*(extract(hour FROM g.feature_time)*60+extract(minute FROM g.feature_time))/1440) minute_sin,
                cos(2*pi()*(extract(hour FROM g.feature_time)*60+extract(minute FROM g.feature_time))/1440) minute_cos,
                sin(2*pi()*extract(dow FROM g.feature_time)/7) weekday_sin,
                cos(2*pi()*extract(dow FROM g.feature_time)/7) weekday_cos,
                coalesce(l15.departures,0)::INTEGER departures_lag_15m,
                coalesce(l15.arrivals,0)::INTEGER arrivals_lag_15m,
                coalesce(l60.departures,0)::INTEGER departures_lag_60m,
                coalesce(l60.arrivals,0)::INTEGER arrivals_lag_60m,
                {", ".join(targets)},
                CASE WHEN g.feature_time < cast({_sql(train_cutoff)} AS TIMESTAMPTZ) THEN 'train'
                     WHEN g.feature_time < cast({_sql(validation_cutoff)} AS TIMESTAMPTZ) THEN 'validation'
                     ELSE 'test' END split
            FROM grid g
            LEFT JOIN counts l15 ON l15.station_id=g.station_id AND l15.feature_time=g.feature_time-INTERVAL '15 minutes'
            LEFT JOIN counts l60 ON l60.station_id=g.station_id AND l60.feature_time=g.feature_time-INTERVAL '60 minutes'
            {" ".join(joins)}
            WHERE coalesce(l15.departures,0)+coalesce(l15.arrivals,0)+coalesce(l60.departures,0)+coalesce(l60.arrivals,0) > 0
               OR hash(g.station_id, g.feature_time) % {policy.inactive_sample_rate} = 0
            ORDER BY g.feature_time, g.station_id
        ) TO {_sql_path(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"""
        connection.execute(query)
        split_counts = connection.execute(
            f"SELECT count(*), count(*) FILTER (split='train'), count(*) FILTER (split='validation'), count(*) FILTER (split='test') FROM read_parquet({_sql_path(temporary)})"
        ).fetchone()
        if split_counts is None:
            raise ValueError("demand dataset contains no split counts")
        count, train, validation, test = split_counts
    temporary.replace(dataset_path)
    manifest = {
        "schema_version": 2,
        "forecast_kind": "station_demand",
        "config": asdict(policy),
        "feature_columns": list(FEATURE_COLUMNS),
        "targets": ["departures", "arrivals", "net_flow"],
        "row_count": count,
        "split_counts": {"train": train, "validation": validation, "test": test},
        "source_files": [{"path": str(path), "sha256": sha256_file(path)} for path in files],
        "dataset_sha256": sha256_file(dataset_path),
        "point_in_time_rule": "lags use completed trip buckets before feature_time",
        "target_rule": "counts trips in [feature_time, feature_time + horizon)",
        "sampling_rule": "retain rows with recent activity plus deterministic inactive grid sample",
        "split_rule": "global timestamps are split chronologically",
    }
    write_json(output_dir / "dataset-manifest.json", manifest)
    return dataset_path
