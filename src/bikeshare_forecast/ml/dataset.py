"""Point-in-time station-demand dataset construction from historical trips."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

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
    """Demand windows and chronological split policy."""

    horizons_minutes: tuple[int, ...] = HORIZONS_MINUTES
    interval_minutes: int = 15
    train_fraction: float = 0.6
    validation_fraction: float = 0.2


def _trip_files(root: Path) -> list[Path]:
    directory = root / "ridership" / "parquet"
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise ValueError(f"no historical ridership parquet files found in {directory}")
    return files


def _bucket(value: datetime, minutes: int) -> datetime:
    return value.replace(minute=(value.minute // minutes) * minutes, second=0, microsecond=0)


def build_dataset(
    historical_dir: Path, output_dir: Path, config: DatasetConfig | None = None
) -> Path:
    """Build demand features and future trip-count targets without future leakage."""
    policy = config or DatasetConfig()
    if policy.interval_minutes <= 0 or any(
        h % policy.interval_minutes for h in policy.horizons_minutes
    ):
        raise ValueError("interval must be positive and divide every horizon")
    if not 0 < policy.train_fraction < 1 or not 0 < policy.validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if policy.train_fraction + policy.validation_fraction >= 1:
        raise ValueError("split fractions must leave a test partition")
    source_files = _trip_files(historical_dir)
    trips = pl.concat([pl.read_parquet(path) for path in source_files])
    counts: dict[tuple[str, datetime], list[int]] = defaultdict(lambda: [0, 0])
    names: dict[str, str] = {}
    for trip in trips.to_dicts():
        started, ended = trip["started_at"], trip["ended_at"]
        if not isinstance(started, datetime) or not isinstance(ended, datetime):
            raise ValueError("historical trip timestamps must be datetimes")
        start_id, end_id = str(trip["start_station_id"]), str(trip["end_station_id"])
        counts[(start_id, _bucket(started, policy.interval_minutes))][0] += 1
        counts[(end_id, _bucket(ended, policy.interval_minutes))][1] += 1
        names[start_id] = str(trip.get("start_station_name") or start_id)
        names[end_id] = str(trip.get("end_station_name") or end_id)
    times = [time for _, time in counts]
    first, last = min(times), max(times)
    step = timedelta(minutes=policy.interval_minutes)
    rows: list[dict[str, object]] = []
    for station_id in sorted(names):
        current = first + timedelta(hours=1)
        while current + timedelta(minutes=max(policy.horizons_minutes)) <= last + step:
            minute, weekday = current.hour * 60 + current.minute, current.weekday()
            row: dict[str, object] = {
                "station_id": station_id,
                "station_name": names[station_id],
                "feature_time": current,
                "minute_sin": math.sin(2 * math.pi * minute / 1440),
                "minute_cos": math.cos(2 * math.pi * minute / 1440),
                "weekday_sin": math.sin(2 * math.pi * weekday / 7),
                "weekday_cos": math.cos(2 * math.pi * weekday / 7),
                "departures_lag_15m": counts[(station_id, current - step)][0],
                "arrivals_lag_15m": counts[(station_id, current - step)][1],
                "departures_lag_60m": counts[(station_id, current - timedelta(hours=1))][0],
                "arrivals_lag_60m": counts[(station_id, current - timedelta(hours=1))][1],
            }
            for horizon in policy.horizons_minutes:
                interval_count = horizon // policy.interval_minutes
                departures = sum(
                    counts[(station_id, current + offset * step)][0]
                    for offset in range(interval_count)
                )
                arrivals = sum(
                    counts[(station_id, current + offset * step)][1]
                    for offset in range(interval_count)
                )
                row[f"target_time_{horizon}m"] = current + timedelta(minutes=horizon)
                row[f"target_departures_{horizon}m"] = departures
                row[f"target_arrivals_{horizon}m"] = arrivals
                row[f"target_net_flow_{horizon}m"] = arrivals - departures
            rows.append(row)
            current += step
    if not rows:
        raise ValueError("historical ridership does not span enough time for demand targets")
    result = pl.DataFrame(rows).sort(["feature_time", "station_id"])
    unique_times = sorted(result["feature_time"].unique().to_list())
    train_end = max(1, int(len(unique_times) * policy.train_fraction))
    validation_end = max(
        train_end + 1, int(len(unique_times) * (policy.train_fraction + policy.validation_fraction))
    )
    if validation_end >= len(unique_times):
        raise ValueError("at least three distinct feature timestamps are required")
    result = result.with_columns(
        pl.when(pl.col("feature_time") < unique_times[train_end])
        .then(pl.lit("train"))
        .when(pl.col("feature_time") < unique_times[validation_end])
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.parquet"
    result.write_parquet(dataset_path, compression="zstd")
    manifest = {
        "schema_version": 2,
        "forecast_kind": "station_demand",
        "config": asdict(policy),
        "feature_columns": list(FEATURE_COLUMNS),
        "targets": ["departures", "arrivals", "net_flow"],
        "row_count": result.height,
        "split_counts": {
            row["split"]: row["len"] for row in result.group_by("split").len().to_dicts()
        },
        "source_files": [{"path": str(path), "sha256": sha256_file(path)} for path in source_files],
        "dataset_sha256": sha256_file(dataset_path),
        "point_in_time_rule": "features use only trip events before feature_time",
        "target_rule": "counts trips in [feature_time, feature_time + horizon)",
        "split_rule": "global feature timestamps are assigned chronologically 60%/20%/20%",
    }
    write_json(output_dir / "dataset-manifest.json", manifest)
    return dataset_path
