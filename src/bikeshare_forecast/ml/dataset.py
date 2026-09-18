"""Point-in-time dataset construction from normalized station values only."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from bikeshare_forecast.ml.common import sha256_file, write_json

HORIZONS_MINUTES = (15, 30, 60)
FEATURE_COLUMNS = (
    "bikes_available",
    "docks_available",
    "capacity",
    "bike_fraction",
    "minute_sin",
    "minute_cos",
    "weekday_sin",
    "weekday_cos",
    "bikes_lag_15m",
    "bikes_lag_60m",
)


@dataclass(frozen=True)
class DatasetConfig:
    """Label matching and chronological split policy."""

    horizons_minutes: tuple[int, ...] = HORIZONS_MINUTES
    target_tolerance_minutes: int = 5
    unavailable_bikes_threshold: int = 0
    train_fraction: float = 0.6
    validation_fraction: float = 0.2


def _nearest_index(times: list[datetime], target: datetime, tolerance: timedelta) -> int | None:
    insertion = bisect_left(times, target)
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(times)]
    if not candidates:
        return None
    # Earlier timestamp wins an exact tie, making matching stable and documented.
    match = min(candidates, key=lambda index: (abs(times[index] - target), times[index]))
    return match if abs(times[match] - target) <= tolerance else None


def _snapshot_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise ValueError(f"no normalized snapshot parquet files found in {root}")
    return files


def build_dataset(silver_dir: Path, output_dir: Path, config: DatasetConfig | None = None) -> Path:
    """Build features using values available at or before each feature timestamp."""
    policy = config or DatasetConfig()
    if not 0 < policy.train_fraction < 1 or not 0 < policy.validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if policy.train_fraction + policy.validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test partition")

    source_files = _snapshot_files(silver_dir)
    frame = pl.concat([pl.read_parquet(path) for path in source_files]).sort(
        ["station_id", "source_last_reported_at"]
    )
    rows: list[dict[str, object]] = []
    tolerance = timedelta(minutes=policy.target_tolerance_minutes)
    for station in frame.partition_by("station_id", maintain_order=True):
        values = station.to_dicts()
        times = [value["source_last_reported_at"] for value in values]
        assert all(isinstance(value, datetime) for value in times)
        for index, current in enumerate(values):
            now = times[index]
            lag_15 = _nearest_index(times[: index + 1], now - timedelta(minutes=15), tolerance)
            lag_60 = _nearest_index(times[: index + 1], now - timedelta(minutes=60), tolerance)
            minute = now.hour * 60 + now.minute
            weekday = now.weekday()
            import math

            features: dict[str, object] = {
                "station_id": current["station_id"],
                "feature_time": now,
                "bikes_available": current["bikes_available"],
                "docks_available": current["docks_available"],
                "capacity": current["capacity"],
                "bike_fraction": (
                    float(current["bikes_available"]) / float(current["capacity"])
                    if current["capacity"]
                    else 0.0
                ),
                "minute_sin": math.sin(2 * math.pi * minute / 1440),
                "minute_cos": math.cos(2 * math.pi * minute / 1440),
                "weekday_sin": math.sin(2 * math.pi * weekday / 7),
                "weekday_cos": math.cos(2 * math.pi * weekday / 7),
                "bikes_lag_15m": (
                    float(values[lag_15]["bikes_available"]) if lag_15 is not None else None
                ),
                "bikes_lag_60m": (
                    float(values[lag_60]["bikes_available"]) if lag_60 is not None else None
                ),
            }
            complete = True
            for horizon in policy.horizons_minutes:
                target_index = _nearest_index(times, now + timedelta(minutes=horizon), tolerance)
                if target_index is None or times[target_index] <= now:
                    complete = False
                    break
                target_bikes = int(values[target_index]["bikes_available"])
                target_docks = int(values[target_index]["docks_available"])
                features[f"target_time_{horizon}m"] = times[target_index]
                features[f"target_bikes_{horizon}m"] = target_bikes
                features[f"target_docks_{horizon}m"] = target_docks
                features[f"target_empty_{horizon}m"] = int(
                    target_bikes <= policy.unavailable_bikes_threshold
                )
                features[f"target_full_{horizon}m"] = int(target_docks == 0)
                features[f"target_unavailable_{horizon}m"] = int(
                    target_bikes <= policy.unavailable_bikes_threshold
                )
            if complete:
                rows.append(features)
    if not rows:
        raise ValueError("no rows have complete horizon targets under the configured tolerance")

    result = pl.DataFrame(rows).sort(["feature_time", "station_id"])
    unique_times = sorted(result["feature_time"].unique().to_list())
    train_end = max(1, int(len(unique_times) * policy.train_fraction))
    validation_end = max(
        train_end + 1, int(len(unique_times) * (policy.train_fraction + policy.validation_fraction))
    )
    if validation_end >= len(unique_times):
        raise ValueError(
            "at least three distinct feature timestamps are required for temporal splits"
        )
    train_cutoff = unique_times[train_end]
    validation_cutoff = unique_times[validation_end]
    result = result.with_columns(
        pl.when(pl.col("feature_time") < train_cutoff)
        .then(pl.lit("train"))
        .when(pl.col("feature_time") < validation_cutoff)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.parquet"
    result.write_parquet(dataset_path, compression="zstd")
    manifest = {
        "schema_version": 1,
        "config": asdict(policy),
        "feature_columns": list(FEATURE_COLUMNS),
        "row_count": result.height,
        "split_counts": {
            row["split"]: row["len"] for row in result.group_by("split").len().to_dicts()
        },
        "source_files": [{"path": str(path), "sha256": sha256_file(path)} for path in source_files],
        "dataset_sha256": sha256_file(dataset_path),
        "point_in_time_rule": "features use only the current or earlier source_last_reported_at",
        "target_rule": (
            "nearest observation to feature_time + horizon within inclusive tolerance; "
            "earlier observation wins ties; target time must be after feature time"
        ),
        "split_rule": "global feature timestamps are assigned chronologically 60%/20%/20%",
    }
    write_json(output_dir / "dataset-manifest.json", manifest)
    return dataset_path
