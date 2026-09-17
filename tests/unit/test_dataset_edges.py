from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from bikeshare_forecast.contracts import StationSnapshot
from bikeshare_forecast.ml import DatasetConfig, build_dataset, train_models
from bikeshare_forecast.storage import SilverStore


def _snapshot(minute: int, bikes: int) -> StationSnapshot:
    event_time = datetime(2026, 9, 1, tzinfo=UTC) + timedelta(minutes=minute)
    return StationSnapshot(
        station_id="7001",
        station_name="Station 7001",
        latitude=43.65,
        longitude=-79.38,
        capacity=10,
        bikes_available=bikes,
        docks_available=10 - bikes,
        is_installed=True,
        is_renting=True,
        is_returning=True,
        source_last_reported_at=event_time,
        ingested_at=event_time,
        source_system_id="bike_share_toronto",
        source_schema_version="3.0",
        raw_content_hash=f"{minute:064x}",
    )


def test_target_tie_uses_earlier_observation_and_lags_never_use_future(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    SilverStore(silver).write_snapshots(
        [_snapshot(minute, index) for index, minute in enumerate((0, 12, 18, 30, 45, 60, 75))]
    )
    output = tmp_path / "dataset"
    build_dataset(
        silver,
        output,
        DatasetConfig(horizons_minutes=(15,), target_tolerance_minutes=3),
    )

    row = pl.read_parquet(output / "dataset.parquet").sort("feature_time").row(0, named=True)
    assert row["feature_time"].minute == 0
    assert row["target_time_15m"].minute == 12
    assert row["target_bikes_15m"] == 1
    assert row["bikes_lag_15m"] is None
    train_models(output, tmp_path / "models")
    assert (tmp_path / "models" / "logistic-empty-15m.json").exists()


def test_missing_target_inside_no_tolerance_is_rejected(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    SilverStore(silver).write_snapshots([_snapshot(0, 1), _snapshot(19, 2), _snapshot(38, 3)])

    with pytest.raises(ValueError, match="no rows have complete horizon targets"):
        build_dataset(
            silver,
            tmp_path / "dataset",
            DatasetConfig(horizons_minutes=(15,), target_tolerance_minutes=3),
        )


def test_late_arriving_snapshot_is_not_used_as_a_historical_feature(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    values = [_snapshot(minute, index) for index, minute in enumerate(range(0, 105, 15))]
    late = values[1].model_copy(
        update={"ingested_at": values[1].source_last_reported_at + timedelta(minutes=25)}
    )
    SilverStore(silver).write_snapshots([values[0], late, *values[2:]])
    output = tmp_path / "dataset"
    build_dataset(silver, output, DatasetConfig(horizons_minutes=(15,)))

    feature_time = datetime(2026, 9, 1, 0, 30, tzinfo=UTC)
    row = (
        pl.read_parquet(output / "dataset.parquet")
        .filter(pl.col("feature_time") == feature_time)
        .row(0, named=True)
    )
    assert row["bikes_lag_15m"] is None


def test_dataset_hash_tampering_blocks_training(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    SilverStore(silver).write_snapshots(
        [_snapshot(minute, index % 11) for index, minute in enumerate(range(0, 180, 15))]
    )
    dataset = tmp_path / "dataset"
    build_dataset(silver, dataset, DatasetConfig(horizons_minutes=(15,)))
    with (dataset / "dataset.parquet").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="hash does not match"):
        train_models(dataset, tmp_path / "models")


@pytest.mark.parametrize(
    "config",
    [
        DatasetConfig(train_fraction=0),
        DatasetConfig(validation_fraction=0),
        DatasetConfig(train_fraction=0.8, validation_fraction=0.2),
    ],
)
def test_invalid_split_geometry_is_rejected(tmp_path: Path, config: DatasetConfig) -> None:
    with pytest.raises(ValueError, match=r"split fractions|leave a test"):
        build_dataset(tmp_path, tmp_path / "dataset", config)
