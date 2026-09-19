from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from bikeshare_forecast.ml import DatasetConfig, build_dataset, train_models


def _history(root: Path) -> None:
    start = datetime(2026, 8, 1, tzinfo=UTC)
    rows = []
    for step in range(40):
        when = start + timedelta(minutes=15 * step)
        rows.append(
            {
                "trip_id": str(step),
                "started_at": when,
                "ended_at": when + timedelta(minutes=8),
                "start_station_id": "7001",
                "end_station_id": "7002",
                "start_station_name": "Station A",
                "end_station_name": "Station B",
            }
        )
    destination = root / "ridership" / "parquet"
    destination.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(destination / "part-test.parquet")


def test_trip_windows_create_departure_arrival_and_net_flow_targets(tmp_path: Path) -> None:
    historical = tmp_path / "historical"
    _history(historical)
    output = tmp_path / "dataset"
    build_dataset(historical, output, DatasetConfig(horizons_minutes=(15,)))
    frame = pl.read_parquet(output / "dataset.parquet")
    station_a = frame.filter(pl.col("station_id") == "7001")
    assert (station_a["target_departures_15m"] == 1).all()
    assert (station_a["target_arrivals_15m"] == 0).all()
    assert (station_a["target_net_flow_15m"] == -1).all()


def test_dataset_hash_tampering_blocks_training(tmp_path: Path) -> None:
    historical = tmp_path / "historical"
    _history(historical)
    dataset = tmp_path / "dataset"
    build_dataset(historical, dataset, DatasetConfig(horizons_minutes=(15,)))
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
