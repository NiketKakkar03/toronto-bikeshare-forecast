from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from bikeshare_forecast.cli import app
from bikeshare_forecast.contracts import StationSnapshot
from bikeshare_forecast.storage import DuckDBCatalogue, SilverStore


def snapshot(*, station_id: str = "7001", minute: int = 0) -> StationSnapshot:
    event_time = datetime(2026, 9, 16, 12, minute, tzinfo=UTC)
    return StationSnapshot(
        station_id=station_id,
        station_name=f"Station {station_id}",
        latitude=43.6532,
        longitude=-79.3832,
        capacity=15,
        bikes_available=4,
        docks_available=11,
        is_installed=True,
        is_renting=True,
        is_returning=True,
        source_last_reported_at=event_time,
        ingested_at=event_time + timedelta(seconds=10),
        source_system_id="bike_share_toronto",
        source_schema_version="3.0",
        raw_content_hash="a" * 64,
    )


def test_parquet_write_is_idempotent_and_queryable(tmp_path: Path) -> None:
    parquet_root = tmp_path / "silver"
    store = SilverStore(parquet_root)
    values = [snapshot(), snapshot(station_id="7002")]

    first = store.write_snapshots(values)
    duplicate = store.write_snapshots(values)
    catalogue = DuckDBCatalogue(tmp_path / "catalogue.duckdb", parquet_root)
    summary = catalogue.summary()

    assert first.rows_written == 2
    assert first.path is not None and first.path.exists()
    assert duplicate.rows_written == 0
    assert duplicate.duplicates_ignored == 2
    assert len(list(parquet_root.glob("*.parquet"))) == 1
    assert summary.row_count == 2
    assert summary.station_count == 2
    assert summary.duplicate_key_count == 0
    assert catalogue.station_rows("7001")[0][2:] == (4, 11)


def test_conflicting_duplicate_is_rejected(tmp_path: Path) -> None:
    store = SilverStore(tmp_path)
    original = snapshot()
    conflicting = original.model_copy(update={"bikes_available": 3})

    with pytest.raises(ValueError, match="conflicting duplicate"):
        store.write_snapshots([original, conflicting])


def test_recollection_of_same_source_state_keeps_first_lineage(tmp_path: Path) -> None:
    store = SilverStore(tmp_path)
    original = snapshot()
    recollected = original.model_copy(
        update={
            "ingested_at": original.ingested_at + timedelta(minutes=5),
            "raw_content_hash": "b" * 64,
        }
    )

    first = store.write_snapshots([original])
    duplicate = store.write_snapshots([recollected])

    assert first.rows_written == 1
    assert duplicate.rows_written == 0
    assert duplicate.duplicates_ignored == 1


def test_parquet_uses_utc_timestamps_and_int32_counts(tmp_path: Path) -> None:
    result = SilverStore(tmp_path).write_snapshots([snapshot()])
    assert result.path is not None

    with duckdb.connect() as connection:
        schema = dict(
            connection.execute(
                "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM read_parquet(?))",
                [str(result.path)],
            ).fetchall()
        )

    assert schema["capacity"] == "INTEGER"
    assert schema["source_last_reported_at"] == "TIMESTAMP WITH TIME ZONE"
    assert schema["ingested_at"] == "TIMESTAMP WITH TIME ZONE"


def test_data_summary_cli(tmp_path: Path) -> None:
    parquet_root = tmp_path / "silver"
    SilverStore(parquet_root).write_snapshots([snapshot()])

    result = CliRunner().invoke(
        app,
        [
            "data-summary",
            "--silver-dir",
            str(parquet_root),
            "--catalogue",
            str(tmp_path / "catalogue.duckdb"),
        ],
    )

    assert result.exit_code == 0
    assert "rows: 1" in result.stdout
    assert "stations: 1" in result.stdout
    assert "duplicate_keys: 0" in result.stdout
