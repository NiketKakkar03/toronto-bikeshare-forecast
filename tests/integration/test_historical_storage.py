from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from bikeshare_forecast.cli import app
from bikeshare_forecast.ingestion import (
    EcccHourlyV1Adapter,
    SourceMetadata,
    TorontoRidershipV1Adapter,
)
from bikeshare_forecast.storage import HistoricalStore

FIXTURES = Path(__file__).parents[2] / "data" / "fixtures" / "historical"


def metadata() -> SourceMetadata:
    return SourceMetadata(source_name="fixture", retrieved_at=datetime(2026, 9, 17, tzinfo=UTC))


def test_historical_imports_are_durable_queryable_and_idempotent(tmp_path: Path) -> None:
    store = HistoricalStore(tmp_path)
    trips = TorontoRidershipV1Adapter().from_path(
        FIXTURES / "ridership_v1.csv", metadata=metadata()
    )
    weather = EcccHourlyV1Adapter().from_path(
        FIXTURES / "weather_hourly_v1.csv", metadata=metadata()
    )

    trip_result = store.write_trips(trips)
    weather_result = store.write_weather(weather)
    duplicate = store.write_trips(trips)

    assert trip_result.rows_written == 2
    assert weather_result.rows_written == 2
    assert duplicate.rows_written == 0
    assert duplicate.duplicates_ignored == 2
    assert store.quality_summaries("ridership") == [trips.summary]
    assert trip_result.parquet_path is not None
    with duckdb.connect() as connection:
        schema = dict(
            connection.execute(
                "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM read_parquet(?))",
                [str(trip_result.parquet_path)],
            ).fetchall()
        )
    assert schema["started_at"] == "TIMESTAMP WITH TIME ZONE"
    assert schema["source_content_hash"] == "VARCHAR"


def test_store_refuses_partial_invalid_import(tmp_path: Path) -> None:
    imported = TorontoRidershipV1Adapter().from_rows(
        [{"trip_id": "broken"}],
        metadata=metadata(),
        source_file="payload",
        source_content_hash="d" * 64,
    )

    with pytest.raises(ValueError, match="rejected 1 row"):
        HistoricalStore(tmp_path).write_trips(imported)
    assert list(tmp_path.iterdir()) == []


def test_historical_import_and_summary_cli(tmp_path: Path) -> None:
    output = tmp_path / "history"
    imported = CliRunner().invoke(
        app,
        [
            "import-ridership",
            str(FIXTURES / "ridership_v1.csv"),
            "--source-name",
            "fixture",
            "--retrieved-at",
            "2026-09-17T12:00:00Z",
            "--output-dir",
            str(output),
        ],
    )
    summary = CliRunner().invoke(
        app, ["historical-summary", "ridership", "--data-dir", str(output)]
    )

    assert imported.exit_code == 0
    assert "rows_written: 2" in imported.stdout
    assert summary.exit_code == 0
    assert "accepted_rows: 2" in summary.stdout
