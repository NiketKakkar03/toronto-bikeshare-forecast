import zipfile
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
    import_toronto_ridership_official,
)
from bikeshare_forecast.storage import HistoricalStore

FIXTURES = Path(__file__).parents[2] / "data" / "fixtures" / "historical"
OFFICIAL_2022_2023_HEADER = ",".join(
    (
        "Trip Id",
        "Trip  Duration",
        "Start Station Id",
        "Start Time",
        "Start Station Name",
        "End Station Id",
        "End Time",
        "End Station Name",
        "Bike Id",
        "User Type",
    )
)


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


def test_official_2023_monthly_import_streams_to_canonical_parquet(tmp_path: Path) -> None:
    source = tmp_path / "Bike share ridership 2023-01.csv"
    source.write_text(
        "\n".join(
            [
                OFFICIAL_2022_2023_HEADER,
                ",".join(
                    (
                        "20354837",
                        "175",
                        "7457",
                        "02/01/2023 00:01",
                        "Queen's Park Cres W / Hoskin Ave",
                        "7190",
                        "02/01/2023 00:03",
                        "St. George St / Hoskin Ave",
                        "538",
                        "Casual Member",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = import_toronto_ridership_official(source, tmp_path / "history", metadata=metadata())

    assert result.rows_written == 1
    assert result.rejected_rows == 0
    assert result.parquet_path.exists()
    with duckdb.connect() as connection:
        row = connection.execute(
            "SELECT trip_id, started_at, start_station_id, bike_type, user_type, "
            "source_schema_version FROM read_parquet(?)",
            [str(result.parquet_path)],
        ).fetchone()
    assert row == (
        "20354837",
        datetime(2023, 2, 1, 5, 1, tzinfo=UTC),
        "7457",
        None,
        "Casual Member",
        "toronto-ridership-2023",
    )


def test_official_2022_nested_zip_import_uses_member_hash(tmp_path: Path) -> None:
    source = tmp_path / "Bike share ridership 2022-11.zip"
    member = "Bike share ridership 2022-11.csv"
    csv_bytes = "\n".join(
        [
            OFFICIAL_2022_2023_HEADER,
            ",".join(
                (
                    "19571966",
                    "523",
                    "7001",
                    "11/01/2022 00:00",
                    "Wellesley Station Green P",
                    "7058",
                    "11/01/2022 00:09",
                    "Huron/ Harbord St",
                    "3921",
                    "Casual Member",
                )
            ),
        ]
    ).encode()
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(member, csv_bytes)

    result = import_toronto_ridership_official(source, tmp_path / "history", metadata=metadata())

    assert result.rows_written == 1
    with duckdb.connect() as connection:
        row = connection.execute(
            "SELECT source_file, source_schema_version FROM read_parquet(?)",
            [str(result.parquet_path)],
        ).fetchone()
    assert row == (
        "Bike share ridership 2022-11.zip!Bike share ridership 2022-11.csv",
        "toronto-ridership-2022",
    )
