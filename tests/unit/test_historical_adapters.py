from datetime import UTC, datetime
from pathlib import Path

import pytest

from bikeshare_forecast.ingestion import (
    EcccHourlyV1Adapter,
    EcccTorontoCity2024Adapter,
    SourceMetadata,
    TorontoRidershipV1Adapter,
)

FIXTURES = Path(__file__).parents[2] / "data" / "fixtures" / "historical"
METADATA = SourceMetadata(
    source_name="fixture",
    retrieved_at=datetime(2026, 9, 17, tzinfo=UTC),
    source_url="https://example.test/dataset",
    licence="test-only",
)


def test_ridership_adapter_normalizes_station_ids_and_utc() -> None:
    imported = TorontoRidershipV1Adapter().from_path(
        FIXTURES / "ridership_v1.csv", metadata=METADATA
    )

    assert imported.failures == ()
    assert imported.summary.accepted_rows == 2
    assert imported.summary.distinct_station_count == 2
    assert imported.records[0].start_station_id == "7001"
    assert imported.records[0].started_at == datetime(2024, 1, 15, 13, tzinfo=UTC)
    assert imported.records[0].lineage.source_row_number == 2


def test_weather_adapter_preserves_missing_values_and_utc() -> None:
    imported = EcccHourlyV1Adapter().from_rows(
        [
            {
                "Climate ID": "6158355.0",
                "Date/Time (UTC)": "2024-01-15T13:00:00Z",
                "Temp (°C)": "",
                "Precip. Amount (mm)": "0",
                "Snow on Grnd (cm)": "",
                "Wind Spd (km/h)": "18",
                "Rel Hum (%)": "72",
                "Weather": "",
            }
        ],
        metadata=METADATA,
        source_file="validated-payload",
        source_content_hash="a" * 64,
    )

    assert imported.records[0].climate_station_id == "6158355"
    assert imported.records[0].temperature_c is None
    assert imported.summary.missing_optional_value_count == 3


def test_toronto_city_weather_adapter_interprets_local_standard_time() -> None:
    imported = EcccTorontoCity2024Adapter().from_rows(
        [
            {
                "Climate ID": "6158359",
                "Date/Time (LST)": "2024-01-01 08:00",
                "Temp (°C)": "-1.2",
                "Precip. Amount (mm)": "0.2",
                "Wind Spd (km/h)": "15",
                "Rel Hum (%)": "80",
                "Weather": "Snow",
            }
        ],
        metadata=METADATA,
        source_file="official.csv",
        source_content_hash="d" * 64,
    )
    assert imported.failures == ()
    assert imported.records[0].observed_at == datetime(2024, 1, 1, 13, tzinfo=UTC)
    assert imported.records[0].condition == "Snow"


def test_invalid_rows_are_reported_and_strict_boundary_fails() -> None:
    imported = TorontoRidershipV1Adapter().from_rows(
        [{"trip_id": "broken"}],
        metadata=METADATA,
        source_file="payload",
        source_content_hash="b" * 64,
    )

    assert imported.summary.rejected_rows == 1
    assert imported.failures[0].code == "missing_column"
    assert imported.failures[0].row_number == 2
    with pytest.raises(ValueError, match="rejected 1 row"):
        imported.require_valid()


def test_duplicate_source_rows_are_visible_in_quality_summary() -> None:
    row = {
        "trip_id": "same",
        "trip_start_time": "2024-01-01T00:00:00-05:00",
        "trip_stop_time": "2024-01-01T00:01:00-05:00",
        "trip_duration_seconds": "60",
        "from_station_id": "1",
        "to_station_id": "2",
    }
    imported = TorontoRidershipV1Adapter().from_rows(
        [row, row],
        metadata=METADATA,
        source_file="payload",
        source_content_hash="c" * 64,
    )

    assert len(imported.records) == 1
    assert imported.summary.duplicate_rows == 1
