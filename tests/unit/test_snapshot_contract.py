from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from bikeshare_forecast.contracts import StationSnapshot


def valid_snapshot() -> dict[str, object]:
    return {
        "station_id": "7001",
        "station_name": "Synthetic Station",
        "latitude": 43.6532,
        "longitude": -79.3832,
        "capacity": 15,
        "bikes_available": 4,
        "docks_available": 11,
        "is_installed": True,
        "is_renting": True,
        "is_returning": True,
        "source_last_reported_at": datetime(2026, 9, 16, 12, tzinfo=UTC),
        "ingested_at": datetime(2026, 9, 16, 12, 0, 10, tzinfo=UTC),
        "source_system_id": "bike_share_toronto",
        "source_schema_version": "3.0",
        "raw_content_hash": "a" * 64,
    }


def test_accepts_utc_snapshot() -> None:
    snapshot = StationSnapshot.model_validate(valid_snapshot())

    assert snapshot.station_id == "7001"
    assert snapshot.bikes_available + snapshot.docks_available == snapshot.capacity


@pytest.mark.parametrize(
    "timestamp",
    [datetime(2026, 9, 16, 12), datetime(2026, 9, 16, 8, tzinfo=timezone(-timedelta(hours=4)))],
)
def test_rejects_non_utc_timestamps(timestamp: datetime) -> None:
    values = valid_snapshot()
    values["source_last_reported_at"] = timestamp

    with pytest.raises(ValidationError, match="timestamp must be"):
        StationSnapshot.model_validate(values)


def test_rejects_negative_inventory() -> None:
    values = valid_snapshot()
    values["bikes_available"] = -1

    with pytest.raises(ValidationError):
        StationSnapshot.model_validate(values)
