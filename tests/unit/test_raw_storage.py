import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from bikeshare_forecast.contracts import GbfsStationStatus
from bikeshare_forecast.storage import RawCaptureStore

FIXTURES = Path(__file__).parents[2] / "data" / "fixtures" / "gbfs"


def status_fixture() -> GbfsStationStatus:
    return GbfsStationStatus.model_validate_json(
        (FIXTURES / "station_status.json").read_text(encoding="utf-8")
    )


def test_raw_capture_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    store = RawCaptureStore(tmp_path)
    payload = status_fixture()
    retrieved_at = datetime(2026, 9, 16, 12, 0, 10, tzinfo=UTC)

    first, first_created = store.capture(
        payload,
        source_url="https://example.test/station_status.json",
        feed_name="station_status",
        retrieved_at=retrieved_at,
        feed_updated_at=payload.last_updated,
        schema_version=payload.version,
        row_count=len(payload.data.stations),
    )
    second, second_created = store.capture(
        payload,
        source_url="https://example.test/station_status.json",
        feed_name="station_status",
        retrieved_at=retrieved_at,
        feed_updated_at=payload.last_updated,
        schema_version=payload.version,
        row_count=len(payload.data.stations),
    )

    raw_path = tmp_path / first.raw_path
    assert first_created is True
    assert second_created is False
    assert second == first
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == first.content_hash
    manifests = list(tmp_path.rglob("*.manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["row_count"] == 1


def test_raw_hash_changes_when_validated_content_changes(tmp_path: Path) -> None:
    payload = status_fixture()
    changed = payload.model_copy(
        update={
            "data": payload.data.model_copy(
                update={
                    "stations": [
                        payload.data.stations[0].model_copy(update={"num_vehicles_available": 3})
                    ]
                }
            )
        }
    )
    store = RawCaptureStore(tmp_path)
    kwargs = {
        "source_url": "https://example.test/status",
        "feed_name": "station_status",
        "retrieved_at": datetime(2026, 9, 16, 12, tzinfo=UTC),
        "feed_updated_at": payload.last_updated,
        "schema_version": payload.version,
        "row_count": 1,
    }

    original, _ = store.capture(payload, **kwargs)
    modified, _ = store.capture(changed, **kwargs)

    assert original.content_hash != modified.content_hash


def test_identical_raw_content_deduplicates_across_retrieval_dates(tmp_path: Path) -> None:
    payload = status_fixture()
    store = RawCaptureStore(tmp_path)
    common = {
        "source_url": "https://example.test/status",
        "feed_name": "station_status",
        "feed_updated_at": payload.last_updated,
        "schema_version": payload.version,
        "row_count": 1,
    }
    first, _ = store.capture(
        payload,
        retrieved_at=datetime(2026, 9, 16, 12, tzinfo=UTC),
        **common,
    )
    duplicate, created = store.capture(
        payload,
        retrieved_at=datetime(2026, 9, 17, 12, tzinfo=UTC),
        **common,
    )

    assert created is False
    assert duplicate == first
    assert len(list(tmp_path.rglob("*.json"))) == 2  # one payload and one manifest
