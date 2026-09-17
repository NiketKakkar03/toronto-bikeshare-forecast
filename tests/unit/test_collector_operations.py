import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from bikeshare_forecast.config import (
    CollectionConfig,
    GbfsCollectionConfig,
    StorageConfig,
    ValidationConfig,
)
from bikeshare_forecast.contracts import GbfsStationInformation
from bikeshare_forecast.ingestion import fetch_station_feeds
from bikeshare_forecast.operations import (
    CollectionReport,
    CollectionReportStore,
    StationCollector,
    run_scheduled,
)
from bikeshare_forecast.storage import MetadataStore

FIXTURES = Path(__file__).parents[2] / "data" / "fixtures" / "gbfs"
NOW = datetime(2026, 9, 16, 12, 0, 10, tzinfo=UTC)


def fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def config(tmp_path: Path) -> CollectionConfig:
    return CollectionConfig(
        gbfs=GbfsCollectionConfig(
            discovery_url="https://example.test/root.json",
            source_system_id="test_system",
            poll_interval_seconds=300,
            freshness_threshold_seconds=600,
            request_timeout_seconds=1,
            max_attempts=3,
            retry_backoff_seconds=0.5,
        ),
        validation=ValidationConfig(
            toronto_latitude_min=43.4,
            toronto_latitude_max=44.0,
            toronto_longitude_min=-79.8,
            toronto_longitude_max=-79.0,
            future_timestamp_tolerance_seconds=60,
        ),
        storage=StorageConfig(
            raw_dir=tmp_path / "raw",
            silver_dir=tmp_path / "silver",
            reports_dir=tmp_path / "reports",
            metadata_dir=tmp_path / "metadata",
        ),
    )


def transport(*, invalid_status: bool = False) -> httpx.MockTransport:
    discovery = fixture("gbfs.json")
    assert isinstance(discovery, dict)
    discovery["data"]["feeds"][0]["url"] = "https://example.test/information.json"
    discovery["data"]["feeds"][1]["url"] = "https://example.test/status.json"
    status = fixture("station_status.json")
    assert isinstance(status, dict)
    if invalid_status:
        status["data"]["stations"][0]["num_docks_available"] = 10

    def respond(request: httpx.Request) -> httpx.Response:
        payloads = {
            "/root.json": discovery,
            "/information.json": fixture("station_information.json"),
            "/status.json": status,
        }
        return httpx.Response(200, json=payloads[request.url.path])

    return httpx.MockTransport(respond)


def test_fetches_advertised_feeds_with_bounded_backoff() -> None:
    attempts = 0
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if request.url.path == "/root.json" and attempts < 3:
            return httpx.Response(503, request=request)
        payloads = {
            "/root.json": fixture("gbfs.json"),
            "/gbfs/v3.0/station_information.json": fixture("station_information.json"),
            "/gbfs/v3.0/station_status.json": fixture("station_status.json"),
        }
        return httpx.Response(200, json=payloads[request.url.path], request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        feeds = fetch_station_feeds(
            "https://example.test/root.json",
            client=client,
            max_attempts=3,
            backoff_seconds=0.5,
            sleep=delays.append,
        )

    assert attempts == 5  # three discovery attempts, then two required feeds
    assert delays == [0.5, 1.0]
    assert feeds.status.data.stations[0].station_id == "7001"


def test_collector_persists_raw_metadata_report_and_silver(tmp_path: Path) -> None:
    with httpx.Client(transport=transport()) as client:
        collector = StationCollector(
            config(tmp_path), client=client, sleep=lambda _: None, clock=lambda: NOW
        )
        first = collector.collect_once()
        second = collector.collect_once()

    assert first.outcome == "success"
    assert first.rows_written == 1
    assert first.metadata_versions_written == 1
    assert second.rows_written == 0
    assert second.duplicates_ignored == 1
    assert second.metadata_versions_written == 0
    assert len(list((tmp_path / "raw").rglob("*.manifest.json"))) == 2
    assert len(list((tmp_path / "silver").glob("*.parquet"))) == 1
    assert len(CollectionReportStore(tmp_path / "reports").reports()) == 2


def test_collector_retains_validation_failure_without_silver_write(tmp_path: Path) -> None:
    with httpx.Client(transport=transport(invalid_status=True)) as client:
        report = StationCollector(
            config(tmp_path), client=client, sleep=lambda _: None, clock=lambda: NOW
        ).collect_once()

    assert report.outcome == "validation_failed"
    assert {issue.code for issue in report.issues} == {"capacity_mismatch"}
    assert not (tmp_path / "silver").exists()
    assert CollectionReportStore(tmp_path / "reports").reports() == (report,)


def test_metadata_changes_receive_incrementing_versions(tmp_path: Path) -> None:
    information = GbfsStationInformation.model_validate(fixture("station_information.json"))
    store = MetadataStore(tmp_path)

    initial = store.record(information, observed_at=NOW)
    unchanged = store.record(information, observed_at=NOW + timedelta(minutes=5))
    changed = information.model_copy(
        update={
            "data": information.data.model_copy(
                update={
                    "stations": [information.data.stations[0].model_copy(update={"capacity": 16})]
                }
            )
        }
    )
    updated = store.record(changed, observed_at=NOW + timedelta(minutes=10))

    assert initial.changes[0].version == 1
    assert unchanged.changes == ()
    assert updated.changes[0].kind == "changed"
    assert updated.changes[0].version == 2


def report(at: datetime, outcome: str = "success") -> CollectionReport:
    return CollectionReport.model_validate({"collected_at": at, "outcome": outcome})


def test_interval_coverage_and_bounded_scheduler(tmp_path: Path) -> None:
    store = CollectionReportStore(tmp_path)
    store.write(report(NOW))
    store.write(report(NOW + timedelta(minutes=10), "collection_failed"))

    coverage = store.coverage(
        start=NOW,
        end=NOW + timedelta(minutes=15),
        interval_seconds=300,
    )
    sleeps: list[float] = []
    scheduled = run_scheduled(
        lambda: report(NOW), interval_seconds=300, max_runs=2, sleep=sleeps.append
    )

    assert coverage.expected_intervals == 3
    assert coverage.covered_intervals == 1
    assert coverage.missed_intervals == 2
    assert coverage.failed_runs == 1
    assert len(scheduled) == 2
    assert sleeps == [300]
