from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bikeshare_forecast.contracts import GbfsStationInformation, GbfsStationStatus
from bikeshare_forecast.validation import (
    SnapshotValidationError,
    ValidationPolicy,
    normalize_station_snapshots,
)

FIXTURES = Path(__file__).parents[2] / "data" / "fixtures" / "gbfs"


def fixtures() -> tuple[GbfsStationInformation, GbfsStationStatus]:
    information = GbfsStationInformation.model_validate_json(
        (FIXTURES / "station_information.json").read_text()
    )
    status = GbfsStationStatus.model_validate_json((FIXTURES / "station_status.json").read_text())
    return information, status


def normalize(
    information: GbfsStationInformation,
    status: GbfsStationStatus,
    *,
    ingested_at: datetime = datetime(2026, 9, 16, 12, 0, 10, tzinfo=UTC),
    policy: ValidationPolicy | None = None,
):
    return normalize_station_snapshots(
        information,
        status,
        ingested_at=ingested_at,
        source_system_id="bike_share_toronto",
        raw_content_hash="a" * 64,
        policy=policy,
    )


def test_normalizes_fixture_and_keeps_event_and_ingestion_time_distinct() -> None:
    information, status = fixtures()

    report = normalize(information, status)

    assert report.is_valid
    assert report.coverage == 1.0
    assert len(report.snapshots) == 1
    snapshot = report.snapshots[0]
    assert snapshot.source_last_reported_at == datetime(2026, 9, 16, 11, 59, 45, tzinfo=UTC)
    assert snapshot.ingested_at == datetime(2026, 9, 16, 12, 0, 10, tzinfo=UTC)
    assert snapshot.station_name == "Synthetic Station"


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"num_docks_available": 10}, "capacity_mismatch"),
        (
            {"last_reported": datetime(2026, 9, 16, 12, 2, tzinfo=UTC)},
            "future_source_timestamp",
        ),
    ],
)
def test_retains_station_issue_without_filling_invalid_time(
    change: dict[str, object], code: str
) -> None:
    information, status = fixtures()
    changed_status = status.model_copy(
        update={
            "data": status.data.model_copy(
                update={"stations": [status.data.stations[0].model_copy(update=change)]}
            )
        }
    )

    report = normalize(information, changed_status)

    assert code in {issue.code for issue in report.issues}
    assert all(issue.severity == "warning" for issue in report.issues)
    if code == "future_source_timestamp":
        assert report.snapshots == ()
    else:
        assert len(report.snapshots) == 1
    report.raise_for_errors()


def test_disabled_inventory_is_included_in_capacity_accounting() -> None:
    information, status = fixtures()
    changed = status.data.stations[0].model_copy(
        update={
            "num_vehicles_available": 3,
            "num_vehicles_disabled": 1,
            "num_docks_available": 10,
            "num_docks_disabled": 1,
        }
    )
    updated = status.model_copy(
        update={"data": status.data.model_copy(update={"stations": [changed]})}
    )

    report = normalize(information, updated)

    assert report.is_valid
    assert report.issues == ()
    assert len(report.snapshots) == 1


def test_reports_staleness_and_missing_station_coverage() -> None:
    information, status = fixtures()
    stale = normalize(
        information,
        status,
        ingested_at=datetime(2026, 9, 16, 12, 20, tzinfo=UTC),
        policy=ValidationPolicy(freshness_threshold=timedelta(minutes=10)),
    )
    missing = status.model_copy(update={"data": status.data.model_copy(update={"stations": []})})
    uncovered = normalize(information, missing)

    assert {issue.code for issue in stale.issues} == {"stale_snapshot"}
    assert stale.snapshots == ()
    assert stale.is_valid
    assert uncovered.coverage == 0.0
    assert {issue.code for issue in uncovered.issues} == {
        "missing_status",
        "coverage_below_threshold",
    }
    with pytest.raises(SnapshotValidationError):
        uncovered.raise_for_errors()


def test_rejects_duplicate_source_records() -> None:
    information, status = fixtures()
    duplicated = status.model_copy(
        update={
            "data": status.data.model_copy(
                update={"stations": [status.data.stations[0], status.data.stations[0]]}
            )
        }
    )

    report = normalize(information, duplicated)

    assert report.snapshots == ()
    assert {issue.code for issue in report.issues} == {
        "duplicate_status",
        "coverage_below_threshold",
    }
