"""Normalize matched GBFS records without manufacturing missing observations."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bikeshare_forecast.contracts import (
    GbfsStationInformation,
    GbfsStationStatus,
    StationSnapshot,
)
from bikeshare_forecast.contracts.gbfs import LocalizedText


class ValidationPolicy(BaseModel):
    """Explicit tolerances for source-quality checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    freshness_threshold: timedelta = timedelta(minutes=10)
    future_timestamp_tolerance: timedelta = timedelta(minutes=1)
    capacity_tolerance: int = Field(default=0, ge=0)
    minimum_station_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    latitude_min: float = Field(default=43.4, ge=-90, le=90)
    latitude_max: float = Field(default=44.0, ge=-90, le=90)
    longitude_min: float = Field(default=-79.8, ge=-180, le=180)
    longitude_max: float = Field(default=-79.0, ge=-180, le=180)


class ValidationIssue(BaseModel):
    """A retained warning or error discovered while normalizing source data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    station_id: str | None = None
    severity: Literal["warning", "error"] = "error"


class ValidationReport(BaseModel):
    """Successful snapshots and every detected source-quality failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshots: tuple[StationSnapshot, ...]
    issues: tuple[ValidationIssue, ...]
    expected_station_count: int
    observed_station_count: int
    coverage: float

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def raise_for_errors(self) -> None:
        if not self.is_valid:
            raise SnapshotValidationError(self)


class SnapshotValidationError(ValueError):
    """Raised after validation has collected all actionable failures."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("; ".join(issue.message for issue in report.issues))


def normalize_station_snapshots(
    information: GbfsStationInformation,
    status: GbfsStationStatus,
    *,
    ingested_at: datetime,
    source_system_id: str,
    raw_content_hash: str,
    policy: ValidationPolicy | None = None,
) -> ValidationReport:
    """Join metadata to status, validate it, and retain failures in a report."""
    selected_policy = policy or ValidationPolicy()
    _require_utc(ingested_at, "ingested_at")
    metadata_ids = [station.station_id for station in information.data.stations]
    status_ids = [station.station_id for station in status.data.stations]
    duplicate_metadata = {key for key, count in Counter(metadata_ids).items() if count > 1}
    duplicate_status = {key for key, count in Counter(status_ids).items() if count > 1}
    metadata = {station.station_id: station for station in information.data.stations}
    statuses = {station.station_id: station for station in status.data.stations}
    expected = len(metadata)
    observed_ids = (metadata.keys() & statuses.keys()) - duplicate_metadata - duplicate_status
    coverage = len(observed_ids) / expected if expected else 1.0
    issues: list[ValidationIssue] = []
    snapshots: list[StationSnapshot] = []

    for station_id in sorted(duplicate_metadata):
        issues.append(
            ValidationIssue(
                code="duplicate_metadata",
                station_id=station_id,
                message=f"station {station_id} occurs more than once in station_information",
            )
        )
    for station_id in sorted(duplicate_status):
        issues.append(
            ValidationIssue(
                code="duplicate_status",
                station_id=station_id,
                message=f"station {station_id} occurs more than once in station_status",
            )
        )

    for station_id in sorted(metadata.keys() - statuses.keys()):
        issues.append(
            ValidationIssue(
                code="missing_status",
                station_id=station_id,
                message=f"station {station_id} is missing from station_status",
            )
        )
    for station_id in sorted(statuses.keys() - metadata.keys()):
        issues.append(
            ValidationIssue(
                code="missing_metadata",
                station_id=station_id,
                message=f"station {station_id} is missing from station_information",
            )
        )
    if coverage < selected_policy.minimum_station_coverage:
        issues.append(
            ValidationIssue(
                code="coverage_below_threshold",
                message=(
                    f"station coverage {coverage:.3f} is below "
                    f"{selected_policy.minimum_station_coverage:.3f}"
                ),
            )
        )

    for station_id in sorted(observed_ids):
        info = metadata[station_id]
        state = statuses[station_id]
        station_issues = _station_issues(
            station_id,
            capacity=info.capacity,
            bikes=state.num_vehicles_available,
            bikes_disabled=state.num_vehicles_disabled,
            docks=state.num_docks_available,
            docks_disabled=state.num_docks_disabled,
            source_time=state.last_reported,
            ingested_at=ingested_at,
            policy=selected_policy,
            latitude=info.lat,
            longitude=info.lon,
        )
        issues.extend(station_issues)
        if any(issue.code != "capacity_mismatch" for issue in station_issues):
            continue
        snapshots.append(
            StationSnapshot(
                station_id=station_id,
                station_name=_station_name(info.name),
                latitude=info.lat,
                longitude=info.lon,
                capacity=info.capacity,
                bikes_available=state.num_vehicles_available,
                docks_available=state.num_docks_available,
                is_installed=state.is_installed,
                is_renting=state.is_renting,
                is_returning=state.is_returning,
                source_last_reported_at=state.last_reported.astimezone(UTC),
                ingested_at=ingested_at,
                source_system_id=source_system_id,
                source_schema_version=status.version,
                raw_content_hash=raw_content_hash,
            )
        )

    return ValidationReport(
        snapshots=tuple(snapshots),
        issues=tuple(issues),
        expected_station_count=expected,
        observed_station_count=len(observed_ids),
        coverage=coverage,
    )


def _station_issues(
    station_id: str,
    *,
    capacity: int,
    bikes: int,
    bikes_disabled: int,
    docks: int,
    docks_disabled: int,
    source_time: datetime,
    ingested_at: datetime,
    policy: ValidationPolicy,
    latitude: float,
    longitude: float,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _require_utc(source_time, "source last_reported")
    if not policy.latitude_min <= latitude <= policy.latitude_max:
        issues.append(
            ValidationIssue(
                code="latitude_out_of_bounds",
                station_id=station_id,
                severity="warning",
                message=f"station {station_id} latitude is outside configured bounds",
            )
        )
    if not policy.longitude_min <= longitude <= policy.longitude_max:
        issues.append(
            ValidationIssue(
                code="longitude_out_of_bounds",
                station_id=station_id,
                severity="warning",
                message=f"station {station_id} longitude is outside configured bounds",
            )
        )
    accounted_inventory = bikes + bikes_disabled + docks + docks_disabled
    difference = abs(capacity - accounted_inventory)
    if difference > policy.capacity_tolerance:
        issues.append(
            ValidationIssue(
                code="capacity_mismatch",
                station_id=station_id,
                severity="warning",
                message=(
                    f"station {station_id} available and disabled inventory differs from "
                    f"capacity by {difference} (tolerance {policy.capacity_tolerance})"
                ),
            )
        )
    if source_time > ingested_at + policy.future_timestamp_tolerance:
        issues.append(
            ValidationIssue(
                code="future_source_timestamp",
                station_id=station_id,
                severity="warning",
                message=f"station {station_id} source timestamp is implausibly in the future",
            )
        )
    age = ingested_at - source_time
    if age > policy.freshness_threshold:
        issues.append(
            ValidationIssue(
                code="stale_snapshot",
                station_id=station_id,
                severity="warning",
                message=f"station {station_id} snapshot age {age} exceeds freshness threshold",
            )
        )
    return issues


def _station_name(names: list[LocalizedText]) -> str:
    english = next((name for name in names if name.language == "en"), None)
    return (english or names[0]).text


def _require_utc(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{label} must be UTC")
