"""Station collection orchestration, durable reports, and scheduled execution."""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bikeshare_forecast.config import CollectionConfig
from bikeshare_forecast.ingestion import fetch_station_feeds
from bikeshare_forecast.storage import MetadataStore, RawCaptureStore, SilverStore
from bikeshare_forecast.validation import ValidationIssue, normalize_station_snapshots

Clock = Callable[[], datetime]
Sleep = Callable[[float], None]


class CollectionReport(BaseModel):
    """Durable outcome for one attempted collection interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    collected_at: datetime
    outcome: Literal["success", "validation_failed", "collection_failed"]
    information_content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status_content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_station_count: int = Field(default=0, ge=0)
    observed_station_count: int = Field(default=0, ge=0)
    station_coverage: float = Field(default=0.0, ge=0, le=1)
    rows_written: int = Field(default=0, ge=0)
    duplicates_ignored: int = Field(default=0, ge=0)
    metadata_versions_written: int = Field(default=0, ge=0)
    metadata_changes: int = Field(default=0, ge=0)
    issues: tuple[ValidationIssue, ...] = ()
    error_type: str | None = None
    error_message: str | None = None

    @field_validator("collected_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("collected_at must be UTC")
        return value


class IntervalCoverageReport(BaseModel):
    """Coverage of expected scheduler intervals in a requested UTC window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: datetime
    end: datetime
    interval_seconds: float = Field(gt=0)
    expected_intervals: int = Field(ge=0)
    covered_intervals: int = Field(ge=0)
    missed_intervals: int = Field(ge=0)
    successful_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)


class CollectionReportStore:
    """Append immutable run reports and calculate time-slot coverage."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, report: CollectionReport) -> Path:
        document = (report.model_dump_json(indent=2) + "\n").encode()
        digest = hashlib.sha256(document).hexdigest()
        stamp = report.collected_at.strftime("%Y%m%dT%H%M%S.%fZ")
        destination = self.root / f"collection-{stamp}-{digest[:12]}.json"
        self.root.mkdir(parents=True, exist_ok=True)
        _write_exclusive(destination, document)
        return destination

    def reports(self) -> tuple[CollectionReport, ...]:
        values = (
            [
                CollectionReport.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(self.root.glob("collection-*.json"))
            ]
            if self.root.exists()
            else []
        )
        return tuple(sorted(values, key=lambda report: report.collected_at))

    def coverage(
        self, *, start: datetime, end: datetime, interval_seconds: float
    ) -> IntervalCoverageReport:
        if start.tzinfo is None or start.utcoffset() != timedelta(0):
            raise ValueError("start must be UTC")
        if end.tzinfo is None or end.utcoffset() != timedelta(0):
            raise ValueError("end must be UTC")
        if end < start:
            raise ValueError("end must not precede start")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        duration = (end - start).total_seconds()
        expected = math.ceil(duration / interval_seconds) if duration else 0
        in_window = [report for report in self.reports() if start <= report.collected_at < end]
        successful = [report for report in in_window if report.outcome == "success"]
        slots = {
            min(
                int((report.collected_at - start).total_seconds() // interval_seconds), expected - 1
            )
            for report in successful
            if expected
        }
        covered = len(slots)
        return IntervalCoverageReport(
            start=start,
            end=end,
            interval_seconds=interval_seconds,
            expected_intervals=expected,
            covered_intervals=covered,
            missed_intervals=expected - covered,
            successful_runs=len(successful),
            failed_runs=len(in_window) - len(successful),
            coverage=covered / expected if expected else 1.0,
        )


class StationCollector:
    """Orchestrate discovery, raw capture, validation, metadata, and Silver writes."""

    def __init__(
        self,
        config: CollectionConfig,
        *,
        client: httpx.Client | None = None,
        sleep: Sleep = time.sleep,
        clock: Clock = lambda: datetime.now(UTC),
    ) -> None:
        self.config = config
        self.client = client
        self.sleep = sleep
        self.clock = clock
        self.raw = RawCaptureStore(config.storage.raw_dir)
        self.silver = SilverStore(config.storage.silver_dir)
        self.metadata = MetadataStore(config.storage.metadata_dir)
        self.reports = CollectionReportStore(config.storage.reports_dir)

    def collect_once(self) -> CollectionReport:
        """Run one collection and always retain a success or failure report."""
        collected_at = self.clock().astimezone(UTC)
        information_hash: str | None = None
        status_hash: str | None = None
        try:
            gbfs = self.config.gbfs
            feeds = fetch_station_feeds(
                str(gbfs.discovery_url),
                timeout_seconds=gbfs.request_timeout_seconds,
                max_attempts=gbfs.max_attempts,
                backoff_seconds=gbfs.retry_backoff_seconds,
                client=self.client,
                sleep=self.sleep,
            )
            information_capture, _ = self.raw.capture(
                feeds.information,
                source_url=feeds.information_url,
                feed_name="station_information",
                retrieved_at=collected_at,
                feed_updated_at=feeds.information.last_updated.astimezone(UTC),
                schema_version=feeds.information.version,
                row_count=len(feeds.information.data.stations),
            )
            status_capture, _ = self.raw.capture(
                feeds.status,
                source_url=feeds.status_url,
                feed_name="station_status",
                retrieved_at=collected_at,
                feed_updated_at=feeds.status.last_updated.astimezone(UTC),
                schema_version=feeds.status.version,
                row_count=len(feeds.status.data.stations),
            )
            information_hash = information_capture.content_hash
            status_hash = status_capture.content_hash
            metadata = self.metadata.record(feeds.information, observed_at=collected_at)
            validation = normalize_station_snapshots(
                feeds.information,
                feeds.status,
                ingested_at=collected_at,
                source_system_id=gbfs.source_system_id,
                raw_content_hash=status_hash,
                policy=self.config.validation_policy,
            )
            if not validation.is_valid:
                report = CollectionReport(
                    collected_at=collected_at,
                    outcome="validation_failed",
                    information_content_hash=information_hash,
                    status_content_hash=status_hash,
                    expected_station_count=validation.expected_station_count,
                    observed_station_count=validation.observed_station_count,
                    station_coverage=validation.coverage,
                    metadata_versions_written=metadata.versions_written,
                    metadata_changes=len(metadata.changes),
                    issues=validation.issues,
                )
            else:
                written = self.silver.write_snapshots(validation.snapshots)
                report = CollectionReport(
                    collected_at=collected_at,
                    outcome="success",
                    information_content_hash=information_hash,
                    status_content_hash=status_hash,
                    expected_station_count=validation.expected_station_count,
                    observed_station_count=validation.observed_station_count,
                    station_coverage=validation.coverage,
                    rows_written=written.rows_written,
                    duplicates_ignored=written.duplicates_ignored,
                    metadata_versions_written=metadata.versions_written,
                    metadata_changes=len(metadata.changes),
                    issues=validation.issues,
                )
        except Exception as error:
            report = CollectionReport(
                collected_at=collected_at,
                outcome="collection_failed",
                information_content_hash=information_hash,
                status_content_hash=status_hash,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        self.reports.write(report)
        return report


def run_scheduled(
    collect: Callable[[], CollectionReport],
    *,
    interval_seconds: float,
    max_runs: int | None = None,
    sleep: Sleep = time.sleep,
) -> tuple[CollectionReport, ...]:
    """Repeatedly invoke a collector; ``max_runs`` makes tests and jobs bounded."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if max_runs is not None and max_runs < 0:
        raise ValueError("max_runs must not be negative")
    reports: list[CollectionReport] = []
    while max_runs is None or len(reports) < max_runs:
        reports.append(collect())
        if max_runs is not None and len(reports) >= max_runs:
            break
        sleep(interval_seconds)
    return tuple(reports)


def _write_exclusive(path: Path, contents: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != contents:
            raise RuntimeError(f"immutable artifact collision: {path}") from None
        return
    with os.fdopen(descriptor, "wb") as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())
