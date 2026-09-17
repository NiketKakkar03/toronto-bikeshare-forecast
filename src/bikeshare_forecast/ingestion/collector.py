"""One-shot GBFS collection from discovery through normalized persistence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from bikeshare_forecast.contracts import GbfsDiscovery, GbfsStationInformation, GbfsStationStatus
from bikeshare_forecast.storage import RawCaptureStore, SilverStore
from bikeshare_forecast.validation.snapshots import ValidationPolicy, normalize_station_snapshots


@dataclass(frozen=True)
class CollectionResult:
    snapshots_written: int
    duplicates_ignored: int
    station_count: int
    raw_objects_created: int


def _get_json(client: httpx.Client, url: str, attempts: int) -> object:
    last_error: httpx.HTTPError | None = None
    for _ in range(attempts):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as error:
            last_error = error
    if last_error is None:
        raise ValueError("attempts must be positive")
    raise last_error


def collect_once(
    *,
    discovery_url: str,
    raw_dir: Path,
    silver_dir: Path,
    source_system_id: str = "bike_share_toronto",
    retrieved_at: datetime | None = None,
    timeout_seconds: float = 15.0,
    max_attempts: int = 3,
    validation_policy: ValidationPolicy | None = None,
    client: httpx.Client | None = None,
) -> CollectionResult:
    """Collect advertised station feeds atomically into the silver snapshot store."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    now = retrieved_at or datetime.now(UTC)
    owned = client is None
    selected_client = client or httpx.Client(timeout=timeout_seconds)
    try:
        discovery = GbfsDiscovery.model_validate(
            _get_json(selected_client, discovery_url, max_attempts)
        )
        feeds = {feed.name: str(feed.url) for feed in discovery.data.feeds}
        missing = {"station_information", "station_status"} - feeds.keys()
        if missing:
            raise ValueError(f"discovery is missing required feeds: {sorted(missing)}")
        information = GbfsStationInformation.model_validate(
            _get_json(selected_client, feeds["station_information"], max_attempts)
        )
        status = GbfsStationStatus.model_validate(
            _get_json(selected_client, feeds["station_status"], max_attempts)
        )
    finally:
        if owned:
            selected_client.close()

    raw_store = RawCaptureStore(raw_dir)
    information_capture, information_created = raw_store.capture(
        information,
        source_url=feeds["station_information"],
        feed_name="station_information",
        retrieved_at=now,
        feed_updated_at=information.last_updated,
        schema_version=information.version,
        row_count=len(information.data.stations),
    )
    status_capture, status_created = raw_store.capture(
        status,
        source_url=feeds["station_status"],
        feed_name="station_status",
        retrieved_at=now,
        feed_updated_at=status.last_updated,
        schema_version=status.version,
        row_count=len(status.data.stations),
    )
    lineage_hash = hashlib.sha256(
        f"{information_capture.content_hash}:{status_capture.content_hash}".encode()
    ).hexdigest()
    report = normalize_station_snapshots(
        information,
        status,
        ingested_at=now,
        source_system_id=source_system_id,
        raw_content_hash=lineage_hash,
        policy=validation_policy,
    )
    report.raise_for_errors()
    write = SilverStore(silver_dir).write_snapshots(report.snapshots)
    return CollectionResult(
        snapshots_written=write.rows_written,
        duplicates_ignored=write.duplicates_ignored,
        station_count=len(report.snapshots),
        raw_objects_created=int(information_created) + int(status_created),
    )
