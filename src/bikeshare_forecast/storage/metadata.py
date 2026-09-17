"""Versioned station metadata history and change detection."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bikeshare_forecast.contracts import GbfsStationInformation


class MetadataVersion(BaseModel):
    """One immutable version of a station information record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    station_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    observed_at: datetime
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    record: dict[str, Any]


class MetadataChange(BaseModel):
    """A station-level metadata change observed in a collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    station_id: str
    kind: Literal["added", "changed", "removed"]
    version: int | None = None


@dataclass(frozen=True)
class MetadataWriteResult:
    path: Path | None
    versions_written: int
    changes: tuple[MetadataChange, ...]


class MetadataStore:
    """Persist changed station records as immutable, versioned JSON batches."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def record(
        self, information: GbfsStationInformation, *, observed_at: datetime
    ) -> MetadataWriteResult:
        if observed_at.tzinfo is None or observed_at.utcoffset() != timedelta(0):
            raise ValueError("observed_at must be UTC")
        prior, active = self._state()
        current = {
            item.station_id: item.model_dump(mode="json") for item in information.data.stations
        }
        versions: list[MetadataVersion] = []
        changes: list[MetadataChange] = []
        for station_id, record in sorted(current.items()):
            record_hash = _record_hash(record)
            previous = prior.get(station_id)
            if (
                previous is not None
                and station_id in active
                and previous.record_hash == record_hash
            ):
                continue
            version = 1 if previous is None else previous.version + 1
            versions.append(
                MetadataVersion(
                    station_id=station_id,
                    version=version,
                    observed_at=observed_at,
                    record_hash=record_hash,
                    record=record,
                )
            )
            changes.append(
                MetadataChange(
                    station_id=station_id,
                    kind="added" if station_id not in active else "changed",
                    version=version,
                )
            )
        for station_id in sorted(active - current.keys()):
            changes.append(MetadataChange(station_id=station_id, kind="removed"))
        if not versions and not changes:
            return MetadataWriteResult(path=None, versions_written=0, changes=())

        document = {
            "observed_at": observed_at.isoformat(),
            "source_updated_at": information.last_updated.isoformat(),
            "source_schema_version": information.version,
            "current_station_ids": sorted(current),
            "versions": [version.model_dump(mode="json") for version in versions],
            "changes": [change.model_dump(mode="json") for change in changes],
        }
        encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
        digest = hashlib.sha256(encoded).hexdigest()
        stamp = observed_at.strftime("%Y%m%dT%H%M%S.%fZ")
        destination = self.root / f"metadata-{stamp}-{digest[:12]}.json"
        self.root.mkdir(parents=True, exist_ok=True)
        _write_exclusive(destination, encoded)
        return MetadataWriteResult(
            path=destination,
            versions_written=len(versions),
            changes=tuple(changes),
        )

    def _state(self) -> tuple[dict[str, MetadataVersion], set[str]]:
        latest: dict[str, MetadataVersion] = {}
        active: set[str] = set()
        paths = sorted(self.root.glob("metadata-*.json")) if self.root.exists() else []
        for path in paths:
            document = json.loads(path.read_text(encoding="utf-8"))
            active = set(document["current_station_ids"])
            for value in document["versions"]:
                version = MetadataVersion.model_validate(value)
                prior = latest.get(version.station_id)
                if prior is None or version.version > prior.version:
                    latest[version.station_id] = version
        return latest, active


def _record_hash(record: dict[str, Any]) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
