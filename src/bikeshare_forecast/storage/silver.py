"""Append-only Parquet persistence for normalized station snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from bikeshare_forecast.contracts import StationSnapshot


@dataclass(frozen=True)
class WriteResult:
    path: Path | None
    rows_written: int
    duplicates_ignored: int


class SilverStore:
    """Persist immutable batches while enforcing the station observation key."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write_snapshots(self, values: Iterable[StationSnapshot]) -> WriteResult:
        incoming = list(values)
        unique: dict[tuple[object, ...], StationSnapshot] = {}
        duplicates = 0
        for snapshot in incoming:
            prior = unique.get(snapshot.identity)
            if prior is not None:
                if not _same_source_observation(prior, snapshot):
                    raise ValueError(f"conflicting duplicate snapshot: {snapshot.identity}")
                duplicates += 1
                continue
            unique[snapshot.identity] = snapshot

        existing = self._existing_by_identity()
        pending: list[StationSnapshot] = []
        for identity, snapshot in unique.items():
            prior = existing.get(identity)
            if prior is None:
                pending.append(snapshot)
            elif _same_source_observation(prior, snapshot):
                duplicates += 1
            else:
                raise ValueError(f"conflicting persisted snapshot: {identity}")
        if not pending:
            return WriteResult(path=None, rows_written=0, duplicates_ignored=duplicates)

        pending.sort(key=lambda item: item.identity)
        digest_input = json.dumps(
            [item.model_dump(mode="json") for item in pending],
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        batch_hash = hashlib.sha256(digest_input).hexdigest()
        destination = self.root / f"part-{batch_hash}.parquet"
        if destination.exists():
            return WriteResult(path=destination, rows_written=0, duplicates_ignored=len(incoming))

        self.root.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        _snapshot_frame(pending).write_parquet(temporary, compression="zstd")
        os.replace(temporary, destination)
        return WriteResult(
            path=destination,
            rows_written=len(pending),
            duplicates_ignored=duplicates,
        )

    def _existing_by_identity(self) -> dict[tuple[object, ...], StationSnapshot]:
        files = sorted(self.root.glob("*.parquet")) if self.root.exists() else []
        if not files:
            return {}
        rows = pl.concat([pl.read_parquet(path) for path in files]).to_dicts()
        snapshots = [StationSnapshot.model_validate(row) for row in rows]
        return {snapshot.identity: snapshot for snapshot in snapshots}


def _snapshot_frame(snapshots: list[StationSnapshot]) -> pl.DataFrame:
    return pl.DataFrame(
        [snapshot.model_dump() for snapshot in snapshots],
        schema_overrides={
            "capacity": pl.Int32,
            "bikes_available": pl.Int32,
            "docks_available": pl.Int32,
            "source_last_reported_at": pl.Datetime("us", "UTC"),
            "ingested_at": pl.Datetime("us", "UTC"),
        },
    )


def _same_source_observation(left: StationSnapshot, right: StationSnapshot) -> bool:
    """Compare source state while keeping first-seen ingestion lineage immutable."""
    excluded = {"ingested_at", "raw_content_hash"}
    return left.model_dump(exclude=excluded) == right.model_dump(exclude=excluded)
