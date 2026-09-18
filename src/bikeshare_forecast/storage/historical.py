"""Durable, content-addressed Parquet storage for historical imports."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from bikeshare_forecast.contracts.historical import (
    HistoricalTrip,
    HistoricalWeatherObservation,
    SourceQualitySummary,
)
from bikeshare_forecast.ingestion.historical import HistoricalImport


@dataclass(frozen=True)
class HistoricalWriteResult:
    parquet_path: Path | None
    quality_path: Path
    rows_written: int
    duplicates_ignored: int


class HistoricalStore:
    """Persist valid historical imports while retaining source and quality lineage."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write_trips(self, imported: HistoricalImport[HistoricalTrip]) -> HistoricalWriteResult:
        imported.require_valid()
        return self._write("ridership", list(imported.records), imported.summary)

    def write_weather(
        self, imported: HistoricalImport[HistoricalWeatherObservation]
    ) -> HistoricalWriteResult:
        imported.require_valid()
        return self._write("weather", list(imported.records), imported.summary)

    def quality_summaries(self, dataset: str) -> list[SourceQualitySummary]:
        return [
            SourceQualitySummary.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted((self.root / dataset / "quality").glob("*.json"))
        ]

    def _write(
        self,
        dataset: str,
        records: list[HistoricalTrip] | list[HistoricalWeatherObservation],
        summary: SourceQualitySummary,
    ) -> HistoricalWriteResult:
        directory = self.root / dataset
        quality_document = (summary.model_dump_json(indent=2) + "\n").encode("utf-8")
        quality_hash = hashlib.sha256(quality_document).hexdigest()
        quality_path = directory / "quality" / f"{quality_hash}.json"
        if not records:
            _write_immutable(quality_path, quality_document)
            return HistoricalWriteResult(None, quality_path, 0, 0)

        frame = _flatten_frame(records)
        existing = _existing_rows(directory / "parquet")
        keys = _identity_columns(dataset)
        pending_rows: list[dict[str, object]] = []
        duplicates = 0
        for row in frame.to_dicts():
            identity = tuple(row[column] for column in keys)
            previous = existing.get(identity)
            if previous is None:
                existing[identity] = row
                pending_rows.append(row)
            elif _source_values(previous) == _source_values(row):
                duplicates += 1
            else:
                raise ValueError(f"conflicting historical {dataset} row: {identity}")
        if not pending_rows:
            _write_immutable(quality_path, quality_document)
            return HistoricalWriteResult(None, quality_path, 0, duplicates)

        batch_key = hashlib.sha256(
            f"{summary.source_name}\0{summary.source_content_hash}".encode()
        ).hexdigest()
        destination = directory / "parquet" / f"part-{batch_key}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        pl.DataFrame(pending_rows, schema=frame.schema).write_parquet(temporary, compression="zstd")
        os.replace(temporary, destination)
        _write_immutable(quality_path, quality_document)
        return HistoricalWriteResult(destination, quality_path, len(pending_rows), duplicates)


def _flatten_frame(
    records: list[HistoricalTrip] | list[HistoricalWeatherObservation],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        values = record.model_dump(exclude={"lineage"})
        values.update(record.lineage.model_dump())
        rows.append(values)
    frame = pl.DataFrame(rows)
    timestamp_columns = [
        name
        for name in ("started_at", "ended_at", "observed_at", "retrieved_at")
        if name in frame.columns
    ]
    return frame.with_columns(
        [pl.col(name).cast(pl.Datetime("us", "UTC")) for name in timestamp_columns]
    )


def _existing_rows(directory: Path) -> dict[tuple[object, ...], dict[str, object]]:
    files = sorted(directory.glob("*.parquet")) if directory.exists() else []
    if not files:
        return {}
    rows = pl.concat([pl.read_parquet(path) for path in files], how="diagonal_relaxed").to_dicts()
    dataset = "ridership" if "trip_id" in rows[0] else "weather"
    keys = _identity_columns(dataset)
    return {tuple(row[column] for column in keys): row for row in rows}


def _identity_columns(dataset: str) -> tuple[str, ...]:
    if dataset == "ridership":
        return ("source_name", "trip_id", "started_at")
    return ("climate_station_id", "observed_at", "source_name")


def _source_values(row: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"retrieved_at", "source_file", "source_row_number", "source_content_hash"}
    }


def _write_immutable(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
