"""Versioned, offline adapters for historical source files and payloads."""

from __future__ import annotations

import csv
import hashlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar, cast
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from bikeshare_forecast.contracts.historical import (
    HistoricalTrip,
    HistoricalWeatherObservation,
    ImportFailure,
    SourceLineage,
    SourceQualitySummary,
)

RecordT = TypeVar("RecordT", HistoricalTrip, HistoricalWeatherObservation)
TORONTO = ZoneInfo("America/Toronto")


@dataclass(frozen=True)
class HistoricalImport[RecordT]:
    records: tuple[RecordT, ...]
    failures: tuple[ImportFailure, ...]
    summary: SourceQualitySummary

    def require_valid(self) -> HistoricalImport[RecordT]:
        """Raise with a concise report when any source rows failed validation."""
        if self.failures:
            raise ValueError(f"historical import rejected {len(self.failures)} row(s)")
        return self


@dataclass(frozen=True)
class SourceMetadata:
    source_name: str
    retrieved_at: datetime
    source_url: str | None = None
    licence: str | None = None


class _CsvAdapter[RecordT]:
    dataset: str
    source_schema_version: str
    adapter_version = "1.0.0"

    def from_path(self, path: Path, *, metadata: SourceMetadata) -> HistoricalImport[RecordT]:
        contents = path.read_bytes()
        rows = list(csv.DictReader(io.StringIO(contents.decode("utf-8-sig"))))
        return self.from_rows(
            rows,
            metadata=metadata,
            source_file=path.name,
            source_content_hash=hashlib.sha256(contents).hexdigest(),
        )

    def from_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        metadata: SourceMetadata,
        source_file: str,
        source_content_hash: str,
    ) -> HistoricalImport[RecordT]:
        records: list[RecordT] = []
        failures: list[ImportFailure] = []
        seen: set[tuple[object, ...]] = set()
        duplicate_rows = 0
        for row_number, row in enumerate(rows, start=2):
            plain_row = dict(row)
            try:
                record = self._adapt(
                    plain_row,
                    lineage=SourceLineage(
                        source_name=metadata.source_name,
                        source_file=source_file,
                        source_row_number=row_number,
                        source_content_hash=source_content_hash,
                        source_schema_version=self.source_schema_version,
                        adapter_version=self.adapter_version,
                        retrieved_at=metadata.retrieved_at,
                        source_url=metadata.source_url,
                        licence=metadata.licence,
                    ),
                )
                normalized = cast(HistoricalTrip | HistoricalWeatherObservation, record)
                if normalized.identity in seen:
                    duplicate_rows += 1
                else:
                    seen.add(normalized.identity)
                    records.append(record)
            except (KeyError, TypeError, ValueError, ValidationError) as error:
                failures.append(
                    ImportFailure(
                        row_number=row_number,
                        code=_failure_code(error),
                        message=str(error),
                        row=plain_row,
                    )
                )
        return HistoricalImport(
            records=tuple(records),
            failures=tuple(failures),
            summary=self._summary(
                records, failures, len(rows), duplicate_rows, metadata, source_content_hash
            ),
        )

    def _adapt(self, row: Mapping[str, Any], *, lineage: SourceLineage) -> RecordT:
        raise NotImplementedError

    def _summary(
        self,
        records: Sequence[RecordT],
        failures: Sequence[ImportFailure],
        total_rows: int,
        duplicates: int,
        metadata: SourceMetadata,
        content_hash: str,
    ) -> SourceQualitySummary:
        normalized = [
            cast(HistoricalTrip | HistoricalWeatherObservation, record) for record in records
        ]
        times = [_record_time(record) for record in normalized]
        stations = _station_ids(normalized)
        return SourceQualitySummary(
            dataset=self.dataset,  # type: ignore[arg-type]
            source_name=metadata.source_name,
            source_schema_version=self.source_schema_version,
            adapter_version=self.adapter_version,
            source_content_hash=content_hash,
            total_rows=total_rows,
            accepted_rows=len(records),
            rejected_rows=len(failures),
            duplicate_rows=duplicates,
            first_observed_at=min(times) if times else None,
            last_observed_at=max(times) if times else None,
            distinct_station_count=len(stations),
            missing_optional_value_count=sum(
                _missing_optional_values(record) for record in normalized
            ),
        )


class TorontoRidershipV1Adapter(_CsvAdapter[HistoricalTrip]):
    """Adapter for the stable, documented v1 fixture/source column contract."""

    dataset = "ridership"
    source_schema_version = "toronto-ridership-v1"

    def _adapt(self, row: Mapping[str, Any], *, lineage: SourceLineage) -> HistoricalTrip:
        started = _local_datetime(_required(row, "trip_start_time"))
        ended = _local_datetime(_required(row, "trip_stop_time"))
        return HistoricalTrip(
            trip_id=_clean_id(_required(row, "trip_id")),
            started_at=started,
            ended_at=ended,
            duration_seconds=int(_required(row, "trip_duration_seconds")),
            start_station_id=_clean_id(_required(row, "from_station_id")),
            start_station_name=_optional(row, "from_station_name"),
            end_station_id=_clean_id(_required(row, "to_station_id")),
            end_station_name=_optional(row, "to_station_name"),
            bike_type=_optional(row, "bike_type"),
            user_type=_optional(row, "user_type"),
            lineage=lineage,
        )


class EcccHourlyV1Adapter(_CsvAdapter[HistoricalWeatherObservation]):
    """Adapter for selected ECCC bulk hourly CSV columns with UTC timestamps."""

    dataset = "weather"
    source_schema_version = "eccc-hourly-v1"

    def _adapt(
        self, row: Mapping[str, Any], *, lineage: SourceLineage
    ) -> HistoricalWeatherObservation:
        return HistoricalWeatherObservation(
            climate_station_id=_clean_id(_required(row, "Climate ID")),
            observed_at=_utc_datetime(_required(row, "Date/Time (UTC)")),
            temperature_c=_optional_float(row, "Temp (°C)"),
            precipitation_mm=_optional_float(row, "Precip. Amount (mm)"),
            snow_cm=_optional_float(row, "Snow on Grnd (cm)"),
            wind_speed_kph=_optional_float(row, "Wind Spd (km/h)"),
            relative_humidity_pct=_optional_float(row, "Rel Hum (%)"),
            condition=_optional(row, "Weather"),
            lineage=lineage,
        )


def _required(row: Mapping[str, Any], column: str) -> str:
    if column not in row:
        raise KeyError(f"missing required column: {column}")
    value = str(row[column]).strip()
    if not value:
        raise ValueError(f"empty required value: {column}")
    return value


def _optional(row: Mapping[str, Any], column: str) -> str | None:
    value = row.get(column)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _optional_float(row: Mapping[str, Any], column: str) -> float | None:
    value = _optional(row, column)
    return None if value is None else float(value)


def _clean_id(value: str) -> str:
    normalized = value.strip()
    if normalized.endswith(".0") and normalized[:-2].isdigit():
        normalized = normalized[:-2]
    if not normalized:
        raise ValueError("identifier is empty")
    return normalized


def _local_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TORONTO)
    return parsed.astimezone(UTC)


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _failure_code(
    error: Exception,
) -> Literal["missing_column", "invalid_value", "invalid_timestamp", "invalid_row"]:
    message = str(error).lower()
    if isinstance(error, KeyError):
        return "missing_column"
    if "time" in message or "date" in message:
        return "invalid_timestamp"
    if isinstance(error, (TypeError, ValueError)):
        return "invalid_value"
    return "invalid_row"


def _record_time(record: HistoricalTrip | HistoricalWeatherObservation) -> datetime:
    return record.started_at if isinstance(record, HistoricalTrip) else record.observed_at


def _station_ids(
    records: Sequence[HistoricalTrip | HistoricalWeatherObservation],
) -> set[str]:
    result: set[str] = set()
    for record in records:
        if isinstance(record, HistoricalTrip):
            result.update((record.start_station_id, record.end_station_id))
        else:
            result.add(record.climate_station_id)
    return result


def _missing_optional_values(record: HistoricalTrip | HistoricalWeatherObservation) -> int:
    values: tuple[object | None, ...]
    if isinstance(record, HistoricalTrip):
        values = (
            record.start_station_name,
            record.end_station_name,
            record.bike_type,
            record.user_type,
        )
    else:
        values = (
            record.temperature_c,
            record.precipitation_mm,
            record.snow_cm,
            record.wind_speed_kph,
            record.relative_humidity_pct,
            record.condition,
        )
    return sum(value is None for value in values)
