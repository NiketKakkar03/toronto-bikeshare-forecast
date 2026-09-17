"""Canonical contracts for historical ridership and weather data."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class SourceLineage(BaseModel):
    """Immutable pointer from a normalized row back to its local source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_name: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_row_number: int = Field(ge=2)
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_schema_version: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    retrieved_at: datetime
    source_url: str | None = None
    licence: str | None = None

    @field_validator("retrieved_at")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        return _require_utc(value)


class HistoricalTrip(BaseModel):
    """A source trip normalized to stable station and UTC time semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trip_id: str = Field(min_length=1)
    started_at: datetime
    ended_at: datetime
    duration_seconds: int = Field(ge=0)
    start_station_id: str = Field(min_length=1)
    start_station_name: str | None = None
    end_station_id: str = Field(min_length=1)
    end_station_name: str | None = None
    bike_type: str | None = None
    user_type: str | None = None
    lineage: SourceLineage

    @field_validator("started_at", "ended_at")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @model_validator(mode="after")
    def validate_interval(self) -> HistoricalTrip:
        if self.ended_at < self.started_at:
            raise ValueError("trip end precedes trip start")
        return self

    @property
    def identity(self) -> tuple[str, str, datetime]:
        return (self.lineage.source_name, self.trip_id, self.started_at)


class HistoricalWeatherObservation(BaseModel):
    """An hourly weather observation normalized to UTC."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    climate_station_id: str = Field(min_length=1)
    observed_at: datetime
    temperature_c: float | None = None
    precipitation_mm: float | None = Field(default=None, ge=0)
    snow_cm: float | None = Field(default=None, ge=0)
    wind_speed_kph: float | None = Field(default=None, ge=0)
    relative_humidity_pct: float | None = Field(default=None, ge=0, le=100)
    condition: str | None = None
    lineage: SourceLineage

    @field_validator("observed_at")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @property
    def identity(self) -> tuple[str, datetime, str]:
        return (self.climate_station_id, self.observed_at, self.lineage.source_name)


class ImportFailure(BaseModel):
    """A rejected source row with enough context to reproduce the failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_number: int = Field(ge=2)
    code: Literal["missing_column", "invalid_value", "invalid_timestamp", "invalid_row"]
    message: str = Field(min_length=1)
    row: dict[str, Any]


class SourceQualitySummary(BaseModel):
    """Deterministic quality metrics emitted by every historical import."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: Literal["ridership", "weather"]
    source_name: str
    source_schema_version: str
    adapter_version: str
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_rows: int = Field(ge=0)
    accepted_rows: int = Field(ge=0)
    rejected_rows: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)
    first_observed_at: datetime | None = None
    last_observed_at: datetime | None = None
    distinct_station_count: int = Field(ge=0)
    missing_optional_value_count: int = Field(ge=0)

    @field_validator("first_observed_at", "last_observed_at")
    @classmethod
    def normalize_optional_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_utc(value)
