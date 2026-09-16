"""Normalized station snapshot contract for the future silver data layer."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StationSnapshot(BaseModel):
    """Point-in-time station state with source and ingestion lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    station_id: str = Field(min_length=1)
    station_name: str = Field(min_length=1)
    latitude: float = Field(ge=43.4, le=44.0)
    longitude: float = Field(ge=-79.8, le=-79.0)
    capacity: int = Field(ge=0)
    bikes_available: int = Field(ge=0)
    docks_available: int = Field(ge=0)
    is_installed: bool
    is_renting: bool
    is_returning: bool
    source_last_reported_at: datetime
    ingested_at: datetime
    source_system_id: str = Field(min_length=1)
    source_schema_version: str = Field(min_length=1)
    raw_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_last_reported_at", "ingested_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        """Reject naive or non-UTC timestamps at the durable-data boundary."""
        utc_offset = value.utcoffset()
        if value.tzinfo is None or utc_offset is None:
            raise ValueError("timestamp must be timezone-aware")
        if utc_offset.total_seconds() != 0:
            raise ValueError("timestamp must be UTC")
        return value

    @property
    def identity(self) -> tuple[str, datetime, str]:
        """Return the durable identity used for idempotent writes."""
        return (self.station_id, self.source_last_reported_at, self.source_system_id)
