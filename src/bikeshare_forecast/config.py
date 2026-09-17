"""Typed configuration for station collection and validation."""

from __future__ import annotations

import tomllib
from datetime import timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from bikeshare_forecast.validation import ValidationPolicy


class GbfsCollectionConfig(BaseModel):
    """Network and scheduling settings for the GBFS source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    discovery_url: HttpUrl
    source_system_id: str = Field(min_length=1)
    poll_interval_seconds: float = Field(gt=0)
    freshness_threshold_seconds: float = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(ge=1)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)


class ValidationConfig(BaseModel):
    """Source-quality boundaries loaded from collection.toml."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    toronto_latitude_min: float = Field(ge=-90, le=90)
    toronto_latitude_max: float = Field(ge=-90, le=90)
    toronto_longitude_min: float = Field(ge=-180, le=180)
    toronto_longitude_max: float = Field(ge=-180, le=180)
    future_timestamp_tolerance_seconds: float = Field(ge=0)
    capacity_tolerance: int = Field(default=0, ge=0)
    minimum_station_coverage: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def ordered_bounds(self) -> ValidationConfig:
        if self.toronto_latitude_min > self.toronto_latitude_max:
            raise ValueError("latitude minimum must not exceed maximum")
        if self.toronto_longitude_min > self.toronto_longitude_max:
            raise ValueError("longitude minimum must not exceed maximum")
        return self

    def policy(self, *, freshness_threshold_seconds: float) -> ValidationPolicy:
        return ValidationPolicy(
            freshness_threshold=timedelta(seconds=freshness_threshold_seconds),
            future_timestamp_tolerance=timedelta(seconds=self.future_timestamp_tolerance_seconds),
            capacity_tolerance=self.capacity_tolerance,
            minimum_station_coverage=self.minimum_station_coverage,
            latitude_min=self.toronto_latitude_min,
            latitude_max=self.toronto_latitude_max,
            longitude_min=self.toronto_longitude_min,
            longitude_max=self.toronto_longitude_max,
        )


class StorageConfig(BaseModel):
    """Local durable-data locations, relative to the process working directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_dir: Path = Path("data/raw/gbfs")
    silver_dir: Path = Path("data/silver/station_snapshots")
    reports_dir: Path = Path("data/reports/collection")
    metadata_dir: Path = Path("data/silver/station_metadata")


class CollectionConfig(BaseModel):
    """Complete configuration for the station collector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gbfs: GbfsCollectionConfig
    validation: ValidationConfig
    storage: StorageConfig = StorageConfig()

    @property
    def validation_policy(self) -> ValidationPolicy:
        return self.validation.policy(
            freshness_threshold_seconds=self.gbfs.freshness_threshold_seconds
        )


def load_collection_config(path: Path) -> CollectionConfig:
    """Load and validate a TOML collection configuration."""
    with path.open("rb") as config_file:
        return CollectionConfig.model_validate(tomllib.load(config_file))
