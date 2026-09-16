"""Minimal GBFS v3 source contracts needed by the first ingestion milestone."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class GbfsModel(BaseModel):
    """Base source model that rejects unexpected schema changes."""

    model_config = ConfigDict(extra="forbid")


class GbfsStationRecord(BaseModel):
    """Required station fields, while allowing optional GBFS extensions.

    The bronze layer will preserve the complete source response. These source
    contracts validate fields needed for normalization without duplicating every
    optional field in the GBFS specification.
    """

    model_config = ConfigDict(extra="ignore")


class FeedReference(GbfsModel):
    name: str = Field(min_length=1)
    url: HttpUrl


class DiscoveryData(GbfsModel):
    feeds: list[FeedReference]


class GbfsDiscovery(GbfsModel):
    last_updated: datetime
    ttl: int = Field(ge=0)
    version: str = Field(min_length=1)
    data: DiscoveryData


class LocalizedText(GbfsModel):
    text: str = Field(min_length=1)
    language: str = Field(min_length=2)


class VehicleDockCapacity(GbfsModel):
    vehicle_type_ids: list[str]
    count: int = Field(ge=0)


class StationInformationRecord(GbfsStationRecord):
    station_id: str = Field(min_length=1)
    name: list[LocalizedText] = Field(min_length=1)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    capacity: int = Field(ge=0)
    vehicle_docks_capacity: list[VehicleDockCapacity] = Field(default_factory=list)


class StationInformationData(GbfsModel):
    stations: list[StationInformationRecord]


class GbfsStationInformation(GbfsModel):
    last_updated: datetime
    ttl: int = Field(ge=0)
    version: str = Field(min_length=1)
    data: StationInformationData


class VehicleTypeAvailability(GbfsModel):
    vehicle_type_id: str = Field(min_length=1)
    count: int = Field(ge=0)


class VehicleDockAvailability(GbfsModel):
    vehicle_type_ids: list[str]
    count: int = Field(ge=0)


class StationStatusRecord(GbfsStationRecord):
    station_id: str = Field(min_length=1)
    num_vehicles_available: int = Field(ge=0)
    num_vehicles_disabled: int = Field(ge=0)
    num_docks_available: int = Field(ge=0)
    num_docks_disabled: int = Field(ge=0)
    is_installed: bool
    is_renting: bool
    is_returning: bool
    last_reported: datetime
    vehicle_docks_available: list[VehicleDockAvailability] = Field(default_factory=list)
    vehicle_types_available: list[VehicleTypeAvailability] = Field(default_factory=list)


class StationStatusData(GbfsModel):
    stations: list[StationStatusRecord]


class GbfsStationStatus(GbfsModel):
    last_updated: datetime
    ttl: int = Field(ge=0)
    version: str = Field(min_length=1)
    data: StationStatusData
