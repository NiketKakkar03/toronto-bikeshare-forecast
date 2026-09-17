"""Public contracts for station status and forecast serving."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServiceState(StrEnum):
    """Availability of a forecast response."""

    AVAILABLE = "available"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class StationStatus(BaseModel):
    """Current rider-facing station state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    station_id: str
    name: str
    latitude: float
    longitude: float
    capacity: int = Field(ge=0)
    bikes_available: int = Field(ge=0)
    docks_available: int = Field(ge=0)
    is_renting: bool
    is_returning: bool
    observed_at: datetime
    data_version: str


class ForecastValues(BaseModel):
    """Point prediction and calibrated station failure risks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bikes_expected: float = Field(ge=0)
    docks_expected: float = Field(ge=0)
    bikes_interval: tuple[float, float]
    docks_interval: tuple[float, float]
    empty_risk: float = Field(ge=0, le=1)
    full_risk: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def intervals_are_ordered(self) -> "ForecastValues":
        if self.bikes_interval[0] > self.bikes_interval[1]:
            raise ValueError("bikes interval must be ordered")
        if self.docks_interval[0] > self.docks_interval[1]:
            raise ValueError("docks interval must be ordered")
        return self


class ForecastResult(BaseModel):
    """Provider result before nearby alternatives are attached."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: ServiceState
    reason: str | None = None
    station: StationStatus
    horizon_minutes: int
    created_at: datetime | None = None
    target_time: datetime | None = None
    data_version: str
    feature_version: str | None = None
    model_version: str | None = None
    calibration_version: str | None = None
    forecast: ForecastValues | None = None


class Alternative(BaseModel):
    """Nearby operational station with the requested forecast."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    station_id: str
    name: str
    distance_metres: int = Field(ge=0)
    bikes_available: int = Field(ge=0)
    docks_available: int = Field(ge=0)
    empty_risk: float = Field(ge=0, le=1)
    full_risk: float = Field(ge=0, le=1)


class ForecastResponse(ForecastResult):
    """Complete API forecast response."""

    freshness_seconds: int = Field(ge=0)
    alternatives: tuple[Alternative, ...] = ()
