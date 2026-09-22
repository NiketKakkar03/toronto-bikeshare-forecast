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
    """Expected station demand during the selected future horizon."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    departures_expected: float = Field(ge=0)
    arrivals_expected: float = Field(ge=0)
    net_flow_expected: float
    demand_pressure: str

    @model_validator(mode="after")
    def net_flow_is_consistent(self) -> "ForecastValues":
        expected = self.arrivals_expected - self.departures_expected
        if abs(self.net_flow_expected - expected) > 0.11:
            raise ValueError("net flow must equal arrivals minus departures")
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
    departures_expected: float = Field(ge=0)
    arrivals_expected: float = Field(ge=0)
    demand_pressure: str


class RiderGuidance(BaseModel):
    """Plain-language station recommendation derived from status and forecast."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    headline: str
    recommendation: str
    pickup_risk: str
    return_risk: str
    explanation: str


class ForecastResponse(ForecastResult):
    """Complete API forecast response."""

    freshness_seconds: int = Field(ge=0)
    guidance: RiderGuidance | None = None
    alternatives: tuple[Alternative, ...] = ()
