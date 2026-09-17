"""Dependency-injected forecast source with an offline deterministic fixture."""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from bikeshare_forecast.serving.models import (
    ForecastResult,
    ForecastValues,
    ServiceState,
    StationStatus,
)

SUPPORTED_HORIZONS = (15, 30, 60)


class ForecastProvider(Protocol):
    """Boundary implemented later by live data and approved-model adapters."""

    def stations(self) -> Sequence[StationStatus]: ...

    def forecast(self, station_id: str, horizon_minutes: int) -> ForecastResult: ...


def _default_clock() -> datetime:
    return datetime.now(tz=UTC)


class FixtureForecastProvider:
    """Small deterministic provider for development, demos, and offline tests."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = _default_clock,
        observed_at: datetime | None = None,
        forecasts_available: bool = True,
    ) -> None:
        self._clock = clock
        now = clock()
        observation_time = observed_at or now - timedelta(seconds=45)
        self._forecasts_available = forecasts_available
        self._stations = (
            StationStatus(
                station_id="7001",
                name="City Hall",
                latitude=43.6534,
                longitude=-79.3839,
                capacity=20,
                bikes_available=6,
                docks_available=14,
                is_renting=True,
                is_returning=True,
                observed_at=observation_time,
                data_version="fixture-2026-09-17",
            ),
            StationStatus(
                station_id="7002",
                name="University & Elm",
                latitude=43.6578,
                longitude=-79.3891,
                capacity=24,
                bikes_available=13,
                docks_available=11,
                is_renting=True,
                is_returning=True,
                observed_at=observation_time,
                data_version="fixture-2026-09-17",
            ),
            StationStatus(
                station_id="7003",
                name="Nathan Phillips Square",
                latitude=43.6526,
                longitude=-79.3841,
                capacity=18,
                bikes_available=1,
                docks_available=17,
                is_renting=True,
                is_returning=True,
                observed_at=observation_time,
                data_version="fixture-2026-09-17",
            ),
            StationStatus(
                station_id="7004",
                name="Offline Station",
                latitude=43.655,
                longitude=-79.38,
                capacity=12,
                bikes_available=0,
                docks_available=0,
                is_renting=False,
                is_returning=False,
                observed_at=observation_time,
                data_version="fixture-2026-09-17",
            ),
        )

    def stations(self) -> Sequence[StationStatus]:
        return self._stations

    def forecast(self, station_id: str, horizon_minutes: int) -> ForecastResult:
        station = next(item for item in self._stations if item.station_id == station_id)
        if not self._forecasts_available:
            return ForecastResult(
                state=ServiceState.UNAVAILABLE,
                reason="forecast model is unavailable",
                station=station,
                horizon_minutes=horizon_minutes,
                data_version=station.data_version,
            )
        now = self._clock()
        drift = {15: 0.5, 30: 1.2, 60: 2.4}[horizon_minutes]
        bikes = max(0.0, min(float(station.capacity), station.bikes_available - drift))
        docks = float(station.capacity) - bikes
        uncertainty = {15: 1.5, 30: 2.5, 60: 4.0}[horizon_minutes]
        values = ForecastValues(
            bikes_expected=round(bikes, 1),
            docks_expected=round(docks, 1),
            bikes_interval=(
                round(max(0.0, bikes - uncertainty), 1),
                round(min(station.capacity, bikes + uncertainty), 1),
            ),
            docks_interval=(
                round(max(0.0, docks - uncertainty), 1),
                round(min(station.capacity, docks + uncertainty), 1),
            ),
            empty_risk=round(
                min(
                    0.95, 0.04 + (horizon_minutes / 60) * 0.12 + (1 / (station.bikes_available + 1))
                ),
                3,
            ),
            full_risk=round(
                min(
                    0.95, 0.03 + (horizon_minutes / 60) * 0.08 + (1 / (station.docks_available + 1))
                ),
                3,
            ),
        )
        return ForecastResult(
            state=ServiceState.AVAILABLE,
            station=station,
            horizon_minutes=horizon_minutes,
            created_at=now,
            target_time=now + timedelta(minutes=horizon_minutes),
            data_version=station.data_version,
            feature_version="fixture-features-v1",
            model_version="fixture-persistence-v1",
            calibration_version="fixture-calibration-v1",
            forecast=values,
        )
