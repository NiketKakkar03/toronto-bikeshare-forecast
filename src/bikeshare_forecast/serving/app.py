"""FastAPI application for station status and forecasts."""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bikeshare_forecast.serving.models import (
    Alternative,
    ForecastResponse,
    ForecastResult,
    RiderGuidance,
    ServiceState,
    StationStatus,
)
from bikeshare_forecast.serving.providers import (
    SUPPORTED_HORIZONS,
    FixtureForecastProvider,
    ForecastProvider,
)

DEFAULT_FRESHNESS_LIMIT_SECONDS = 300
ALTERNATIVE_FORECAST_CANDIDATES = 20
STATIC_DIR = Path(__file__).with_name("static")


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _distance_metres(first: StationStatus, second: StationStatus) -> int:
    radius = 6_371_000
    lat1, lat2 = radians(first.latitude), radians(second.latitude)
    delta_lat = lat2 - lat1
    delta_lon = radians(second.longitude - first.longitude)
    haversine = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
    return round(2 * radius * asin(sqrt(haversine)))


def _station(provider: ForecastProvider, station_id: str) -> StationStatus:
    station = next((item for item in provider.stations() if item.station_id == station_id), None)
    if station is None:
        raise HTTPException(status_code=404, detail="station not found")
    return station


def _alternatives(
    provider: ForecastProvider,
    selected: StationStatus,
    horizon: int,
) -> tuple[Alternative, ...]:
    candidates: list[Alternative] = []
    nearby = sorted(
        (
            (_distance_metres(selected, station), station)
            for station in provider.stations()
            if station.station_id != selected.station_id
            and station.is_renting
            and station.is_returning
        ),
        key=lambda item: (item[0], item[1].station_id),
    )
    for distance, station in nearby[:ALTERNATIVE_FORECAST_CANDIDATES]:
        result = provider.forecast(station.station_id, horizon)
        if result.state is not ServiceState.AVAILABLE or result.forecast is None:
            continue
        candidates.append(
            Alternative(
                station_id=station.station_id,
                name=station.name,
                distance_metres=distance,
                bikes_available=station.bikes_available,
                docks_available=station.docks_available,
                departures_expected=result.forecast.departures_expected,
                arrivals_expected=result.forecast.arrivals_expected,
                demand_pressure=result.forecast.demand_pressure,
            )
        )
    return tuple(sorted(candidates, key=lambda item: (item.distance_metres, item.station_id))[:3])


def _risk_level(available: float, capacity: int) -> str:
    if available <= 1:
        return "high"
    if available <= 3 or (capacity and available / capacity < 0.15):
        return "medium"
    return "low"


def _guidance(result: ForecastResult) -> RiderGuidance | None:
    if result.forecast is None:
        return None
    station = result.station
    forecast = result.forecast
    bikes_expected = max(0.0, station.bikes_available + forecast.net_flow_expected)
    docks_expected = max(0.0, station.docks_available - forecast.net_flow_expected)
    pickup_risk = _risk_level(bikes_expected, station.capacity)
    return_risk = _risk_level(docks_expected, station.capacity)
    if pickup_risk == "low" and return_risk == "low":
        headline = "Good for pickup and return"
        recommendation = "Use this station"
    elif pickup_risk == "high":
        headline = "Risky for pickup"
        recommendation = "Consider a nearby station with more bikes"
    elif return_risk == "high":
        headline = "Risky for return"
        recommendation = "Consider a nearby station with more open docks"
    elif pickup_risk == "medium" or return_risk == "medium":
        headline = "Usable, but keep a backup"
        recommendation = "Check a nearby alternative before relying on it"
    else:
        headline = "Station looks stable"
        recommendation = "Use this station"
    explanation = (
        f"{station.bikes_available} bikes and {station.docks_available} docks are available now. "
        f"Over the next {result.horizon_minutes} minutes, the model expects about "
        f"{forecast.departures_expected:.1f} pickups and {forecast.arrivals_expected:.1f} returns."
    )
    return RiderGuidance(
        headline=headline,
        recommendation=recommendation,
        pickup_risk=pickup_risk,
        return_risk=return_risk,
        explanation=explanation,
    )


def create_app(
    provider: ForecastProvider | None = None,
    *,
    clock: Callable[[], datetime] = _utc_now,
    freshness_limit_seconds: int = DEFAULT_FRESHNESS_LIMIT_SECONDS,
) -> FastAPI:
    """Build an application whose data and forecast source is replaceable."""
    source = provider or FixtureForecastProvider(clock=clock)
    app = FastAPI(title="Toronto Bike Share Forecast", version="0.1.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def freshness(station: StationStatus) -> int:
        return max(0, int((clock() - station.observed_at).total_seconds()))

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, object]:
        stations = source.stations()
        ages = [freshness(station) for station in stations]
        healthy = bool(stations) and max(ages) <= freshness_limit_seconds
        return {
            "status": "ok" if healthy else "degraded",
            "station_count": len(stations),
            "max_data_age_seconds": max(ages, default=None),
            "freshness_limit_seconds": freshness_limit_seconds,
        }

    @app.get("/api/stations", response_model=list[StationStatus])
    def stations(q: str | None = Query(default=None, max_length=100)) -> Sequence[StationStatus]:
        values = source.stations()
        if not q:
            return values
        needle = q.casefold().strip()
        return [
            station
            for station in values
            if needle in station.name.casefold() or needle in station.station_id.casefold()
        ]

    @app.get("/api/stations/{station_id}", response_model=StationStatus)
    def station_status(station_id: str) -> StationStatus:
        return _station(source, station_id)

    @app.get("/api/stations/{station_id}/forecast", response_model=ForecastResponse)
    def station_forecast(
        station_id: str,
        horizon: int = Query(default=30),
    ) -> ForecastResponse:
        if horizon not in SUPPORTED_HORIZONS:
            raise HTTPException(status_code=422, detail="horizon must be one of 15, 30, or 60")
        current = _station(source, station_id)
        age = freshness(current)
        if age > freshness_limit_seconds:
            return ForecastResponse(
                state=ServiceState.STALE,
                reason=f"station data is {age} seconds old; forecasts are suppressed",
                station=current,
                horizon_minutes=horizon,
                data_version=current.data_version,
                freshness_seconds=age,
            )
        result = source.forecast(station_id, horizon)
        return ForecastResponse(
            **result.model_dump(),
            freshness_seconds=age,
            guidance=_guidance(result),
            alternatives=_alternatives(source, current, horizon)
            if result.state is ServiceState.AVAILABLE
            else (),
        )

    return app


app = create_app()
