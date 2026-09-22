from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from bikeshare_forecast.serving import FixtureForecastProvider, create_app
from bikeshare_forecast.serving.models import (
    ForecastResult,
    ForecastValues,
    ServiceState,
    StationStatus,
)

NOW = datetime(2026, 9, 17, 16, 0, tzinfo=UTC)


def client(
    *,
    observed_at: datetime | None = None,
    forecasts_available: bool = True,
) -> TestClient:
    clock = lambda: NOW  # noqa: E731
    provider = FixtureForecastProvider(
        clock=clock,
        observed_at=observed_at,
        forecasts_available=forecasts_available,
    )
    return TestClient(create_app(provider, clock=clock))


def test_health_and_station_search() -> None:
    api = client()

    health = api.get("/health")
    result = api.get("/api/stations", params={"q": "university"})

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "station_count": 4,
        "max_data_age_seconds": 45,
        "freshness_limit_seconds": 300,
    }
    assert [station["station_id"] for station in result.json()] == ["7002"]


def test_forecast_includes_versions_demand_and_operational_alternatives() -> None:
    response = client().get("/api/stations/7001/forecast", params={"horizon": 30})

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "available"
    assert body["horizon_minutes"] == 30
    assert body["freshness_seconds"] == 45
    assert body["data_version"] == "fixture-2026-09-17"
    assert body["feature_version"] == "fixture-features-v1"
    assert body["model_version"] == "fixture-persistence-v1"
    assert body["calibration_version"] is None
    assert body["forecast"]["departures_expected"] >= 0
    assert body["forecast"]["arrivals_expected"] >= 0
    assert body["forecast"]["net_flow_expected"] == -0.8
    assert body["forecast"]["demand_pressure"] == "moderate"
    assert body["guidance"] == {
        "headline": "Good for pickup and return",
        "recommendation": "Use this station",
        "pickup_risk": "low",
        "return_risk": "low",
        "explanation": (
            "6 bikes and 14 docks are available now. Over the next 30 minutes, "
            "the model expects about 2.8 pickups and 2.0 returns."
        ),
    }
    assert [alternative["station_id"] for alternative in body["alternatives"]] == [
        "7003",
        "7002",
    ]
    assert "7004" not in {item["station_id"] for item in body["alternatives"]}


def test_stale_data_suppresses_forecast_but_keeps_current_status() -> None:
    response = client(observed_at=NOW - timedelta(minutes=10)).get("/api/stations/7001/forecast")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "stale"
    assert body["forecast"] is None
    assert body["station"]["bikes_available"] == 6
    assert body["alternatives"] == []
    assert "suppressed" in body["reason"]


def test_guidance_flags_return_risk_when_docks_are_full() -> None:
    station = StationStatus(
        station_id="7100",
        name="Full Station",
        latitude=43.65,
        longitude=-79.38,
        capacity=27,
        bikes_available=25,
        docks_available=0,
        is_renting=True,
        is_returning=True,
        observed_at=NOW,
        data_version="test",
    )

    class Provider:
        def stations(self) -> tuple[StationStatus, ...]:
            return (station,)

        def forecast(self, station_id: str, horizon_minutes: int) -> ForecastResult:
            return ForecastResult(
                state=ServiceState.AVAILABLE,
                station=station,
                horizon_minutes=horizon_minutes,
                data_version=station.data_version,
                forecast=ForecastValues(
                    departures_expected=0.79,
                    arrivals_expected=0.24,
                    net_flow_expected=-0.56,
                    demand_pressure="low",
                ),
            )

    response = TestClient(create_app(Provider(), clock=lambda: NOW)).get(
        "/api/stations/7100/forecast"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["guidance"]["headline"] == "Risky for return"
    assert body["guidance"]["pickup_risk"] == "low"
    assert body["guidance"]["return_risk"] == "high"


def test_model_unavailable_keeps_fresh_current_status() -> None:
    response = client(forecasts_available=False).get("/api/stations/7001/forecast")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "unavailable"
    assert body["forecast"] is None
    assert body["freshness_seconds"] == 45
    assert body["station"]["docks_available"] == 14


def test_unavailable_alternative_does_not_break_forecast_response() -> None:
    stations = (
        StationStatus(
            station_id="7001",
            name="Selected",
            latitude=43.65,
            longitude=-79.38,
            capacity=20,
            bikes_available=6,
            docks_available=14,
            is_renting=True,
            is_returning=True,
            observed_at=NOW,
            data_version="test",
        ),
        StationStatus(
            station_id="7382",
            name="Missing History",
            latitude=43.651,
            longitude=-79.381,
            capacity=20,
            bikes_available=7,
            docks_available=13,
            is_renting=True,
            is_returning=True,
            observed_at=NOW,
            data_version="test",
        ),
    )

    class Provider:
        def stations(self) -> tuple[StationStatus, ...]:
            return stations

        def forecast(self, station_id: str, horizon_minutes: int) -> ForecastResult:
            station = next(item for item in stations if item.station_id == station_id)
            if station_id == "7382":
                return ForecastResult(
                    state=ServiceState.UNAVAILABLE,
                    reason="station 7382 has no historical demand features",
                    station=station,
                    horizon_minutes=horizon_minutes,
                    data_version=station.data_version,
                )
            return ForecastResult(
                state=ServiceState.AVAILABLE,
                station=station,
                horizon_minutes=horizon_minutes,
                data_version=station.data_version,
                forecast=ForecastValues(
                    departures_expected=1,
                    arrivals_expected=2,
                    net_flow_expected=1,
                    demand_pressure="low",
                ),
            )

    response = TestClient(create_app(Provider(), clock=lambda: NOW)).get(
        "/api/stations/7001/forecast"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "available"
    assert body["alternatives"] == []


def test_alternatives_only_forecast_nearby_candidate_window() -> None:
    stations = [
        StationStatus(
            station_id="7001",
            name="Selected",
            latitude=43.65,
            longitude=-79.38,
            capacity=20,
            bikes_available=6,
            docks_available=14,
            is_renting=True,
            is_returning=True,
            observed_at=NOW,
            data_version="test",
        )
    ]
    stations.extend(
        StationStatus(
            station_id=f"8{index:03d}",
            name=f"Candidate {index}",
            latitude=43.65 + index * 0.0001,
            longitude=-79.38,
            capacity=20,
            bikes_available=6,
            docks_available=14,
            is_renting=True,
            is_returning=True,
            observed_at=NOW,
            data_version="test",
        )
        for index in range(1, 31)
    )
    forecasted: list[str] = []

    class Provider:
        def stations(self) -> tuple[StationStatus, ...]:
            return tuple(stations)

        def forecast(self, station_id: str, horizon_minutes: int) -> ForecastResult:
            forecasted.append(station_id)
            station = next(item for item in stations if item.station_id == station_id)
            return ForecastResult(
                state=ServiceState.AVAILABLE,
                station=station,
                horizon_minutes=horizon_minutes,
                data_version=station.data_version,
                forecast=ForecastValues(
                    departures_expected=1,
                    arrivals_expected=2,
                    net_flow_expected=1,
                    demand_pressure="low",
                ),
            )

    response = TestClient(create_app(Provider(), clock=lambda: NOW)).get(
        "/api/stations/7001/forecast"
    )

    assert response.status_code == 200
    assert forecasted == ["7001", *(f"8{index:03d}" for index in range(1, 21))]
    assert [item["station_id"] for item in response.json()["alternatives"]] == [
        "8001",
        "8002",
        "8003",
    ]


def test_rejects_unknown_station_and_horizon() -> None:
    api = client()

    assert api.get("/api/stations/missing/forecast").status_code == 404
    response = api.get("/api/stations/7001/forecast", params={"horizon": 45})
    assert response.status_code == 422
    assert response.json()["detail"] == "horizon must be one of 15, 30, or 60"
