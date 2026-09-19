from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from bikeshare_forecast.serving import FixtureForecastProvider, create_app

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


def test_model_unavailable_keeps_fresh_current_status() -> None:
    response = client(forecasts_available=False).get("/api/stations/7001/forecast")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "unavailable"
    assert body["forecast"] is None
    assert body["freshness_seconds"] == 45
    assert body["station"]["docks_available"] == 14


def test_rejects_unknown_station_and_horizon() -> None:
    api = client()

    assert api.get("/api/stations/missing/forecast").status_code == 404
    response = api.get("/api/stations/7001/forecast", params={"horizon": 45})
    assert response.status_code == 422
    assert response.json()["detail"] == "horizon must be one of 15, 30, or 60"
