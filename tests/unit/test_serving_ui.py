from datetime import UTC, datetime

from fastapi.testclient import TestClient

from bikeshare_forecast.serving import FixtureForecastProvider, create_app


def test_ui_shell_and_static_assets_are_served_offline() -> None:
    now = datetime(2026, 9, 17, 16, 0, tzinfo=UTC)
    clock = lambda: now  # noqa: E731
    api = TestClient(create_app(FixtureForecastProvider(clock=clock), clock=clock))

    page = api.get("/")
    script = api.get("/static/app.js")
    stylesheet = api.get("/static/app.css")

    assert page.status_code == 200
    assert "Find a station" in page.text
    assert 'id="detail"' in page.text
    assert "/static/app.js" in page.text
    assert script.status_code == 200
    assert "/api/stations/" in script.text
    assert "data-horizon" in script.text
    assert stylesheet.status_code == 200
    assert ".risk-high" in stylesheet.text
