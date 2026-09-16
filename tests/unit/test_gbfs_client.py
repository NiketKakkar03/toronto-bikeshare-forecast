import json
from pathlib import Path

import httpx

from bikeshare_forecast.ingestion.gbfs import fetch_discovery

FIXTURE = Path(__file__).parents[2] / "data" / "fixtures" / "gbfs" / "gbfs.json"


def test_fetch_discovery_uses_advertised_feed_urls() -> None:
    fixture_content = FIXTURE.read_bytes()

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.invalid/gbfs.json"
        return httpx.Response(200, json=json.loads(fixture_content))

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as client:
        discovery = fetch_discovery("https://example.invalid/gbfs.json", client=client)

    feeds = {feed.name: str(feed.url) for feed in discovery.data.feeds}
    assert feeds["station_information"].endswith("station_information.json")
    assert feeds["station_status"].endswith("station_status.json")
