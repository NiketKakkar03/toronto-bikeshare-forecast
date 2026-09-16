import json
from pathlib import Path

from bikeshare_forecast.contracts import (
    GbfsDiscovery,
    GbfsStationInformation,
    GbfsStationStatus,
)

FIXTURES = Path(__file__).parents[2] / "data" / "fixtures" / "gbfs"


def load_fixture(name: str) -> object:
    with (FIXTURES / name).open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def test_discovery_advertises_required_feeds() -> None:
    discovery = GbfsDiscovery.model_validate(load_fixture("gbfs.json"))

    feed_names = {feed.name for feed in discovery.data.feeds}
    assert {"station_information", "station_status"} <= feed_names


def test_station_information_fixture_matches_contract() -> None:
    information = GbfsStationInformation.model_validate(load_fixture("station_information.json"))

    assert information.version == "3.0"
    station = information.data.stations[0]
    assert station.capacity == 15
    assert station.name[0].text == "Synthetic Station"
    assert station.name[0].language == "en"


def test_station_status_fixture_matches_contract() -> None:
    status = GbfsStationStatus.model_validate(load_fixture("station_status.json"))

    station = status.data.stations[0]
    assert station.num_vehicles_available == 4
    assert station.num_docks_available == 11
    assert sum(vehicle.count for vehicle in station.vehicle_types_available) == 4
