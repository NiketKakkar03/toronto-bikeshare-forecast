"""GBFS discovery and required-feed retrieval with bounded retries."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from pydantic import BaseModel

from bikeshare_forecast.contracts import (
    GbfsDiscovery,
    GbfsStationInformation,
    GbfsStationStatus,
)

Sleep = Callable[[float], None]


class MissingFeedError(ValueError):
    """Raised when discovery does not advertise a required GBFS feed."""


@dataclass(frozen=True)
class StationFeeds:
    """Validated required GBFS payloads and their advertised source URLs."""

    discovery: GbfsDiscovery
    information: GbfsStationInformation
    status: GbfsStationStatus
    information_url: str
    status_url: str


def fetch_discovery(
    url: str,
    *,
    timeout_seconds: float = 15.0,
    max_attempts: int = 1,
    backoff_seconds: float = 1.0,
    client: httpx.Client | None = None,
    sleep: Sleep = time.sleep,
) -> GbfsDiscovery:
    """Fetch and validate a GBFS discovery document."""
    return _with_client(
        url,
        GbfsDiscovery,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        client=client,
        sleep=sleep,
    )


def fetch_station_feeds(
    discovery_url: str,
    *,
    timeout_seconds: float = 15.0,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    client: httpx.Client | None = None,
    sleep: Sleep = time.sleep,
) -> StationFeeds:
    """Discover, fetch, and validate station information and station status."""
    if client is not None:
        return _fetch_station_feeds_with_client(
            client,
            discovery_url,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
        )
    with httpx.Client(timeout=timeout_seconds) as owned_client:
        return _fetch_station_feeds_with_client(
            owned_client,
            discovery_url,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
        )


def _fetch_station_feeds_with_client(
    client: httpx.Client,
    discovery_url: str,
    *,
    max_attempts: int,
    backoff_seconds: float,
    sleep: Sleep,
) -> StationFeeds:
    discovery = _fetch_model(
        client,
        discovery_url,
        GbfsDiscovery,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        sleep=sleep,
    )
    advertised = {feed.name: str(feed.url) for feed in discovery.data.feeds}
    missing = sorted({"station_information", "station_status"} - advertised.keys())
    if missing:
        raise MissingFeedError(f"GBFS discovery is missing required feeds: {', '.join(missing)}")
    information_url = advertised["station_information"]
    status_url = advertised["station_status"]
    information = _fetch_model(
        client,
        information_url,
        GbfsStationInformation,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        sleep=sleep,
    )
    status = _fetch_model(
        client,
        status_url,
        GbfsStationStatus,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        sleep=sleep,
    )
    return StationFeeds(
        discovery=discovery,
        information=information,
        status=status,
        information_url=information_url,
        status_url=status_url,
    )


def _with_client[ModelT: BaseModel](
    url: str,
    model: type[ModelT],
    *,
    timeout_seconds: float,
    max_attempts: int,
    backoff_seconds: float,
    client: httpx.Client | None,
    sleep: Sleep,
) -> ModelT:
    if client is not None:
        return _fetch_model(
            client,
            url,
            model,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
        )
    with httpx.Client(timeout=timeout_seconds) as owned_client:
        return _fetch_model(
            owned_client,
            url,
            model,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
        )


def _fetch_model[ModelT: BaseModel](
    client: httpx.Client,
    url: str,
    model: type[ModelT],
    *,
    max_attempts: int,
    backoff_seconds: float,
    sleep: Sleep,
) -> ModelT:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    for attempt in range(max_attempts):
        try:
            response = client.get(url)
            response.raise_for_status()
            return model.model_validate(response.json())
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status not in {408, 429} and status < 500:
                raise
            if attempt + 1 == max_attempts:
                raise
            sleep(backoff_seconds * (2**attempt))
        except httpx.RequestError:
            if attempt + 1 == max_attempts:
                raise
            sleep(backoff_seconds * (2**attempt))
    raise AssertionError("unreachable")
