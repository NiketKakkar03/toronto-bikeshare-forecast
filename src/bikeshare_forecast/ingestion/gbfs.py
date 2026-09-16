"""Read-only client for the GBFS auto-discovery document."""

import httpx

from bikeshare_forecast.contracts import GbfsDiscovery


def fetch_discovery(
    url: str,
    *,
    timeout_seconds: float = 15.0,
    client: httpx.Client | None = None,
) -> GbfsDiscovery:
    """Fetch and validate a GBFS discovery document.

    A caller may supply a client for testing or connection reuse. When no client is
    supplied, this function creates and closes one for the request.
    """
    if client is not None:
        return _fetch_with_client(client, url)

    with httpx.Client(timeout=timeout_seconds) as owned_client:
        return _fetch_with_client(owned_client, url)


def _fetch_with_client(client: httpx.Client, url: str) -> GbfsDiscovery:
    response = client.get(url)
    response.raise_for_status()
    return GbfsDiscovery.model_validate(response.json())
