"""External data ingestion clients."""

from bikeshare_forecast.ingestion.gbfs import (
    MissingFeedError,
    StationFeeds,
    fetch_discovery,
    fetch_station_feeds,
)
from bikeshare_forecast.ingestion.historical import (
    EcccHourlyV1Adapter,
    HistoricalImport,
    SourceMetadata,
    TorontoRidershipV1Adapter,
)

__all__ = [
    "EcccHourlyV1Adapter",
    "HistoricalImport",
    "MissingFeedError",
    "SourceMetadata",
    "StationFeeds",
    "TorontoRidershipV1Adapter",
    "fetch_discovery",
    "fetch_station_feeds",
]
