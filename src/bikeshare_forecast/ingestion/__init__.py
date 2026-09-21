"""External data ingestion clients."""

from bikeshare_forecast.ingestion.collector import CollectionResult, collect_once
from bikeshare_forecast.ingestion.gbfs import (
    MissingFeedError,
    StationFeeds,
    fetch_discovery,
    fetch_station_feeds,
)
from bikeshare_forecast.ingestion.historical import (
    EcccHourlyV1Adapter,
    EcccTorontoCity2024Adapter,
    HistoricalImport,
    SourceMetadata,
    TorontoRidershipV1Adapter,
)
from bikeshare_forecast.ingestion.toronto_bulk import (
    BulkImportResult,
    import_toronto_ridership_2024,
)

__all__ = [
    "BulkImportResult",
    "CollectionResult",
    "EcccHourlyV1Adapter",
    "EcccTorontoCity2024Adapter",
    "HistoricalImport",
    "MissingFeedError",
    "SourceMetadata",
    "StationFeeds",
    "TorontoRidershipV1Adapter",
    "collect_once",
    "fetch_discovery",
    "fetch_station_feeds",
    "import_toronto_ridership_2024",
]
