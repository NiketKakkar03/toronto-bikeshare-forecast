"""External data ingestion clients."""

from bikeshare_forecast.ingestion.collector import CollectionResult, collect_once

__all__ = ["CollectionResult", "collect_once"]
