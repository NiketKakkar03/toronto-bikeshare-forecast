"""Typed external-source and normalized-domain contracts."""

from bikeshare_forecast.contracts.gbfs import (
    GbfsDiscovery,
    GbfsStationInformation,
    GbfsStationStatus,
)
from bikeshare_forecast.contracts.historical import (
    HistoricalTrip,
    HistoricalWeatherObservation,
    ImportFailure,
    SourceLineage,
    SourceQualitySummary,
)
from bikeshare_forecast.contracts.snapshot import StationSnapshot

__all__ = [
    "GbfsDiscovery",
    "GbfsStationInformation",
    "GbfsStationStatus",
    "HistoricalTrip",
    "HistoricalWeatherObservation",
    "ImportFailure",
    "SourceLineage",
    "SourceQualitySummary",
    "StationSnapshot",
]
