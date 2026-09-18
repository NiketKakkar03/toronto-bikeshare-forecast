"""Dependency-injected forecast sources for artifacts and offline demos."""

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import polars as pl

from bikeshare_forecast.ml.common import read_json, sha256_file
from bikeshare_forecast.ml.evaluate import predict_probability
from bikeshare_forecast.serving.models import (
    ForecastResult,
    ForecastValues,
    ServiceState,
    StationStatus,
)

SUPPORTED_HORIZONS = (15, 30, 60)


class ForecastProvider(Protocol):
    """Boundary implemented later by live data and approved-model adapters."""

    def stations(self) -> Sequence[StationStatus]: ...

    def forecast(self, station_id: str, horizon_minutes: int) -> ForecastResult: ...


def _default_clock() -> datetime:
    return datetime.now(tz=UTC)


class FixtureForecastProvider:
    """Small deterministic provider for development, demos, and offline tests."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = _default_clock,
        observed_at: datetime | None = None,
        forecasts_available: bool = True,
    ) -> None:
        self._clock = clock
        now = clock()
        observation_time = observed_at or now - timedelta(seconds=45)
        self._forecasts_available = forecasts_available
        self._stations = (
            StationStatus(
                station_id="7001",
                name="City Hall",
                latitude=43.6534,
                longitude=-79.3839,
                capacity=20,
                bikes_available=6,
                docks_available=14,
                is_renting=True,
                is_returning=True,
                observed_at=observation_time,
                data_version="fixture-2026-09-17",
            ),
            StationStatus(
                station_id="7002",
                name="University & Elm",
                latitude=43.6578,
                longitude=-79.3891,
                capacity=24,
                bikes_available=13,
                docks_available=11,
                is_renting=True,
                is_returning=True,
                observed_at=observation_time,
                data_version="fixture-2026-09-17",
            ),
            StationStatus(
                station_id="7003",
                name="Nathan Phillips Square",
                latitude=43.6526,
                longitude=-79.3841,
                capacity=18,
                bikes_available=1,
                docks_available=17,
                is_renting=True,
                is_returning=True,
                observed_at=observation_time,
                data_version="fixture-2026-09-17",
            ),
            StationStatus(
                station_id="7004",
                name="Offline Station",
                latitude=43.655,
                longitude=-79.38,
                capacity=12,
                bikes_available=0,
                docks_available=0,
                is_renting=False,
                is_returning=False,
                observed_at=observation_time,
                data_version="fixture-2026-09-17",
            ),
        )

    def stations(self) -> Sequence[StationStatus]:
        return self._stations

    def forecast(self, station_id: str, horizon_minutes: int) -> ForecastResult:
        station = next(item for item in self._stations if item.station_id == station_id)
        if not self._forecasts_available:
            return ForecastResult(
                state=ServiceState.UNAVAILABLE,
                reason="forecast model is unavailable",
                station=station,
                horizon_minutes=horizon_minutes,
                data_version=station.data_version,
            )
        now = self._clock()
        drift = {15: 0.5, 30: 1.2, 60: 2.4}[horizon_minutes]
        bikes = max(0.0, min(float(station.capacity), station.bikes_available - drift))
        docks = float(station.capacity) - bikes
        uncertainty = {15: 1.5, 30: 2.5, 60: 4.0}[horizon_minutes]
        values = ForecastValues(
            bikes_expected=round(bikes, 1),
            docks_expected=round(docks, 1),
            bikes_interval=(
                round(max(0.0, bikes - uncertainty), 1),
                round(min(station.capacity, bikes + uncertainty), 1),
            ),
            docks_interval=(
                round(max(0.0, docks - uncertainty), 1),
                round(min(station.capacity, docks + uncertainty), 1),
            ),
            empty_risk=round(
                min(
                    0.95, 0.04 + (horizon_minutes / 60) * 0.12 + (1 / (station.bikes_available + 1))
                ),
                3,
            ),
            full_risk=round(
                min(
                    0.95, 0.03 + (horizon_minutes / 60) * 0.08 + (1 / (station.docks_available + 1))
                ),
                3,
            ),
        )
        return ForecastResult(
            state=ServiceState.AVAILABLE,
            station=station,
            horizon_minutes=horizon_minutes,
            created_at=now,
            target_time=now + timedelta(minutes=horizon_minutes),
            data_version=station.data_version,
            feature_version="fixture-features-v1",
            model_version="fixture-persistence-v1",
            calibration_version="fixture-calibration-v1",
            forecast=values,
        )


class ArtifactForecastProvider:
    """Serve approved local model artifacts over the latest silver station history."""

    def __init__(
        self,
        silver_dir: Path,
        model_dir: Path,
        *,
        clock: Callable[[], datetime] = _default_clock,
    ) -> None:
        files = sorted(silver_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no normalized snapshots in {silver_dir}")
        self._history = pl.concat([pl.read_parquet(path) for path in files]).sort(
            ["station_id", "source_last_reported_at"]
        )
        self._model_dir = model_dir
        self._clock = clock
        manifest_path = model_dir / "model-manifest.json"
        self._manifest = read_json(manifest_path)
        expected = {str(item["path"]): str(item["sha256"]) for item in self._manifest["models"]}
        for name, digest in expected.items():
            if sha256_file(model_dir / name) != digest:
                raise ValueError(f"model hash mismatch: {name}")
        latest = self._history.group_by("station_id", maintain_order=True).tail(1)
        self._stations = tuple(self._station_status(row) for row in latest.to_dicts())

    @staticmethod
    def _station_status(row: dict[str, object]) -> StationStatus:
        observed_at = row["source_last_reported_at"]
        if not isinstance(observed_at, datetime):
            raise ValueError("source timestamp is not a datetime")
        numeric = {
            key: value
            for key in ("latitude", "longitude", "capacity", "bikes_available", "docks_available")
            if isinstance((value := row[key]), int | float)
        }
        if len(numeric) != 5:
            raise ValueError("station row contains a non-numeric value")
        return StationStatus(
            station_id=str(row["station_id"]),
            name=str(row["station_name"]),
            latitude=float(numeric["latitude"]),
            longitude=float(numeric["longitude"]),
            capacity=int(numeric["capacity"]),
            bikes_available=int(numeric["bikes_available"]),
            docks_available=int(numeric["docks_available"]),
            is_renting=bool(row["is_renting"]),
            is_returning=bool(row["is_returning"]),
            observed_at=observed_at,
            data_version=str(row["raw_content_hash"]),
        )

    def stations(self) -> Sequence[StationStatus]:
        return self._stations

    def forecast(self, station_id: str, horizon_minutes: int) -> ForecastResult:
        station = next(item for item in self._stations if item.station_id == station_id)
        feature_row = self._feature_row(station)
        models = {
            risk: read_json(self._model_dir / f"logistic-{risk}-{horizon_minutes}m.json")
            for risk in ("empty", "full")
        }
        now = self._clock()
        uncertainty = math.sqrt(horizon_minutes / 15) * 1.5
        bikes = float(station.bikes_available)
        docks = float(station.docks_available)
        return ForecastResult(
            state=ServiceState.AVAILABLE,
            station=station,
            horizon_minutes=horizon_minutes,
            created_at=now,
            target_time=now + timedelta(minutes=horizon_minutes),
            data_version=station.data_version,
            feature_version=str(self._manifest["dataset_sha256"]),
            model_version=sha256_file(self._model_dir / "model-manifest.json"),
            calibration_version="uncalibrated-logistic-v1",
            forecast=ForecastValues(
                bikes_expected=bikes,
                docks_expected=docks,
                bikes_interval=(
                    max(0.0, bikes - uncertainty),
                    min(station.capacity, bikes + uncertainty),
                ),
                docks_interval=(
                    max(0.0, docks - uncertainty),
                    min(station.capacity, docks + uncertainty),
                ),
                empty_risk=predict_probability(models["empty"], feature_row),
                full_risk=predict_probability(models["full"], feature_row),
            ),
        )

    def _feature_row(self, station: StationStatus) -> dict[str, object]:
        current = station.observed_at
        station_history = self._history.filter(pl.col("station_id") == station.station_id)

        def lag(minutes: int) -> float | None:
            eligible = station_history.filter(
                pl.col("source_last_reported_at") <= current - timedelta(minutes=minutes)
            )
            return float(eligible.tail(1)["bikes_available"][0]) if eligible.height else None

        minute = current.hour * 60 + current.minute
        weekday = current.weekday()
        return {
            "bikes_available": station.bikes_available,
            "docks_available": station.docks_available,
            "capacity": station.capacity,
            "bike_fraction": station.bikes_available / station.capacity
            if station.capacity
            else 0.0,
            "minute_sin": math.sin(2 * math.pi * minute / 1440),
            "minute_cos": math.cos(2 * math.pi * minute / 1440),
            "weekday_sin": math.sin(2 * math.pi * weekday / 7),
            "weekday_cos": math.cos(2 * math.pi * weekday / 7),
            "bikes_lag_15m": lag(15),
            "bikes_lag_60m": lag(60),
        }
