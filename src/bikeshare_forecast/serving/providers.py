"""Dependency-injected forecast sources for artifacts and offline demos."""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import polars as pl

from bikeshare_forecast.ml.common import read_json, sha256_file
from bikeshare_forecast.ml.evaluate import predict_value
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
        departures = {15: 1.4, 30: 2.8, 60: 5.6}[horizon_minutes]
        arrivals = {15: 1.0, 30: 2.0, 60: 4.0}[horizon_minutes]
        net_flow = arrivals - departures
        values = ForecastValues(
            departures_expected=departures,
            arrivals_expected=arrivals,
            net_flow_expected=round(net_flow, 2),
            demand_pressure="high" if abs(net_flow) >= 3 else "moderate",
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
            calibration_version=None,
            forecast=values,
        )


class ArtifactForecastProvider:
    """Serve approved local model artifacts over the latest silver station history."""

    def __init__(
        self,
        silver_dir: Path,
        model_dir: Path,
        *,
        dataset_dir: Path | None = None,
        clock: Callable[[], datetime] = _default_clock,
    ) -> None:
        files = sorted(silver_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no normalized snapshots in {silver_dir}")
        self._history = pl.concat([pl.read_parquet(path) for path in files]).sort(
            ["station_id", "source_last_reported_at"]
        )
        self._model_dir = model_dir
        demand_path = (dataset_dir or model_dir.parent / "dataset") / "dataset.parquet"
        if not demand_path.exists():
            raise FileNotFoundError(f"no station-demand dataset at {demand_path}")
        self._demand = pl.read_parquet(demand_path)
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
        try:
            feature_row = self._feature_row(station)
        except ValueError as error:
            return ForecastResult(
                state=ServiceState.UNAVAILABLE,
                reason=str(error),
                station=station,
                horizon_minutes=horizon_minutes,
                data_version=station.data_version,
                feature_version=str(self._manifest["dataset_sha256"]),
                model_version=sha256_file(self._model_dir / "model-manifest.json"),
            )
        models = {
            target: read_json(self._model_dir / f"ridge-{target}-{horizon_minutes}m.json")
            for target in ("departures", "arrivals")
        }
        now = self._clock()
        departures = predict_value(models["departures"], feature_row)
        arrivals = predict_value(models["arrivals"], feature_row)
        net_flow = arrivals - departures
        return ForecastResult(
            state=ServiceState.AVAILABLE,
            station=station,
            horizon_minutes=horizon_minutes,
            created_at=now,
            target_time=now + timedelta(minutes=horizon_minutes),
            data_version=station.data_version,
            feature_version=str(self._manifest["dataset_sha256"]),
            model_version=sha256_file(self._model_dir / "model-manifest.json"),
            calibration_version=None,
            forecast=ForecastValues(
                departures_expected=round(departures, 2),
                arrivals_expected=round(arrivals, 2),
                net_flow_expected=round(net_flow, 2),
                demand_pressure="high"
                if abs(net_flow) >= 5
                else "moderate"
                if abs(net_flow) >= 2
                else "low",
            ),
        )

    def _feature_row(self, station: StationStatus) -> dict[str, object]:
        station_rows = self._demand.filter(pl.col("station_id") == station.station_id)
        if station_rows.is_empty():
            raise ValueError(f"station {station.station_id} has no historical demand features")
        return station_rows.sort("feature_time").tail(1).to_dicts()[0]
