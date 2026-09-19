from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from bikeshare_forecast.cli import app
from bikeshare_forecast.contracts import StationSnapshot
from bikeshare_forecast.ml import build_dataset, evaluate_run, train_models
from bikeshare_forecast.ml.common import read_json
from bikeshare_forecast.serving import ArtifactForecastProvider
from bikeshare_forecast.storage import SilverStore


def _history(root: Path) -> None:
    start = datetime(2026, 8, 1, tzinfo=UTC)
    rows = []
    for step in range(10 * 24 * 4):
        when = start + timedelta(minutes=15 * step)
        for trip in range((step % 4) + 1):
            rows.append(
                {
                    "trip_id": f"{step}-{trip}",
                    "started_at": when + timedelta(minutes=trip),
                    "ended_at": when + timedelta(minutes=8 + trip),
                    "start_station_id": "7001" if step % 2 else "7002",
                    "end_station_id": "7002" if step % 2 else "7001",
                    "start_station_name": "Station A" if step % 2 else "Station B",
                    "end_station_name": "Station B" if step % 2 else "Station A",
                }
            )
    destination = root / "ridership" / "parquet"
    destination.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(destination / "part-test.parquet")


def _snapshots() -> list[StationSnapshot]:
    when = datetime(2026, 8, 11, tzinfo=UTC)
    return [
        StationSnapshot(
            station_id=station_id,
            station_name=name,
            latitude=43.65 + index / 100,
            longitude=-79.38,
            capacity=20,
            bikes_available=8,
            docks_available=12,
            is_installed=True,
            is_renting=True,
            is_returning=True,
            source_last_reported_at=when,
            ingested_at=when + timedelta(seconds=5),
            source_system_id="bike_share_toronto",
            source_schema_version="3.0",
            raw_content_hash=f"{index:064x}",
        )
        for index, (station_id, name) in enumerate(
            (("7001", "Station A"), ("7002", "Station B")), 1
        )
    ]


def test_demand_pipeline_is_temporal_and_persists_artifacts(tmp_path: Path) -> None:
    historical = tmp_path / "historical"
    _history(historical)
    first, second = tmp_path / "first", tmp_path / "second"
    build_dataset(historical, first)
    build_dataset(historical, second)
    assert (
        read_json(first / "dataset-manifest.json")["dataset_sha256"]
        == read_json(second / "dataset-manifest.json")["dataset_sha256"]
    )
    frame = pl.read_parquet(first / "dataset.parquet")
    assert set(frame["split"].unique()) == {"train", "validation", "test"}
    assert (frame["target_time_60m"] > frame["feature_time"]).all()
    assert "target_departures_30m" in frame.columns
    assert "target_arrivals_30m" in frame.columns
    models, reports = tmp_path / "models", tmp_path / "reports"
    train_models(first, models)
    report = read_json(evaluate_run(first, models, reports))
    assert "rmse" in report["metrics"]["30m"]["departures"]["ridge"]
    assert "mae" in report["metrics"]["60m"]["net_flow"]["derived_ridge"]
    silver = tmp_path / "silver"
    SilverStore(silver).write_snapshots(_snapshots())
    provider = ArtifactForecastProvider(
        silver, models, dataset_dir=first, clock=lambda: datetime(2026, 8, 11, tzinfo=UTC)
    )
    forecast = provider.forecast("7001", 30).forecast
    assert forecast is not None
    assert forecast.departures_expected >= 0
    assert forecast.arrivals_expected >= 0
    assert forecast.demand_pressure in {"low", "moderate", "high"}


def test_pipeline_cli_commands(tmp_path: Path) -> None:
    historical, dataset, models, reports = (
        tmp_path / "historical",
        tmp_path / "dataset",
        tmp_path / "models",
        tmp_path / "reports",
    )
    _history(historical)
    runner = CliRunner()
    built = runner.invoke(
        app, ["dataset-build", "--historical-dir", str(historical), "--output-dir", str(dataset)]
    )
    trained = runner.invoke(
        app, ["train", "--dataset-dir", str(dataset), "--output-dir", str(models)]
    )
    evaluated = runner.invoke(
        app,
        [
            "evaluate",
            "--dataset-dir",
            str(dataset),
            "--model-dir",
            str(models),
            "--output-dir",
            str(reports),
        ],
    )
    assert built.exit_code == trained.exit_code == evaluated.exit_code == 0
    assert (reports / "evaluation.json").exists()
