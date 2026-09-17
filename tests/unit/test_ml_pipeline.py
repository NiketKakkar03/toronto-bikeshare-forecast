from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from bikeshare_forecast.cli import app
from bikeshare_forecast.contracts import StationSnapshot
from bikeshare_forecast.ml import build_dataset, evaluate_run, train_models
from bikeshare_forecast.ml.common import read_json
from bikeshare_forecast.storage import SilverStore


def _fixtures() -> list[StationSnapshot]:
    """Return reproducible normalized history with both unavailable classes."""
    start = datetime(2026, 8, 31, tzinfo=UTC)
    values = []
    for station_index, station_id in enumerate(("7001", "7002")):
        for step in range(9 * 24 * 4):
            event_time = start + timedelta(minutes=15 * step)
            bikes = (step // 4 + step + station_index * 3) % 11
            values.append(
                StationSnapshot(
                    station_id=station_id,
                    station_name=f"Station {station_id}",
                    latitude=43.65 + station_index / 100,
                    longitude=-79.38,
                    capacity=10,
                    bikes_available=bikes,
                    docks_available=10 - bikes,
                    is_installed=True,
                    is_renting=True,
                    is_returning=True,
                    source_last_reported_at=event_time,
                    ingested_at=event_time + timedelta(seconds=5),
                    source_system_id="bike_share_toronto",
                    source_schema_version="3.0",
                    raw_content_hash=f"{step + station_index:064x}",
                )
            )
    return values


def test_pipeline_is_deterministic_point_in_time_and_persists_artifacts(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    SilverStore(silver).write_snapshots(_fixtures())
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_dataset(silver, first)
    build_dataset(silver, second)
    first_manifest = read_json(first / "dataset-manifest.json")
    second_manifest = read_json(second / "dataset-manifest.json")
    assert first_manifest["dataset_sha256"] == second_manifest["dataset_sha256"]

    frame = pl.read_parquet(first / "dataset.parquet")
    assert set(frame["split"].unique()) == {"train", "validation", "test"}
    assert (frame["target_time_15m"] > frame["feature_time"]).all()
    assert (frame["target_time_60m"] > frame["feature_time"]).all()
    split_ranges = frame.group_by("split").agg(
        pl.col("feature_time").min().alias("minimum"),
        pl.col("feature_time").max().alias("maximum"),
    )
    ranges = {row["split"]: row for row in split_ranges.to_dicts()}
    assert ranges["train"]["maximum"] < ranges["validation"]["minimum"]
    assert ranges["validation"]["maximum"] < ranges["test"]["minimum"]

    models = tmp_path / "models"
    report_dir = tmp_path / "reports"
    train_models(first, models)
    report = read_json(evaluate_run(first, models, report_dir))
    assert report["rows"] > 0
    assert set(report["metrics"]) == {"15m", "30m", "60m"}
    assert "roc_auc" in report["metrics"]["15m"]["classification"]["logistic"]
    assert "rmse" in report["metrics"]["60m"]["inventory_regression"]["persistence"]


def test_pipeline_cli_commands(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    SilverStore(silver).write_snapshots(_fixtures())
    dataset = tmp_path / "dataset"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    runner = CliRunner()

    built = runner.invoke(
        app, ["dataset-build", "--silver-dir", str(silver), "--output-dir", str(dataset)]
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

    assert built.exit_code == 0
    assert trained.exit_code == 0
    assert evaluated.exit_code == 0
    assert (reports / "evaluation.json").exists()
