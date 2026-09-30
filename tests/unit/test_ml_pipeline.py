from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from bikeshare_forecast.cli import app
from bikeshare_forecast.contracts import StationSnapshot
from bikeshare_forecast.ml import (
    backtest_run,
    build_dataset,
    diagnostics_run,
    evaluate_run,
    score_batch,
    train_models,
)
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
    boosted = report["metrics"]["30m"]["departures"]["hist_gradient_boosting"]
    assert "rmse" in boosted["regression"]
    assert {"precision_at_0_5", "recall_at_0_5", "pr_auc", "roc_auc", "brier_score"} <= boosted[
        "demand_event"
    ].keys()
    assert "mae" in report["metrics"]["60m"]["net_flow"]["derived_ridge"]
    predictions = tmp_path / "predictions"
    scored = score_batch(first, models, predictions)
    assert scored.exists()
    prediction_frame = pl.read_parquet(scored)
    assert "predicted_departures_30m" in prediction_frame.columns
    assert "probability_departures_event_30m" in prediction_frame.columns
    backtest = read_json(
        backtest_run(first, tmp_path / "backtest", folds=2, minimum_train_fraction=0.4)
    )
    assert backtest["fold_count"] == 2
    assert "30m" in backtest["mean_metrics"]
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
    historical, dataset, models, reports, predictions, backtests = (
        tmp_path / "historical",
        tmp_path / "dataset",
        tmp_path / "models",
        tmp_path / "reports",
        tmp_path / "predictions",
        tmp_path / "backtests",
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
    scored = runner.invoke(
        app,
        [
            "score-batch",
            "--dataset-dir",
            str(dataset),
            "--model-dir",
            str(models),
            "--output-dir",
            str(predictions),
        ],
    )
    backtested = runner.invoke(
        app,
        [
            "backtest",
            "--dataset-dir",
            str(dataset),
            "--output-dir",
            str(backtests),
            "--folds",
            "2",
            "--minimum-train-fraction",
            "0.4",
        ],
    )
    assert built.exit_code == trained.exit_code == evaluated.exit_code == scored.exit_code == 0
    assert backtested.exit_code == 0, backtested.output
    assert (reports / "evaluation.json").exists()
    assert (predictions / "prediction-manifest.json").exists()
    assert (backtests / "backtest.json").exists()


def test_station_diagnostics_report_and_cli(tmp_path: Path) -> None:
    historical = tmp_path / "historical"
    dataset = tmp_path / "dataset"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    _history(historical)
    build_dataset(historical, dataset)
    train_models(dataset, models)

    report_path = diagnostics_run(dataset, models, reports, worst_station_limit=1)
    report = read_json(report_path)
    departures = report["diagnostics"]["15m"]["departures"]
    assert report["evaluation_split"] == "test"
    assert departures["station_count"] == 2
    assert 0 <= departures["stations_beating_baseline_percent"] <= 100
    assert len(departures["worst_stations"]) == 1
    assert {"mae", "rmse", "bias", "baseline_mae"} <= departures["stations"][0].keys()
    assert departures["hour_of_day"]
    assert departures["weekday"]
    assert {row["period"] for row in departures["rush_hour"]} <= {"rush_hour", "other"}
    assert {row["activity_segment"] for row in departures["activity_segments"]} <= {
        "quiet",
        "medium",
        "busy",
    }

    cli_reports = tmp_path / "cli-reports"
    result = CliRunner().invoke(
        app,
        [
            "diagnostics",
            "--dataset-dir",
            str(dataset),
            "--model-dir",
            str(models),
            "--output-dir",
            str(cli_reports),
            "--worst-stations",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (cli_reports / "diagnostics.json").exists()
