"""Command-line reporting for locally persisted station data."""

import tomllib
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from bikeshare_forecast.ingestion import collect_once
from bikeshare_forecast.ml import (
    DatasetConfig,
    backtest_run,
    build_dataset,
    evaluate_run,
    train_models,
)
from bikeshare_forecast.storage import DuckDBCatalogue

app = typer.Typer(no_args_is_help=True)


def _format_time(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "none"


@app.callback()
def main() -> None:
    """Inspect locally persisted Toronto Bike Share data."""


@app.command("data-summary")
def data_summary(
    silver_dir: Annotated[Path, typer.Option()] = Path("data/silver/station_snapshots"),
    catalogue: Annotated[Path, typer.Option()] = Path("data/catalogue.duckdb"),
) -> None:
    """Print a deterministic summary of normalized station snapshots."""
    summary = DuckDBCatalogue(catalogue, silver_dir).summary()
    typer.echo(f"rows: {summary.row_count}")
    typer.echo(f"stations: {summary.station_count}")
    typer.echo(f"source_time_start: {_format_time(summary.first_source_at)}")
    typer.echo(f"source_time_end: {_format_time(summary.last_source_at)}")
    typer.echo(f"duplicate_keys: {summary.duplicate_key_count}")


@app.command("ingest-stations")
def ingest_stations(
    config: Annotated[Path, typer.Option()] = Path("configs/collection.toml"),
    raw_dir: Annotated[Path, typer.Option()] = Path("data/raw/gbfs"),
    silver_dir: Annotated[Path, typer.Option()] = Path("data/silver/station_snapshots"),
) -> None:
    """Discover, validate, and persist one live GBFS station snapshot."""
    with config.open("rb") as handle:
        values = tomllib.load(handle)["gbfs"]
    result = collect_once(
        discovery_url=str(values["discovery_url"]),
        raw_dir=raw_dir,
        silver_dir=silver_dir,
        source_system_id=str(values["source_system_id"]),
        timeout_seconds=float(values["request_timeout_seconds"]),
        max_attempts=int(values["max_attempts"]),
    )
    typer.echo(f"stations: {result.station_count}")
    typer.echo(f"snapshots_written: {result.snapshots_written}")
    typer.echo(f"duplicates_ignored: {result.duplicates_ignored}")


@app.command("dataset-build")
def dataset_build(
    silver_dir: Annotated[Path, typer.Option()] = Path("data/silver/station_snapshots"),
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/dataset"),
    config: Annotated[Path, typer.Option()] = Path("configs/training.toml"),
) -> None:
    """Build a point-in-time dataset with 15/30/60-minute targets."""
    path = build_dataset(
        silver_dir,
        output_dir,
        DatasetConfig.from_toml(config),
    )
    typer.echo(path)


@app.command("train")
def train(
    dataset_dir: Annotated[Path, typer.Option()] = Path("artifacts/dataset"),
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/models"),
) -> None:
    """Train deterministic logistic unavailability classifiers."""
    typer.echo(train_models(dataset_dir, output_dir))


@app.command("evaluate")
def evaluate(
    dataset_dir: Annotated[Path, typer.Option()] = Path("artifacts/dataset"),
    model_dir: Annotated[Path, typer.Option()] = Path("artifacts/models"),
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/evaluation"),
) -> None:
    """Evaluate classifiers and inventory baselines on the held-out test split."""
    typer.echo(evaluate_run(dataset_dir, model_dir, output_dir))


@app.command("backtest")
def backtest(
    dataset_dir: Annotated[Path, typer.Option()] = Path("artifacts/dataset"),
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/backtest"),
    config: Annotated[Path, typer.Option()] = Path("configs/training.toml"),
) -> None:
    """Run configured expanding-window temporal evaluation folds."""
    with config.open("rb") as handle:
        values = tomllib.load(handle)["backtest"]
    typer.echo(
        backtest_run(
            dataset_dir,
            output_dir,
            folds=int(values["folds"]),
            minimum_train_fraction=float(values["minimum_train_fraction"]),
            evaluation_fraction=float(values["evaluation_fraction"]),
        )
    )


if __name__ == "__main__":
    app()
