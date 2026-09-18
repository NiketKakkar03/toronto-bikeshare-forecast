"""Command-line collection, modeling, and reporting for station data."""

import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from bikeshare_forecast.config import load_collection_config
from bikeshare_forecast.ingestion import (
    EcccHourlyV1Adapter,
    SourceMetadata,
    TorontoRidershipV1Adapter,
)
from bikeshare_forecast.ml import (
    DatasetConfig,
    backtest_run,
    build_dataset,
    evaluate_run,
    train_models,
)
from bikeshare_forecast.operations import CollectionReportStore, StationCollector
from bikeshare_forecast.storage import DuckDBCatalogue, HistoricalStore

app = typer.Typer(no_args_is_help=True)


def _format_time(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "none"


def _parse_utc(value: str | None, *, option: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise typer.BadParameter("must be an ISO 8601 timestamp", param_hint=option) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise typer.BadParameter("must include a UTC offset", param_hint=option)
    return parsed.astimezone(UTC)


@app.callback()
def main() -> None:
    """Collect and inspect locally persisted Toronto Bike Share data."""


@app.command("ingest-stations")
def ingest_stations(
    config_path: Annotated[Path, typer.Option("--config")] = Path("configs/collection.toml"),
) -> None:
    """Discover and collect one station information/status snapshot."""
    report = StationCollector(load_collection_config(config_path)).collect_once()
    typer.echo(f"outcome: {report.outcome}")
    typer.echo(f"station_coverage: {report.station_coverage:.3f}")
    typer.echo(f"rows_written: {report.rows_written}")
    typer.echo(f"duplicates_ignored: {report.duplicates_ignored}")
    if report.error_message:
        typer.echo(f"error: {report.error_type}: {report.error_message}", err=True)
    for issue in report.issues:
        typer.echo(f"issue: {issue.code}: {issue.message}", err=True)
    if report.outcome != "success":
        raise typer.Exit(code=1)


@app.command("validate-data")
def validate_data(
    config_path: Annotated[Path, typer.Option("--config")] = Path("configs/collection.toml"),
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Report successful coverage of expected collection intervals."""
    config = load_collection_config(config_path)
    store = CollectionReportStore(config.storage.reports_dir)
    reports = store.reports()
    if not reports:
        typer.echo("no collection reports found", err=True)
        raise typer.Exit(code=1)
    interval = config.gbfs.poll_interval_seconds
    selected_start = _parse_utc(start, option="--start") or reports[0].collected_at
    selected_end = _parse_utc(end, option="--end") or (
        reports[-1].collected_at + timedelta(seconds=interval)
    )
    coverage = store.coverage(
        start=selected_start,
        end=selected_end,
        interval_seconds=interval,
    )
    typer.echo(f"window_start: {coverage.start.isoformat()}")
    typer.echo(f"window_end: {coverage.end.isoformat()}")
    typer.echo(f"expected_intervals: {coverage.expected_intervals}")
    typer.echo(f"covered_intervals: {coverage.covered_intervals}")
    typer.echo(f"missed_intervals: {coverage.missed_intervals}")
    typer.echo(f"failed_runs: {coverage.failed_runs}")
    typer.echo(f"collection_coverage: {coverage.coverage:.3f}")


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


def _metadata(source_name: str, retrieved_at: str) -> SourceMetadata:
    try:
        parsed = datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
    except ValueError:
        raise typer.BadParameter("retrieved-at must be an ISO 8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise typer.BadParameter("retrieved-at must include a UTC offset")
    return SourceMetadata(source_name=source_name, retrieved_at=parsed)


@app.command("import-ridership")
def import_ridership(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    source_name: Annotated[str, typer.Option()],
    retrieved_at: Annotated[str, typer.Option()],
    output_dir: Annotated[Path, typer.Option()] = Path("data/historical"),
) -> None:
    """Validate and persist a local Toronto ridership v1 CSV."""
    imported = TorontoRidershipV1Adapter().from_path(
        source, metadata=_metadata(source_name, retrieved_at)
    )
    if imported.failures:
        for failure in imported.failures:
            typer.echo(f"row {failure.row_number}: {failure.code}: {failure.message}", err=True)
        raise typer.Exit(code=2)
    result = HistoricalStore(output_dir).write_trips(imported)
    typer.echo(f"accepted_rows: {imported.summary.accepted_rows}")
    typer.echo(f"duplicate_rows: {imported.summary.duplicate_rows + result.duplicates_ignored}")
    typer.echo(f"rows_written: {result.rows_written}")


@app.command("import-weather")
def import_weather(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    source_name: Annotated[str, typer.Option()],
    retrieved_at: Annotated[str, typer.Option()],
    output_dir: Annotated[Path, typer.Option()] = Path("data/historical"),
) -> None:
    """Validate and persist a local ECCC hourly v1 CSV."""
    imported = EcccHourlyV1Adapter().from_path(
        source, metadata=_metadata(source_name, retrieved_at)
    )
    if imported.failures:
        for failure in imported.failures:
            typer.echo(f"row {failure.row_number}: {failure.code}: {failure.message}", err=True)
        raise typer.Exit(code=2)
    result = HistoricalStore(output_dir).write_weather(imported)
    typer.echo(f"accepted_rows: {imported.summary.accepted_rows}")
    typer.echo(f"duplicate_rows: {imported.summary.duplicate_rows + result.duplicates_ignored}")
    typer.echo(f"rows_written: {result.rows_written}")


@app.command("historical-summary")
def historical_summary(
    dataset: Annotated[str, typer.Argument()],
    data_dir: Annotated[Path, typer.Option()] = Path("data/historical"),
) -> None:
    """Print accumulated source-quality summaries for a historical dataset."""
    if dataset not in {"ridership", "weather"}:
        raise typer.BadParameter("dataset must be ridership or weather")
    summaries = HistoricalStore(data_dir).quality_summaries(dataset)
    typer.echo(f"imports: {len(summaries)}")
    typer.echo(f"source_rows: {sum(item.total_rows for item in summaries)}")
    typer.echo(f"accepted_rows: {sum(item.accepted_rows for item in summaries)}")
    typer.echo(f"rejected_rows: {sum(item.rejected_rows for item in summaries)}")
    typer.echo(f"duplicate_rows: {sum(item.duplicate_rows for item in summaries)}")


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
