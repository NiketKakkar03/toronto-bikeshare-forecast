"""Command-line collection and reporting for station data."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from bikeshare_forecast.config import load_collection_config
from bikeshare_forecast.operations import CollectionReportStore, StationCollector
from bikeshare_forecast.storage import DuckDBCatalogue

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


if __name__ == "__main__":
    app()
