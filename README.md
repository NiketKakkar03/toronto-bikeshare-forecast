# Toronto Bike Share Availability Forecasting Platform

An independent, public-data project for predicting whether a Toronto Bike Share station will
have a rentable bike or an open dock 15, 30, or 60 minutes in the future. The initial product
focus is the 30-minute horizon and honest probabilities rather than guaranteed inventory.

This repository contains the Milestone 0 contracts plus the collector and storage foundation of
Milestone 1. The collector discovers advertised GBFS URLs, retries bounded transient failures,
captures immutable raw JSON, versions station metadata, retains validation/failure reports, and
writes valid snapshots to append-only Parquet. It does not yet produce forecasts.

GBFS v3 calls rentable bikes and e-bikes `vehicles`. The source contract therefore uses
`num_vehicles_available`; a later normalization step will map that source terminology into the
project's rider-facing bike inventory fields.

## Principles

- Every feature must have been available by the forecast creation time.
- Evaluation uses chronological backtests, never random row splits.
- Empty/full probabilities will be calibrated and compared with persistence and seasonal
  baselines.
- Batch and online inference will share preprocessing and prediction code.
- Stale or unavailable feeds and models produce explicit degraded behavior, not false precision.
- Public data sources retain their attribution and licence metadata.

## Set up

Install [uv](https://docs.astral.sh/uv/), then use Python 3.12:

```bash
uv python install 3.12
uv sync --python 3.12 --locked
```

Run all local checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

Inspect the feeds advertised by the live GBFS discovery document:

```bash
uv run python -m bikeshare_forecast.discovery
```

Collect one snapshot and report expected-interval coverage:

```bash
uv run bikeshare ingest-stations --config configs/collection.toml
uv run bikeshare validate-data --config configs/collection.toml
```

The committed tests use only synthetic fixtures and temporary directories and require no network
access. The storage functions accept already validated source models or normalized snapshot
values so a later collector can connect without mixing HTTP behavior into durable-data logic.

Summarize a local silver snapshot directory:

```bash
uv run bikeshare data-summary \
  --silver-dir data/silver/station_snapshots \
  --catalogue data/catalogue.duckdb
```

The snapshot identity is `(station_id, source_last_reported_at, source_system_id)`. Recollecting
the same source state is an idempotent no-op even when ingestion lineage differs; conflicting
states for the same identity are rejected. Validation reports retain missing-station, coverage,
freshness, future-time, and capacity failures instead of filling absent observations.

Import historical source files offline with explicit adapter versions and source lineage:

```bash
uv run bikeshare import-ridership data/trips.csv \
  --source-name toronto-open-data \
  --retrieved-at 2026-09-17T12:00:00Z
uv run bikeshare import-weather data/weather.csv \
  --source-name eccc-historical \
  --retrieved-at 2026-09-17T12:00:00Z
uv run bikeshare historical-summary ridership
```

These commands use the `toronto-ridership-v1` and `eccc-hourly-v1` source contracts. They never
perform network calls. Invalid rows are printed with their source row number and abort the durable
write; valid imports retain file hash, row number, retrieval time, source schema, and adapter
version in Parquet. Add a new adapter version rather than silently changing a published mapping.

## Repository map

- `src/bikeshare_forecast/contracts/`: source and normalized data contracts
- `src/bikeshare_forecast/storage/`: immutable JSON, Parquet, and DuckDB catalogue interfaces
- `src/bikeshare_forecast/validation/`: GBFS normalization and quality reporting
- `tests/`: unit and contract tests
- `configs/collection.toml`: initial collection and freshness policy
- `data/fixtures/gbfs/`: small synthetic GBFS v3 responses
- `docs/prediction-contract.md`: forecast-time and label semantics
- `docs/data-sources.md`: licences, attribution, and source boundaries
- `docs/project-plan.md`: complete staged implementation plan

## Data and disclaimer

The planned system uses the public Bike Share Toronto GBFS feed and City of Toronto open data.
See [docs/data-sources.md](docs/data-sources.md) for attribution and verification requirements.

This project is not affiliated with or endorsed by the City of Toronto, the Toronto Parking
Authority, or Bike Share Toronto. Future predictions are estimates, not guarantees.
