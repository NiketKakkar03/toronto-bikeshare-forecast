# Toronto Bike Share Availability Forecasting Platform

An independent, public-data project for predicting whether a Toronto Bike Share station will
have a rentable bike or an open dock 15, 30, or 60 minutes in the future. The initial product
focus is the 30-minute horizon and honest probabilities rather than guaranteed inventory.

This repository is currently at **Milestone 0: repository and problem contract**. It contains
the Python package, development checks, typed GBFS and normalized snapshot contracts, synthetic
offline fixtures, and the documented time, target, attribution, and failure semantics. It does
not yet collect live data or produce forecasts.

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

The committed tests use only synthetic fixtures and require no network access. Milestone 1 will
add GBFS discovery, immutable raw capture, normalization, validation, and an idempotent station
snapshot ingestion command.

## Repository map

- `src/bikeshare_forecast/contracts/`: source and normalized data contracts
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
