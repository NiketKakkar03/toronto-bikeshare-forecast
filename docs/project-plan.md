# Toronto Bike Share Availability Forecasting Platform

## High-level overview

### What are we building?

We will build a public application that predicts whether a Toronto Bike Share station will have a bike or an open dock when a rider expects to arrive. The initial forecast horizons will be 15, 30, and 60 minutes.

A user will be able to:

1. View current station inventory on a map.
2. Select a station and an arrival time.
3. See the predicted number of bikes and docks.
4. See the probability that the station will be empty or full.
5. Find nearby alternatives when the selected station is risky.
6. See when the data and forecast were last updated.

### How does the system work?

```text
Toronto live station feed ─┐
Historical trip data ──────┼──> ingestion and validation ──> versioned data
Weather data ──────────────┘                                  │
                                                              v
                                                    point-in-time features
                                                              │
                                  ┌───────────────────────────┴────────────────┐
                                  v                                            v
                          model training                                live feature builder
                                  │                                            │
                                  v                                            v
                         temporal backtesting ──> approved model ──> forecast service
                                                                            │
                                              ┌─────────────────────────────┼──────────────┐
                                              v                             v              v
                                         web map                      batch forecast   monitoring
```

### What will we learn?

The project covers the workflow used in applied machine-learning and ML-engineering roles:

- Automated ingestion of real, changing data
- Schema validation and reproducible datasets
- Time-aware feature engineering without future leakage
- Regression and classification baselines
- Gradient-boosted models and probability calibration
- Temporal backtesting and model comparison
- Real-time and batch inference
- API design and an interactive user interface
- Experiment and model tracking
- Data, model, and service monitoring
- CI, testing, deployment, and rollback

### What are the major stages?

| Stage | Outcome |
|---|---|
| 1. Data foundation | We automatically collect and validate station snapshots. |
| 2. Baseline | We measure whether simple persistence and historical-average forecasts work. |
| 3. ML model | We train and calibrate a point-in-time-correct boosted-tree model. |
| 4. Product | A map and API provide live forecasts and alternatives. |
| 5. Operations | We monitor data freshness, forecast quality, latency, and drift. |
| 6. Advanced work | We test sequence models, orchestration, and larger-scale processing only when justified. |

### What is the first deliverable?

The first deliverable is not a model. It is a reliable collector:

```bash
uv run bikeshare ingest-stations
uv run bikeshare validate-data
uv run bikeshare data-summary
```

It must safely collect repeated station snapshots, reject invalid records, prevent duplicates, and produce a coverage report. This creates the historical labels required for forecasting.

---

## 1. Project summary

Build a production-style machine-learning platform for forecasting short-term Toronto Bike Share station availability. It will ingest live station information, historical ridership, weather, and calendar information; construct point-in-time-correct features; train and evaluate forecasting models; expose live and batch predictions; and monitor both model and service health.

The project is intended to become a genuinely useful public application. It is not affiliated with or endorsed by the City of Toronto, the Toronto Parking Authority, or Bike Share Toronto. Predictions must be presented as estimates, not guarantees.

## 2. User problem

A rider often needs to answer one of two questions:

- Will a bike be available when I reach my departure station?
- Will an open dock be available when I reach my destination station?

Current inventory alone may be insufficient because station conditions can change between checking the application and arriving. The product therefore forecasts near-future inventory and directs the rider to nearby alternatives when the selected station is likely to be unusable.

### Initial user story

> As a Toronto Bike Share rider, I want to select a station and my expected arrival time so that I can assess whether a bike or dock is likely to be available and identify a nearby alternative.

### Initial product boundary

The system will:

- Display public station information.
- Produce short-horizon station-level forecasts.
- Express uncertainty and data freshness.
- Suggest nearby stations based on availability and distance.

The system will not initially:

- Plan complete cycling routes.
- Guarantee inventory.
- Reserve bicycles or docks.
- Process payments or Bike Share accounts.
- Collect user location histories or other personal information.
- Claim to be an official Bike Share Toronto product.

## 3. Prediction problem

### Prediction unit

One station at one forecast creation time and one forecast horizon.

Example:

```text
station_id: 7001
forecast_created_at: 2026-09-16T08:00:00-04:00
horizon_minutes: 30
target_time: 2026-09-16T08:30:00-04:00
```

### Primary classification targets

For each horizon `h`:

- `empty_at_h`: station has zero rentable bikes at the target time.
- `full_at_h`: station has zero available docks at the target time.

An optional operational target is whether a station becomes empty or full at any point during the interval. This is a separate label and must not be confused with state at the target time.

### Secondary regression targets

- `bikes_available_at_h`
- `docks_available_at_h`

Counts must be constrained to plausible values relative to station capacity. Classification is the initial product priority because the risk of finding no bike or dock is directly actionable.

### Forecast horizons

Start with 15, 30, and 60 minutes. Build and evaluate the 30-minute horizon first. Add other horizons only after the full 30-minute pipeline works.

### Prediction-time semantics

Every feature must be available no later than `forecast_created_at`. A target is derived from the first valid snapshot within a documented tolerance around `target_time`.

Example label rule:

```text
Use the nearest valid snapshot within ±3 minutes of target_time.
If no snapshot exists in that interval, mark the example as lacking a label.
Never interpolate a classification label across a long collection gap.
```

All persisted timestamps should be UTC. The user interface may display `America/Toronto`, with daylight-saving transitions handled by an IANA timezone library.

## 4. Success criteria

### User success

- A user can find a station and request a forecast in a few interactions.
- Current availability and forecast freshness are unambiguous.
- Risky stations have useful nearby alternatives.
- The application remains useful when the model is temporarily unavailable by showing current status.

### ML success

The trained model must be compared with:

1. Persistence: future inventory equals current inventory.
2. Seasonal historical average: expected inventory for the station, weekday, and time bucket.
3. A simple logistic-regression classification baseline.

The boosted model is accepted only if it improves the selected primary metric over relevant baselines across multiple temporal test windows—not merely on one random split.

### Engineering success

- One command constructs a deterministic dataset from an immutable raw snapshot range.
- Training and evaluation are reproducible from a configuration file.
- Online and batch predictions agree for identical inputs.
- Every forecast includes data, feature, model, and calibration versions.
- CI executes tests, linting, type checks, and a small end-to-end pipeline.

### Operational success

- Feed staleness and collection gaps are detected.
- Prediction latency and failures are observable.
- Forecast quality is calculated after future snapshots make labels available.
- An approved previous model can be restored.

## 5. Data sources

### 5.1 Live station information and status

Use the public Bike Share Toronto station feed discovered through the City of Toronto catalogue or the official GBFS system catalogue. GBFS separates relatively static station metadata from frequently changing station status.

Expected inputs include:

- Station identifier and name
- Latitude and longitude
- Station capacity
- Number of rentable bikes
- Number of available docks
- Bicycle-type counts when provided
- Station operational status
- Feed generation or update time

The ingestion implementation must discover feed URLs from the GBFS root document rather than permanently assuming that every endpoint path will remain unchanged.

References:

- [City of Toronto Open Data](https://open.toronto.ca/)
- [GBFS documentation](https://gbfs.org/get-started/)
- [GBFS project and schemas](https://github.com/MobilityData/gbfs)
- [Bike Share Toronto GBFS feed catalogue entry](https://mobilitydatabase.org/feeds/gbfs/gbfs-bike_share_toronto)

The current GBFS v3 auto-discovery endpoint is:

```text
https://toronto.publicbikesystem.net/customer/gbfs/v3.0/gbfs.json
```

The application must begin with this discovery document and read the advertised URLs for `station_information` and `station_status`. Do not hard-code guessed child paths.

### 5.2 Historical Bike Share ridership

Use the Bike Share Toronto ridership releases available through the City of Toronto Open Data catalogue. Historical trip records can provide station flows and demand patterns, but they do not replace archived station-status snapshots.

Potential inputs include:

- Trip start and end time
- Start and end station
- Trip duration
- Bicycle type
- User type, if consistently available and appropriate

The pipeline must tolerate schema differences across release periods through versioned source adapters.

### 5.3 Weather

Use Environment and Climate Change Canada data for historical observations and, later, forecast weather available at prediction time.

Candidate variables:

- Temperature
- Precipitation
- Snow
- Wind speed
- Relative humidity
- Weather condition

References:

- [ECCC historical climate data](https://climate.meteo.gc.ca/)
- [MSC GeoMet API](https://api.weather.gc.ca/?f=html)

Observed weather may be used for training only when its observation time precedes the prediction time. A live system forecasting future availability must use weather forecasts or most-recent observations, not future observed weather.

### 5.4 Calendar data

Generate locally:

- Hour and minute bucket
- Day of week
- Weekend indicator
- Month and season
- Ontario statutory holiday indicator
- Rush-hour indicator
- Daylight-saving-time state

Public events and transit disruptions are possible later extensions, not initial dependencies.

### 5.5 Licensing and attribution

The repository and application must retain attribution and link to the relevant source terms. Do not assume that every source uses the same licence.

The current Mobility Database catalogue entry for the Bike Share Toronto GBFS feed identifies its licence as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). That licence permits sharing and adaptation, including commercial use, provided appropriate credit is given, a link to the licence is included, and changes are indicated. Record the feed's licence metadata during implementation in case the publisher changes it.

Datasets obtained from the City of Toronto Open Data catalogue use the Open Government Licence – Toronto unless the dataset page states otherwise. It permits lawful reuse, including commercial reuse, subject to attribution and the other licence conditions.

Include this attribution where City data is displayed:

> Contains information licensed under the Open Government Licence – Toronto.

References:

- [Bike Share Toronto feed catalogue and current licence](https://mobilitydatabase.org/feeds/gbfs/gbfs-bike_share_toronto)
- [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- [Open Government Licence – Toronto](https://open.toronto.ca/open-data-licence/)

Verify the licence and source metadata when each dataset is first ingested and record them in the data card. Do not use City names or logos in a way that suggests endorsement.

## 6. Data collection design

### Snapshot frequency

Begin with a five-minute interval. Store the actual source timestamp and ingestion timestamp. A shorter interval can be considered after measuring feed behavior, storage requirements, and rate expectations.

### Raw data policy

- Raw responses are immutable.
- Each response receives a content hash.
- A manifest records source URL, retrieval time, feed time, status, schema version, hash, and row count.
- Retries use bounded exponential backoff.
- Identical source records are deduplicated.
- Collection failures are logged rather than silently filled.
- A raw file is never modified after successful validation.

### Layered storage

```text
Bronze: original source responses plus ingestion metadata
Silver: validated, normalized station snapshots and weather observations
Gold: point-in-time feature tables and labeled modeling datasets
```

Use Parquet for durable analytical files and DuckDB for local querying. Object storage can replace the local filesystem after deployment without changing the logical data contracts.

### Snapshot key

A normalized station observation should be unique on a key such as:

```text
(station_id, source_last_reported_at, source_system_id)
```

The ingestion timestamp is not sufficient for deduplication because identical feed states may be collected repeatedly.

### Initial station snapshot schema

```text
station_id                   string
station_name                 string
latitude                     float64
longitude                    float64
capacity                     int32
bikes_available              int32
docks_available              int32
is_installed                 boolean
is_renting                   boolean
is_returning                 boolean
source_last_reported_at      timestamp[UTC]
ingested_at                  timestamp[UTC]
source_system_id             string
source_schema_version        string
raw_content_hash             string
```

Preserve source-specific values in bronze data even when the silver schema omits them.

### Core validation rules

- Required fields exist and have expected types.
- Station identifiers are non-empty.
- Coordinates are plausible for the Toronto region.
- Counts are non-negative.
- `bikes_available + docks_available` is checked against capacity with documented tolerances.
- Source timestamps are not implausibly far in the future.
- Feed age is below the configured freshness threshold.
- Station metadata changes are detected and versioned.
- Duplicates are rejected or idempotently ignored.
- Missing-station and collection-coverage rates are measured.

## 7. Dataset construction and leakage prevention

### Point-in-time rule

For a forecast created at time `t`, every input must have an availability time less than or equal to `t`. Event time and ingestion time must both be retained so that late data cannot silently enter historical features.

### Training example construction

For each eligible station snapshot at time `t`:

1. Select the station state known at `t`.
2. Compute lag and rolling features using observations no later than `t`.
3. Join calendar data for `t`.
4. Join weather known by `t`.
5. Set `target_time = t + horizon`.
6. Locate a valid target snapshot within the label tolerance.
7. Construct classification and count labels.
8. Record all dataset and feature versions.

### Dataset manifest

Every generated dataset must include:

- Dataset identifier
- Creation timestamp
- Code commit
- Configuration hash
- Input manifest hashes
- Source time range
- Station count
- Row count
- Feature specification version
- Label specification version
- Exclusion and missing-label counts
- Output file hashes

### Temporal splits

Do not randomly split individual observations. Use chronological blocks.

Initial pattern:

```text
Training: earliest 70% of eligible time
Validation: following 15%
Test: latest 15%
```

The final evaluation should use rolling-origin backtests with multiple cutoffs. Keep all examples at one timestamp in the same split.

### Leakage tests

Automated tests must verify that:

- No feature timestamp exceeds prediction time.
- Target snapshots occur after prediction time.
- Rolling windows exclude future rows.
- Weather observations or forecasts respect publication time.
- Preprocessing is fit on training data only.
- Calibration is not fit on the final test period.

## 8. Feature plan

### Current-state features

- Current bikes and docks
- Bikes and docks as a fraction of capacity
- Station capacity
- Operational flags
- Mechanical versus electric bicycle counts, if available

### Recent-history features

- Inventory lags at 5, 10, 15, 30, and 60 minutes
- Inventory change over recent intervals
- Rolling mean, minimum, maximum, and standard deviation
- Number of transitions into empty or full state
- Time since last empty or full state
- Missing-observation and snapshot-age indicators

### Seasonal features

- Minute-of-day represented cyclically
- Day-of-week represented categorically or cyclically
- Weekend and holiday indicators
- Month and season
- Station-by-time historical averages computed only from prior data

### Spatial features

- Station coordinates
- Capacity
- Nearby station count and aggregate availability
- Distance to the closest stations
- Neighborhood or spatial cluster
- Net flow estimates from historical trips

Nearby-station features must also be point-in-time correct.

### Weather features

- Temperature and apparent temperature
- Precipitation or snow indicators
- Wind speed
- Humidity
- Weather condition
- Forecast-versus-observation marker
- Weather age

### Feature registry

For every feature, document:

- Name and data type
- Business meaning
- Source
- Transformation
- Event time and availability-time rule
- Missing-value behavior
- Valid range
- Training and serving implementation
- Owner or responsible module

## 9. Modeling strategy

### Baseline 1: persistence

Predict that future inventory equals current inventory. Derive empty/full classifications from the predicted count.

This is a strong short-horizon baseline and must not be omitted.

### Baseline 2: seasonal historical average

Predict from the station’s prior behavior for comparable weekday and time buckets. Apply smoothing for stations with little history.

### Baseline 3: logistic regression

Train separate or multi-output logistic models for empty and full events. This provides an interpretable probabilistic baseline.

### Primary model

Use LightGBM initially. Train classification models for empty/full risk and optionally count-regression models.

Investigate:

- Class weighting
- Station identifier treatment
- Categorical variables
- Missing-history indicators
- Monotonic or output constraints only when logically justified
- Feature ablations
- Conservative hyperparameter tuning

### Calibration

Evaluate uncalibrated probabilities first. Fit Platt scaling or isotonic calibration on a temporally later calibration set. Never calibrate on the final test set.

Reliability matters because a displayed “70% empty risk” should correspond approximately to the observed event frequency for comparable forecasts.

### Optional sequence model

After the tabular system is complete, test a small GRU, temporal convolutional network, or another modest sequence model over recent station histories. It must be compared against tabular lag and rolling features under the same backtest.

Do not add a Transformer without evidence that the data volume and dependencies justify it.

## 10. Evaluation

### Classification metrics

- Precision-recall AUC
- ROC-AUC as a secondary measure
- Log loss
- Brier score
- Calibration curve and expected calibration error
- Precision and recall for empty/full events
- Recall at a fixed alert budget

### Regression metrics

- Mean absolute error
- Root mean squared error
- Mean absolute scaled error relative to persistence
- Error by forecast horizon

### Product metrics

- Percentage of unusable-station events warned about
- False-alarm rate
- Alternative-station success rate
- Forecast coverage
- Percentage of forecasts suppressed because data is stale

### Segment analysis

Report performance by:

- Forecast horizon
- Station
- Capacity band
- Time of day
- Weekday versus weekend
- Season
- Weather condition
- Initially busy versus quiet station

### Backtesting

Use rolling-origin evaluation:

1. Select a historical cutoff.
2. Fit on information before the cutoff.
3. Calibrate using a later but pre-test interval.
4. Forecast the next evaluation period.
5. Allow target labels to mature.
6. Measure forecast, calibration, product, and segment metrics.
7. Advance the cutoff and repeat.

The model should not be accepted solely because aggregate performance improves. Investigate whether improvement is consistent across time and station groups.

## 11. Product behavior

### Forecast response

```json
{
  "station_id": "7001",
  "forecast_created_at": "2026-09-16T12:00:00Z",
  "target_time": "2026-09-16T12:30:00Z",
  "current_bikes": 4,
  "current_docks": 11,
  "predicted_bikes": 1.8,
  "predicted_docks": 13.2,
  "empty_probability": 0.63,
  "full_probability": 0.01,
  "risk_level": "high",
  "data_freshness_seconds": 74,
  "model_version": "availability-lgbm-0.3.0",
  "feature_version": "station-features-v2",
  "calibration_version": "empty-iso-v1",
  "warnings": []
}
```

### Forecast versus recommendation

Keep these separate:

- **Forecast:** estimates future station inventory and probabilities.
- **Recommendation policy:** converts forecasts, walking distance, freshness, and user intent into a suggested station.

Policy thresholds should be versioned independently of the model.

### Failure behavior

- If live data is stale, show its timestamp prominently and suppress forecasts past a configured limit.
- If the model is unavailable, show current availability without a forecast.
- If weather is unavailable, use a documented degraded feature path if the model supports it.
- If a station is disabled, do not recommend it.
- If uncertainty is excessive, communicate low confidence rather than manufacturing precision.

## 12. Architecture

### Initial local architecture

```text
Scheduled collector
      │
      v
Raw JSON + manifests
      │
      v
Validation and normalization
      │
      v
Parquet datasets + DuckDB catalogue
      │
      ├──> feature builder ──> training/backtesting ──> MLflow artifacts
      │
      └──> live feature builder ──> FastAPI ──> web application
                                             │
                                             v
                                    forecasts and telemetry
```

### Recommended initial stack

| Concern | Technology | Reason |
|---|---|---|
| Language | Python 3.12 | Strong ML and data ecosystem |
| Environment | `uv` | Fast, reproducible dependency management |
| Dataframes | Polars | Efficient transformations and lazy execution |
| Local analytics | DuckDB | Query Parquet directly and build reproducible datasets |
| Storage | JSON and Parquet | Preserve raw input and efficient analytical tables |
| Schemas | Pydantic and typed Polars schemas | API and ingestion validation |
| Baselines | scikit-learn | Logistic regression, calibration, and metrics |
| Main model | LightGBM | Strong tabular baseline with efficient training |
| Deep learning | PyTorch, later | Optional sequence modeling |
| API | FastAPI | Typed prediction interface and generated documentation |
| Application | React map or Streamlit initially | User-facing station exploration |
| Tracking | MLflow | Experiments, artifacts, and model registry |
| Testing | pytest | Unit, integration, contract, and end-to-end tests |
| Quality | Ruff and mypy | Formatting, linting, and type checking |
| CI | GitHub Actions | Automated verification |
| Packaging | Docker | Reproducible deployment |
| Load testing | Locust or k6, later | Service performance validation |

Choose one tool per concern. Do not introduce distributed computing, Kubernetes, or a message broker until measurements demonstrate a need.

## 13. Suggested repository structure

```text
toronto-bikeshare-forecast/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── Makefile
├── .env.example
├── .gitignore
├── docker-compose.yml
├── configs/
│   ├── collection.yaml
│   ├── features.yaml
│   ├── training.yaml
│   └── policy.yaml
├── data/
│   ├── raw/                 # ignored locally; immutable source captures
│   ├── silver/              # ignored; normalized tables
│   ├── gold/                # ignored; modeling datasets
│   └── fixtures/            # small synthetic/test examples committed to Git
├── docs/
│   ├── architecture.md
│   ├── data-card.md
│   ├── model-card.md
│   ├── feature-registry.md
│   ├── evaluation-report.md
│   ├── operations.md
│   └── incident-review.md
├── notebooks/
│   └── exploration.ipynb
├── src/bikeshare_forecast/
│   ├── cli.py
│   ├── config.py
│   ├── contracts/
│   ├── ingestion/
│   │   ├── gbfs.py
│   │   ├── ridership.py
│   │   └── weather.py
│   ├── validation/
│   ├── storage/
│   ├── datasets/
│   ├── features/
│   ├── models/
│   ├── calibration/
│   ├── evaluation/
│   ├── policy/
│   ├── serving/
│   ├── monitoring/
│   └── workflows/
├── web/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   └── end_to_end/
├── scripts/
└── .github/workflows/
```

Notebooks may call package functions, but production transformations must live under `src/` and be tested.

## 14. Command-line interface

Target a coherent interface such as:

```bash
uv run bikeshare ingest-stations
uv run bikeshare ingest-weather
uv run bikeshare ingest-ridership --year 2025
uv run bikeshare validate-data
uv run bikeshare build-dataset --start 2026-09-01 --end 2026-10-01 --horizon 30
uv run bikeshare train --config configs/training.yaml
uv run bikeshare backtest --config configs/training.yaml
uv run bikeshare batch-forecast
uv run bikeshare serve
uv run bikeshare monitor
```

Commands must return nonzero exit codes on failure, emit structured logs, and print identifiers for created datasets, runs, or artifacts.

## 15. Testing strategy

### Unit tests

- Feed parsing
- Timestamp normalization and daylight-saving transitions
- Deduplication
- Validation rules
- Lag and rolling-window boundaries
- Label matching tolerance
- Baseline predictions
- Calibration
- Recommendation policy
- Distance calculations

### Data tests

- Key uniqueness
- Non-negative inventory
- Plausible capacity relationships
- Monotonic manifest coverage
- Source freshness
- Label occurs after prediction time
- No future data in features

### Integration tests

- Saved GBFS fixture to normalized Parquet
- Raw snapshots to labeled dataset
- Dataset to trained artifact
- Approved artifact to batch predictions
- Feature builder to API response

### Contract tests

- GBFS adapter expectations
- Request and response models
- Feature specification compatibility
- Model, calibration, and policy metadata requirements

### End-to-end test

Using small committed fixtures:

1. Ingest snapshots.
2. Build a small dataset.
3. Train a tiny baseline model.
4. Generate batch predictions.
5. Start or invoke the API.
6. Assert that online and batch forecasts agree.

Live network calls should not be required for the standard test suite. A separate scheduled smoke test may verify current external feeds.

## 16. Experiment tracking and model registry

Every training run should record:

- Code commit
- Dataset and feature versions
- Configuration
- Time ranges and split boundaries
- Model type and parameters
- Metrics by horizon and important segment
- Calibration metrics
- Feature importance or diagnostic explanations
- Serialized preprocessing, model, and calibration artifacts
- Environment and dependency information

### Promotion gates

A challenger may be approved only when:

- Required tests pass.
- Dataset and feature lineage are complete.
- It beats or matches required baselines on primary metrics.
- Calibration is within an accepted tolerance.
- No critical station group shows unexplained severe degradation.
- Latency and artifact size meet service limits.
- A model card is updated.

Promotion should require an explicit command or approval. Retraining must not automatically imply deployment.

## 17. Serving and batch inference

### API endpoints

Initial endpoints may include:

```text
GET  /health
GET  /ready
GET  /stations
GET  /stations/{station_id}/status
GET  /stations/{station_id}/forecast?horizon_minutes=30
GET  /stations/{station_id}/alternatives?horizon_minutes=30
POST /batch/forecast
GET  /model/info
```

### Parity requirement

Batch and online scoring must call the same preprocessing and prediction components. Golden fixtures should verify identical outputs within a documented floating-point tolerance.

### Structured prediction log

Record:

- Request or forecast identifier
- Forecast creation and target time
- Station and horizon
- Feature/data versions
- Model and calibration versions
- Predictions and recommendation-policy result
- Data age and warnings
- Latency and status

Do not log unnecessary user identifiers or precise user location.

## 18. Monitoring

### Data health

- Feed freshness
- Successful collection rate
- Missing stations
- Schema changes
- Invalid ranges
- Station metadata changes
- Snapshot gaps
- Weather availability

### Model health before labels arrive

- Forecast distribution
- Empty/full alert rate
- Feature missingness
- Feature and prediction drift
- Out-of-range feature rates
- Difference between persistence and model forecasts

### Model health after labels arrive

- Brier score and log loss
- Calibration
- PR-AUC
- Empty/full recall and false-alarm rate
- Count MAE
- Performance by horizon, station group, and time period

### Service health

- Request rate
- Error rate
- P50, P95, and P99 latency
- Timeout rate
- Model load failures
- Batch completion time
- Current serving model and last successful refresh

### Alerts

Every alert needs a threshold, time window, severity, owner, diagnostic link or command, and response procedure. Initial alerts should prioritize stale data, stopped collection, invalid schema, forecast failure, and elevated service errors.

## 19. Deployment and reliability

### Deployment stages

1. Local command-line pipeline
2. Local API and web interface
3. Public read-only demonstration
4. Scheduled collection and forecasts
5. Monitored deployment with rollback

### Safe degradation

The application must distinguish:

- Current status is fresh and forecast is available.
- Current status is fresh but forecast is unavailable.
- Current status is stale.
- Station is not operational.

Current status can remain useful when forecast infrastructure fails. The UI must not show an old forecast as if it were current.

### Rollback demonstration

Maintain at least one previous approved model. Simulate a challenger with degraded calibration or segment performance, demonstrate detection, and restore the approved model without rebuilding the application.

## 20. Milestone plan

### Milestone 0: Repository and problem contract

- Create the repository and Python package.
- Add development tooling and CI skeleton.
- Document prediction time, targets, horizons, and label tolerance.
- Record data-source licences and attribution.
- Create synthetic GBFS fixtures.

**Exit criterion:** The project installs, tests run in CI, and the prediction contract is unambiguous.

### Milestone 1: Live data collector

- Discover and ingest station metadata and status.
- Save immutable raw responses and manifests.
- Normalize snapshots to Parquet.
- Add schema, range, freshness, and duplicate validation.
- Produce collection-coverage reports.
- Schedule collection at a conservative interval.

**Exit criterion:** The collector runs unattended for seven days with measured coverage and no silent data loss.

### Milestone 2: Historical and weather data

- Add versioned historical ridership adapters.
- Ingest historical weather observations.
- Normalize timestamps and station identifiers.
- Create source-quality summaries.

**Exit criterion:** One command builds documented silver tables from raw source inputs.

### Milestone 3: Labels and simple baselines

- Construct 30-minute labels.
- Implement persistence and seasonal-average forecasts.
- Define temporal train, validation, calibration, and test periods.
- Create the first backtest report.

**Exit criterion:** A reproducible temporal evaluation establishes honest performance floors.

### Milestone 4: Feature pipeline and boosted model

- Add point-in-time current, lag, rolling, calendar, spatial, and weather features.
- Train logistic regression and LightGBM.
- Calibrate empty/full probabilities.
- Perform feature ablations and segment analysis.

**Exit criterion:** A written comparison explains whether and where the boosted model beats each baseline.

### Milestone 5: Reproducible lifecycle

- Add dataset manifests and hashes.
- Track experiments and artifacts in MLflow.
- Add model metadata and registry gates.
- Turn training and backtesting into tested workflows.

**Exit criterion:** A workflow produces a traceable dataset, model, calibration artifact, evaluation, and registry decision.

### Milestone 6: API and batch forecasts

- Implement typed API contracts.
- Generate scheduled forecasts for all active stations.
- Separate forecasting from recommendation policy.
- Add structured logs and golden prediction fixtures.

**Exit criterion:** Online and batch predictions agree for identical inputs.

### Milestone 7: Public application

- Build a station map.
- Add search, forecast horizon, freshness, confidence, and alternatives.
- Make the interface mobile-friendly and accessible.
- Display attribution and non-endorsement language.

**Exit criterion:** A user can make a real station decision without understanding the underlying ML system.

### Milestone 8: Monitoring and controlled deployment

- Monitor data, model, and service health.
- Score matured labels automatically.
- Add champion/challenger reports and optional shadow scoring.
- Simulate degradation and rollback.

**Exit criterion:** A failure or degradation is detected, communicated, and recoverable.

### Milestone 9: Advanced modeling and scale

- Add 15- and 60-minute horizons.
- Test a sequence model against tabular history features.
- Load-test the API.
- Optimize or parallelize demonstrated bottlenecks.

**Exit criterion:** Additional complexity is accepted or rejected using measured evidence.

### Milestone 10: Portfolio release

- Complete README, architecture diagram, data card, model card, operations guide, and incident review.
- Record a concise demonstration video.
- Publish a technical article.
- Confirm that a new user can reproduce the local workflow.

**Exit criterion:** The repository, application, evidence, and limitations are understandable without private explanation.

## 21. Immediate first steps

After creating the new directory and GitHub repository:

1. Copy this file into `docs/project-plan.md`.
2. Create the repository skeleton through `src/`, `tests/`, `configs/`, `data/fixtures/`, and `.github/workflows/`.
3. Initialize a Python 3.12 project with `uv`.
4. Add Ruff, mypy, pytest, Polars, DuckDB, Pydantic, HTTPX, and Typer.
5. Add `.gitignore` entries for raw/processed data, MLflow state, secrets, and local databases.
6. Create saved station-information and station-status fixtures.
7. Define typed source and normalized schemas.
8. Implement a read-only GBFS discovery client.
9. Implement one snapshot ingestion command.
10. Test parsing, validation, idempotency, and stale-feed handling.
11. Run the collector manually and inspect the first normalized snapshot.
12. Schedule it only after the local command is reliable.

Do not begin model training until enough valid snapshot history has accumulated and the label-building logic is tested.

## 22. Documentation deliverables

- Public README with quick start and screenshots
- Architecture diagram
- Data card with source, licence, coverage, and limitations
- Feature registry with availability-time rules
- Model card with intended use and limitations
- Baseline and temporal backtesting report
- Calibration report
- API documentation
- Monitoring and operations guide
- Simulated incident and post-incident review
- Two-to-three-minute demonstration video
- Technical article about leakage, calibration, or delayed labels

## 23. Definition of done

The portfolio release is complete when:

- The product solves a clear rider problem.
- Live and historical data acquisition is automated.
- Raw sources and derived datasets are versioned and traceable.
- Time and label semantics are explicit.
- Feature construction is point-in-time correct.
- Persistence, seasonal, logistic, and boosted-tree approaches are compared.
- Probabilities are calibrated and evaluated.
- Temporal backtesting is reproducible.
- Batch and online predictions agree.
- Forecast and recommendation policy are separate.
- The public interface clearly displays freshness and uncertainty.
- Monitoring covers data, model, and service health.
- Failure modes and rollback are demonstrated.
- Core contracts and pipelines have automated tests.
- CI passes on the public repository.
- Data attribution and non-endorsement language are present.
- A new developer can run the local demonstration from the README.

## 24. Risks and limitations

### Source availability

External feeds can change, become stale, or disappear. Source adapters, fixtures, schema checks, retry limits, and cached current status reduce—but do not eliminate—this dependency.

### Historical snapshot availability

Historical trip records do not fully reconstruct station inventory. The project must collect live snapshots over time before rigorous station-state forecasting is possible.

### Operational intervention

Bike redistribution, station maintenance, temporary closures, and capacity changes may be difficult to predict from public data.

### Unobserved demand

The system observes realized inventory and trips, not every rider who wanted a bicycle or dock but could not obtain one.

### Forecast uncertainty

Short-term station behavior can be volatile. The product must communicate uncertainty and offer alternatives instead of implying certainty.

### Dataset changes

Station identifiers, names, capacity, and source schemas can change. Metadata must be treated as versioned data rather than permanent constants.

### Public-service boundaries

The project is an independent forecast tool and not an authoritative source for station operations. It must not imply City or Bike Share endorsement.

## 25. Interview discussion topics

Be prepared to explain:

- Why random row splitting would overestimate performance
- The difference between event time, source update time, and ingestion time
- How target snapshots are selected
- How every feature avoids future information
- Why persistence is a demanding short-horizon baseline
- Why PR-AUC, calibration, and product metrics complement ROC-AUC
- How weather observations can cause leakage
- How delayed labels affect monitoring
- Why current status remains separate from the forecast
- How online and offline transformations remain consistent
- How a challenger is approved or rejected
- What happens when the feed, weather service, or model fails
- Why a sequence model may not justify its complexity
- Which bottleneck would motivate distributed computation
- How the design would change for multiple cities

## 26. Potential resume bullet

> Built and deployed a Toronto Bike Share availability forecasting platform using live GBFS, ridership, and weather data; implemented point-in-time feature pipelines, calibrated short-horizon forecasts, temporal backtesting, batch and API inference, data-quality monitoring, and controlled model promotion.

Do not use this bullet until the described capabilities actually exist. Replace general claims with measured results—for example, forecast coverage, improvement over persistence, calibration error, API latency, and collection reliability.

## 27. Final success statement

The strongest version of this project is not the one with the most sophisticated model. It is the one that reliably transforms changing public data into honest, timely, measurable forecasts that help a rider make a better station choice. Its value comes from combining temporal correctness, useful probability estimates, resilient software, and clear communication.
