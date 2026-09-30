# Offline ML pipeline

The ML pipeline reads only normalized station snapshot Parquet values. It does not fetch live
history. Forecast creation time is the ingestion timestamp, and event timestamps are separately
retained so a late-arriving observation cannot appear before it was actually available.

`bikeshare dataset-build` constructs one row per station and forecast creation timestamp. Features
contain cyclic calendar values, station history, seasonal station history, completed demand lags,
Ontario holiday/rush-hour flags, and optional hourly weather. Every lag is measured strictly before
`feature_time`; each target counts departures, arrivals, and net flow over
`[feature_time, feature_time + horizon)`. These rules and all input hashes are written to
`dataset-manifest.json`.

The builder assigns global feature timestamps chronologically: the first 60% to training, the next
20% to validation, and the final 20% to test by default. A timestamp is never split between
partitions.

`bikeshare train` fits two model families from training rows only:

- deterministic standardized ridge regressors in portable JSON for lightweight serving
- histogram gradient boosting regressors for departure/arrival counts and classifiers for
  "any demand in the horizon" event probabilities

Each artifact is hashed in `model-manifest.json`. When a classifier sees a single class in training,
the trainer writes an explicit constant-probability degraded artifact instead of pretending a
discriminative model exists.

`bikeshare evaluate` reports count MAE/RMSE for ridge, histogram gradient boosting, and recent-rate
baselines on the held-out chronological test split. It also reports precision, recall, F1, ROC-AUC,
PR-AUC, Brier score, positive rate, and mean predicted probability for demand-event classifiers.
Prediction intervals are calibrated from validation residuals and checked on test rows. The report
records dataset and model hashes.

`bikeshare score-batch` writes a reproducible Parquet prediction table and manifest for a selected
chronological split. `bikeshare backtest` repeats training and evaluation over expanding
chronological windows and writes per-fold reports plus aggregate mean metrics.

Example:

```console
bikeshare dataset-build --historical-dir data/historical --output-dir artifacts/dataset
bikeshare train --dataset-dir artifacts/dataset --output-dir artifacts/models
bikeshare evaluate --dataset-dir artifacts/dataset --model-dir artifacts/models --output-dir artifacts/evaluation
bikeshare score-batch --dataset-dir artifacts/dataset --model-dir artifacts/models --output-dir artifacts/predictions
bikeshare backtest --dataset-dir artifacts/dataset --output-dir artifacts/backtest
```
