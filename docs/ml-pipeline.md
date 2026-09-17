# Offline ML pipeline

The ML pipeline reads only normalized station snapshot Parquet values. It does not fetch live
history. Forecast creation time is the ingestion timestamp, and event timestamps are separately
retained so a late-arriving observation cannot appear before it was actually available.

`bikeshare dataset-build` constructs one row per station and forecast creation timestamp. Features contain
the current inventory, capacity, cyclic calendar values, and backward-looking 15- and 60-minute
inventory lags. A lag is nullable when no historical observation is available. For each 15-, 30-,
and 60-minute horizon, the label is the nearest station observation to the requested future time
within an inclusive three-minute tolerance. The earlier observation wins an exact tie, and a target
must be strictly later than its feature timestamp. These rules and all input hashes are written to
`dataset-manifest.json`.

The builder assigns global feature timestamps chronologically: the first 70% to training, the next
15% to validation, and the final 15% to test. A timestamp is never split between partitions. These
values and label thresholds are versioned in `configs/training.toml`.

`bikeshare train` fits deterministic, standardized batch-gradient logistic models for both zero
bikes and zero docks at each horizon. It uses training rows only and writes portable JSON models
plus hashes in `model-manifest.json`.

`bikeshare evaluate` reports accuracy, precision, recall, F1, precision-recall AUC, ROC AUC, Brier
score, log loss, and expected calibration error for the empty and full models on test rows. It also
reports MAE, RMSE, and MASE for bike and dock inventory persistence and seasonal baselines. The
seasonal baseline is fit only on training rows by station, weekday, and 15-minute bucket, with a
persistence fallback. The report records dataset and model hashes. `bikeshare backtest` repeats
this evaluation over configured expanding chronological windows.

Example:

```console
bikeshare dataset-build --silver-dir data/silver/station_snapshots --output-dir artifacts/dataset
bikeshare train --dataset-dir artifacts/dataset --output-dir artifacts/models
bikeshare evaluate --dataset-dir artifacts/dataset --model-dir artifacts/models --output-dir artifacts/evaluation
bikeshare backtest --dataset-dir artifacts/dataset --output-dir artifacts/backtest
```
