# Offline ML pipeline

The ML pipeline reads only normalized station snapshot Parquet values. It does not fetch live
history and does not use ingestion timestamps as predictive information.

`bikeshare dataset-build` constructs one row per station and feature timestamp. Features contain
the current inventory, capacity, cyclic calendar values, and backward-looking 15- and 60-minute
inventory lags. A lag is nullable when no historical observation is available. For each 15-, 30-,
and 60-minute horizon, the label is the nearest station observation to the requested future time
within an inclusive five-minute tolerance. The earlier observation wins an exact tie, and a target
must be strictly later than its feature timestamp. These rules and all input hashes are written to
`dataset-manifest.json`.

The builder assigns global feature timestamps chronologically: the first 60% to training, the next
20% to validation, and the final 20% to test. A timestamp is never split between partitions.

`bikeshare train` fits a deterministic, standardized batch-gradient logistic model for the event
that zero bikes will be available at each horizon. It uses training rows only and writes portable
JSON models plus hashes in `model-manifest.json`.

`bikeshare evaluate` reports accuracy, precision, recall, F1, Brier score, log loss, and ROC AUC for
the logistic models on test rows. It also reports MAE and RMSE for numeric inventory persistence
and seven-day seasonal baselines. The seasonal baseline uses the same station's exact value seven
days earlier and falls back to persistence when that value is absent. The report records dataset
and model hashes.

Example:

```console
bikeshare dataset-build --silver-dir data/silver/station_snapshots --output-dir artifacts/dataset
bikeshare train --dataset-dir artifacts/dataset --output-dir artifacts/models
bikeshare evaluate --dataset-dir artifacts/dataset --model-dir artifacts/models --output-dir artifacts/evaluation
```
