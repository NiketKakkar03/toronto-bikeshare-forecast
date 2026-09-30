# Model benchmark

This benchmark uses the City of Toronto's official
[Bike Share ridership release](https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/)
for trips whose start and end timestamps fall between January 1 and March 31, 2024. The import
accepted 778,928 trips, rejected 268 invalid rows, and found no duplicate trip IDs. Dataset
construction produced 1,861,094 point-in-time station rows across 797 stations at 15-minute
intervals.

The fixed holdout uses a global chronological 60/20/20 train/validation/test split. Its test
partition contains 390,808 rows. The robustness check uses three expanding-window folds with a
10% evaluation window. No random split is used. Weather was not included in this run, so weather
features follow the pipeline's documented zero-fill behavior.

## Expanding-window results

The table reports the mean across the three backtest folds. Improvement is the MAE reduction
relative to extrapolating demand from the most recent 15-minute bucket.

| Horizon | Target | HistGB MAE | HistGB RMSE | Baseline MAE | MAE improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| 15 min | Departures | 0.370 | 0.599 | 0.501 | 26.1% |
| 15 min | Arrivals | 0.368 | 0.603 | 0.492 | 25.2% |
| 30 min | Departures | 0.566 | 0.879 | 0.951 | 40.5% |
| 30 min | Arrivals | 0.564 | 0.894 | 0.938 | 39.9% |
| 60 min | Departures | 0.856 | 1.325 | 1.844 | 53.6% |
| 60 min | Arrivals | 0.859 | 1.370 | 1.826 | 53.0% |

Demand-event classification asks whether at least one departure or arrival will occur within the
forecast horizon.

| Horizon | Target | ROC-AUC | PR-AUC | Brier | Precision at 0.5 | Recall at 0.5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 15 min | Departures | 0.788 | 0.515 | 0.138 | 0.623 | 0.249 |
| 15 min | Arrivals | 0.790 | 0.514 | 0.137 | 0.621 | 0.243 |
| 30 min | Departures | 0.807 | 0.689 | 0.167 | 0.674 | 0.537 |
| 30 min | Arrivals | 0.810 | 0.687 | 0.165 | 0.673 | 0.538 |
| 60 min | Departures | 0.838 | 0.834 | 0.164 | 0.756 | 0.753 |
| 60 min | Arrivals | 0.838 | 0.831 | 0.164 | 0.753 | 0.754 |

On the final fixed holdout, histogram gradient boosting achieved departure/arrival MAE of
0.367/0.365 at 15 minutes, 0.561/0.560 at 30 minutes, and 0.850/0.853 at 60 minutes. The
validation-calibrated 80% intervals covered 80.7-80.9% of holdout outcomes, while 90% intervals
covered 90.6-90.8%.

## Interpretation and limitations

MAE is measured in station trips per forecast horizon. For example, a 60-minute MAE of 0.856
means the departure forecast differs from the observed count by 0.856 trips per station-time row
on average. Longer horizons are better at identifying whether any activity will occur, but exact
count error increases because there is more potential activity to predict.

These models forecast demand flow, not guaranteed future inventory. A station can still become
empty or full because current inventory, dock capacity, rebalancing, and trips already in progress
are not labels in the historical ridership release. The benchmark covers winter and early spring,
so a production promotion decision should also require a full-year backtest with weather and
seasonal coverage.

Reproduce the pipeline after importing the same public source slice:

```bash
uv run bikeshare dataset-build --historical-dir data/historical-q1
uv run bikeshare train
uv run bikeshare evaluate
uv run bikeshare score-batch
uv run bikeshare backtest
```

Generated datasets, models, predictions, and raw evaluation JSON remain ignored because they are
large reproducible artifacts. Every persisted artifact includes SHA-256 lineage tying it to the
source dataset and trained models.
