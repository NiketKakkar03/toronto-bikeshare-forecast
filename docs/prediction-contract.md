# Station-demand prediction contract

One prediction describes one station, one forecast creation time, and one horizon (15, 30, or 60 minutes).

## Targets

- `departures`: trips starting during `[forecast_created_at, target_time)`
- `arrivals`: trips ending during the same interval
- `net_flow`: arrivals minus departures
- `demand_pressure`: low, moderate, or high based on predicted net flow

Historical trips provide these labels directly. They do not reveal inventory, unmet demand, operator rebalancing, or disabled equipment. Demand predictions must never be described as guaranteed future bike or dock availability.

## Temporal and serving rules

Every feature must be known before forecast creation. Splits are chronological, recent-demand lags use completed intervals only, and evaluation reports MAE and RMSE against a recent-rate baseline.

Current bikes and docks come from one GBFS refresh when the application starts. They are displayed separately from historical demand forecasts. Stale current status suppresses forecasts. No continuous five-minute laptop collection is required.
