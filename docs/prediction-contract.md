# Prediction contract

## Unit and time semantics

One prediction describes one station, one forecast creation time, and one horizon. Persisted
timestamps are timezone-aware UTC timestamps. User-facing Toronto times must be converted with
the `America/Toronto` IANA timezone, including daylight-saving transitions.

The initial supported horizons are 15, 30, and 60 minutes. End-to-end development and evaluation
start with 30 minutes. For a forecast created at time `t`, `target_time = t + horizon`.

Every input must have an availability time no later than `t`. Event time, source update time, and
ingestion time are distinct and must be retained. Late-arriving data cannot be included merely
because its event time precedes `t` if it was not available then.

## Targets and labels

The primary classification targets are:

- `empty_at_h`: zero rentable bikes at `target_time`
- `full_at_h`: zero available docks at `target_time`

Secondary count targets are rentable bikes and available docks at `target_time`. “Became empty
or full during the interval” is a different optional target and must never be substituted for
state at the target time.

Select the nearest valid station snapshot within ±3 minutes of `target_time`. An exact-distance
tie selects the earlier snapshot. If none exists, the example has no label. Do not interpolate a
classification label across a collection gap.

## Evaluation boundary

Observations are split in chronological blocks, with every station observation at a timestamp in
the same split. Model calibration uses a temporally later pre-test interval and never the final
test interval. Rolling-origin backtests are the release evidence; random observation-level splits
are prohibited.

## Serving and degradation

Every forecast will include data, feature, model, and calibration versions plus data freshness.
If the station feed exceeds the configured freshness limit, forecasts are suppressed. Fresh
current status remains available if the model is unavailable. A disabled station cannot be
recommended. Missing weather may use only a documented degraded feature path supported by the
approved model. The interface must never present a stale forecast as current.
