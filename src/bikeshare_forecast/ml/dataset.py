# ruff: noqa: E501
"""Scalable point-in-time station-demand dataset construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb

from bikeshare_forecast.ml.common import sha256_file, write_json

HORIZONS_MINUTES = (15, 30, 60)
FEATURE_COLUMNS = (
    "minute_sin",
    "minute_cos",
    "weekday_sin",
    "weekday_cos",
    "departures_lag_15m",
    "arrivals_lag_15m",
    "departures_lag_30m",
    "arrivals_lag_30m",
    "departures_lag_2h",
    "arrivals_lag_2h",
    "departures_lag_1d",
    "arrivals_lag_1d",
    "departures_lag_7d",
    "arrivals_lag_7d",
    "station_departures_avg",
    "station_arrivals_avg",
    "station_seasonal_departures_avg",
    "station_seasonal_arrivals_avg",
    "is_weekend",
    "is_ontario_holiday",
    "is_rush_hour",
    "temperature_c",
    "precipitation_mm",
    "wind_speed_kph",
    "relative_humidity_pct",
)

LAGS = (("15m", 15), ("30m", 30), ("2h", 120), ("1d", 1440), ("7d", 10080))


@dataclass(frozen=True)
class DatasetConfig:
    """Demand windows, inactive sampling, and chronological split policy."""

    horizons_minutes: tuple[int, ...] = HORIZONS_MINUTES
    interval_minutes: int = 15
    inactive_sample_rate: int = 20
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    sort_output: bool = True


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _sql(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _trip_files(root: Path) -> list[Path]:
    files = sorted((root / "ridership" / "parquet").glob("*.parquet"))
    if not files:
        raise ValueError(f"no historical ridership parquet files found in {root}")
    return files


def _weather_files(root: Path) -> list[Path]:
    return sorted((root / "weather" / "parquet").glob("*.parquet"))


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (occurrence - 1))


def _easter_sunday(year: int) -> date:
    """Return Gregorian Easter using the Meeus/Jones/Butcher algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    return date(year, month, (h + ell - 7 * m + 114) % 31 + 1)


def _ontario_holidays(first_year: int, last_year: int) -> list[date]:
    holidays: list[date] = []
    for year in range(first_year, last_year + 1):
        victoria_day = date(year, 5, 24)
        while victoria_day.weekday() != 0:
            victoria_day -= timedelta(days=1)
        holidays.extend(
            (
                date(year, 1, 1),
                _nth_weekday(year, 2, 0, 3),  # Family Day
                _easter_sunday(year) - timedelta(days=2),  # Good Friday
                victoria_day,
                date(year, 7, 1),
                _nth_weekday(year, 8, 0, 1),  # Civic Holiday
                _nth_weekday(year, 9, 0, 1),  # Labour Day
                _nth_weekday(year, 10, 0, 2),  # Thanksgiving
                date(year, 12, 25),
                date(year, 12, 26),
            )
        )
    return holidays


def build_dataset(
    historical_dir: Path, output_dir: Path, config: DatasetConfig | None = None
) -> Path:
    """Stream trip aggregation and sampled station-time features into Parquet."""
    policy = config or DatasetConfig()
    if policy.interval_minutes <= 0 or any(
        h % policy.interval_minutes for h in policy.horizons_minutes
    ):
        raise ValueError("interval must be positive and divide every horizon")
    if policy.inactive_sample_rate < 1:
        raise ValueError("inactive_sample_rate must be positive")
    if not 0 < policy.train_fraction < 1 or not 0 < policy.validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if policy.train_fraction + policy.validation_fraction >= 1:
        raise ValueError("split fractions must leave a test partition")
    files = _trip_files(historical_dir)
    weather_files = _weather_files(historical_dir)
    sources = ", ".join(_sql_path(path) for path in files)
    interval = policy.interval_minutes
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.parquet"
    temporary = dataset_path.with_suffix(".parquet.tmp")
    with duckdb.connect() as connection:
        connection.execute(f"CREATE TEMP VIEW trips AS SELECT * FROM read_parquet([{sources}])")
        connection.execute(f"""CREATE TEMP TABLE counts AS
            SELECT station_id, time_bucket(INTERVAL '{interval} minutes', event_time) feature_time,
                   sum(departures)::INTEGER departures, sum(arrivals)::INTEGER arrivals,
                   any_value(station_name) station_name
            FROM (
                SELECT start_station_id station_id, started_at event_time,
                       start_station_name station_name, 1 departures, 0 arrivals FROM trips
                UNION ALL
                SELECT end_station_id, ended_at, end_station_name, 0, 1 FROM trips
            ) GROUP BY station_id, feature_time""")
        bounds = connection.execute(
            "SELECT min(feature_time), max(feature_time) FROM counts"
        ).fetchone()
        if bounds is None:
            raise ValueError("historical ridership contains no event-time bounds")
        first, last = bounds
        if first is None or last is None:
            raise ValueError("historical ridership contains no usable event times")
        train_cutoff = first + (last - first) * policy.train_fraction
        validation_cutoff = first + (last - first) * (
            policy.train_fraction + policy.validation_fraction
        )
        local_years = connection.execute(
            "SELECT year(timezone('America/Toronto', ?)), year(timezone('America/Toronto', ?))",
            [first, last],
        ).fetchone()
        if local_years is None:
            raise ValueError("historical ridership contains no local calendar bounds")
        holiday_values = ", ".join(
            f"(DATE {_sql(day.isoformat())})"
            for day in _ontario_holidays(int(local_years[0]), int(local_years[1]))
        )
        connection.execute(
            f"CREATE TEMP TABLE ontario_holidays(day DATE); INSERT INTO ontario_holidays VALUES {holiday_values}"
        )
        if weather_files:
            weather_sources = ", ".join(_sql_path(path) for path in weather_files)
            connection.execute(f"""CREATE TEMP VIEW hourly_weather AS
                SELECT time_bucket(INTERVAL '1 hour', observed_at) weather_hour,
                       avg(temperature_c) temperature_c,
                       max(precipitation_mm) precipitation_mm,
                       avg(wind_speed_kph) wind_speed_kph,
                       avg(relative_humidity_pct) relative_humidity_pct
                FROM read_parquet([{weather_sources}])
                GROUP BY weather_hour""")
        else:
            connection.execute("""CREATE TEMP VIEW hourly_weather AS
                SELECT NULL::TIMESTAMPTZ weather_hour, NULL::DOUBLE temperature_c,
                       NULL::DOUBLE precipitation_mm, NULL::DOUBLE wind_speed_kph,
                       NULL::DOUBLE relative_humidity_pct WHERE false""")
        joins: list[str] = []
        targets: list[str] = []
        for horizon in policy.horizons_minutes:
            pieces = horizon // interval
            for offset in range(pieces):
                alias = f"h{horizon}_{offset}"
                joins.append(
                    f"LEFT JOIN counts {alias} ON {alias}.station_id=g.station_id AND {alias}.feature_time=g.feature_time + INTERVAL '{offset * interval} minutes'"
                )
            departures = " + ".join(f"coalesce(h{horizon}_{i}.departures,0)" for i in range(pieces))
            arrivals = " + ".join(f"coalesce(h{horizon}_{i}.arrivals,0)" for i in range(pieces))
            targets.extend(
                (
                    f"g.feature_time + INTERVAL '{horizon} minutes' AS target_time_{horizon}m",
                    f"({departures})::INTEGER AS target_departures_{horizon}m",
                    f"({arrivals})::INTEGER AS target_arrivals_{horizon}m",
                    f"(({arrivals}) - ({departures}))::INTEGER AS target_net_flow_{horizon}m",
                )
            )
        lag_joins = " ".join(
            f"LEFT JOIN counts l_{name} ON l_{name}.station_id=g.station_id "
            f"AND l_{name}.feature_time=g.feature_time-INTERVAL '{minutes} minutes'"
            for name, minutes in LAGS
        )
        lag_columns = ",\n                ".join(
            f"coalesce(l_{name}.departures,0)::INTEGER departures_lag_{name},\n"
            f"                coalesce(l_{name}.arrivals,0)::INTEGER arrivals_lag_{name}"
            for name, _ in LAGS
        )
        output_order = "ORDER BY g.feature_time, g.station_id" if policy.sort_output else ""
        query = f"""COPY (
            WITH stations AS (
                SELECT station_id, any_value(station_name) station_name,
                       min(feature_time) first_feature_time,
                       max(feature_time) last_feature_time
                FROM counts GROUP BY station_id
            ),
            grid AS (
                SELECT s.station_id, s.station_name, gs feature_time
                FROM stations s,
                     generate_series(
                         s.first_feature_time,
                         least(s.last_feature_time, cast({_sql(last)} AS TIMESTAMPTZ) - INTERVAL '{max(policy.horizons_minutes)} minutes'),
                         INTERVAL '{interval} minutes'
                     ) t(gs)
            ), history_grid AS (
                SELECT g.*,
                       extract(dow FROM timezone('America/Toronto', g.feature_time)) local_weekday,
                       extract(hour FROM timezone('America/Toronto', g.feature_time))*60
                           + extract(minute FROM timezone('America/Toronto', g.feature_time)) local_minute,
                       CASE WHEN g.feature_time < cast({_sql(train_cutoff)} AS TIMESTAMPTZ)
                            THEN coalesce(c.departures,0) END history_departures,
                       CASE WHEN g.feature_time < cast({_sql(train_cutoff)} AS TIMESTAMPTZ)
                            THEN coalesce(c.arrivals,0) END history_arrivals
                FROM grid g LEFT JOIN counts c USING (station_id, feature_time)
            ), featured_grid AS (
                SELECT *,
                       avg(history_departures) OVER station_history station_departures_avg,
                       avg(history_arrivals) OVER station_history station_arrivals_avg,
                       avg(history_departures) OVER seasonal_history station_seasonal_departures_avg,
                       avg(history_arrivals) OVER seasonal_history station_seasonal_arrivals_avg
                FROM history_grid
                WINDOW station_history AS (
                           PARTITION BY station_id ORDER BY feature_time
                           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                       seasonal_history AS (
                           PARTITION BY station_id, local_weekday, local_minute ORDER BY feature_time
                           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            )
            SELECT g.station_id, g.station_name, g.feature_time,
                sin(2*pi()*(extract(hour FROM timezone('America/Toronto', g.feature_time))*60+extract(minute FROM timezone('America/Toronto', g.feature_time)))/1440) minute_sin,
                cos(2*pi()*(extract(hour FROM timezone('America/Toronto', g.feature_time))*60+extract(minute FROM timezone('America/Toronto', g.feature_time)))/1440) minute_cos,
                sin(2*pi()*extract(dow FROM timezone('America/Toronto', g.feature_time))/7) weekday_sin,
                cos(2*pi()*extract(dow FROM timezone('America/Toronto', g.feature_time))/7) weekday_cos,
                {lag_columns},
                coalesce(g.station_departures_avg,0) station_departures_avg,
                coalesce(g.station_arrivals_avg,0) station_arrivals_avg,
                coalesce(g.station_seasonal_departures_avg,g.station_departures_avg,0) station_seasonal_departures_avg,
                coalesce(g.station_seasonal_arrivals_avg,g.station_arrivals_avg,0) station_seasonal_arrivals_avg,
                (extract(isodow FROM timezone('America/Toronto', g.feature_time)) >= 6)::INTEGER is_weekend,
                (oh.day IS NOT NULL)::INTEGER is_ontario_holiday,
                (extract(isodow FROM timezone('America/Toronto', g.feature_time)) <= 5 AND
                 (extract(hour FROM timezone('America/Toronto', g.feature_time)) BETWEEN 7 AND 9 OR
                  extract(hour FROM timezone('America/Toronto', g.feature_time)) BETWEEN 16 AND 18))::INTEGER is_rush_hour,
                coalesce(w.temperature_c,0) temperature_c,
                coalesce(w.precipitation_mm,0) precipitation_mm,
                coalesce(w.wind_speed_kph,0) wind_speed_kph,
                coalesce(w.relative_humidity_pct,0) relative_humidity_pct,
                {", ".join(targets)},
                CASE WHEN g.feature_time < cast({_sql(train_cutoff)} AS TIMESTAMPTZ) THEN 'train'
                     WHEN g.feature_time < cast({_sql(validation_cutoff)} AS TIMESTAMPTZ) THEN 'validation'
                     ELSE 'test' END split
            FROM featured_grid g
            {lag_joins}
            LEFT JOIN ontario_holidays oh ON oh.day=cast(timezone('America/Toronto', g.feature_time) AS DATE)
            LEFT JOIN hourly_weather w ON w.weather_hour=time_bucket(INTERVAL '1 hour', g.feature_time)
            {" ".join(joins)}
            WHERE coalesce(l_15m.departures,0)+coalesce(l_15m.arrivals,0)+coalesce(l_2h.departures,0)+coalesce(l_2h.arrivals,0) > 0
               OR hash(g.station_id, g.feature_time) % {policy.inactive_sample_rate} = 0
            {output_order}
        ) TO {_sql_path(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"""
        connection.execute(query)
        split_counts = connection.execute(
            f"SELECT count(*), count(*) FILTER (split='train'), count(*) FILTER (split='validation'), count(*) FILTER (split='test') FROM read_parquet({_sql_path(temporary)})"
        ).fetchone()
        if split_counts is None:
            raise ValueError("demand dataset contains no split counts")
        count, train, validation, test = split_counts
    temporary.replace(dataset_path)
    manifest = {
        "schema_version": 3,
        "forecast_kind": "station_demand",
        "config": asdict(policy),
        "feature_columns": list(FEATURE_COLUMNS),
        "targets": ["departures", "arrivals", "net_flow"],
        "row_count": count,
        "split_counts": {"train": train, "validation": validation, "test": test},
        "source_files": [{"path": str(path), "sha256": sha256_file(path)} for path in files],
        "weather_source_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in weather_files
        ],
        "dataset_sha256": sha256_file(dataset_path),
        "point_in_time_rule": "lags use completed trip buckets before feature_time; seasonal averages use only the training-time history; weather observations are joined at their normalized hour",
        "target_rule": "counts trips in [feature_time, feature_time + horizon)",
        "sampling_rule": "retain rows with recent activity plus deterministic inactive grid sample",
        "split_rule": "global timestamps are split chronologically",
    }
    write_json(output_dir / "dataset-manifest.json", manifest)
    return dataset_path
