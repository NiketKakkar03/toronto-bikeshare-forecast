"""Streaming import for the official Bike Share Toronto 2024 CSV."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from bikeshare_forecast.contracts.historical import SourceQualitySummary
from bikeshare_forecast.ingestion.historical import SourceMetadata


@dataclass(frozen=True)
class BulkImportResult:
    parquet_path: Path
    quality_path: Path
    rows_written: int
    rejected_rows: int
    duplicates_ignored: int


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sql(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def import_toronto_ridership_2024(
    source: Path, output_dir: Path, *, metadata: SourceMetadata
) -> BulkImportResult:
    """Validate and stream the official 2024 source into canonical Parquet."""
    content_hash = _hash(source)
    parquet_dir = output_dir / "ridership" / "parquet"
    quality_dir = output_dir / "ridership" / "quality"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    destination = parquet_dir / f"part-{content_hash}.parquet"
    retrieved = metadata.retrieved_at.astimezone(UTC).isoformat()
    source_url = "NULL" if metadata.source_url is None else _sql(metadata.source_url)
    licence = "NULL" if metadata.licence is None else _sql(metadata.licence)
    csv_path = _sql(source.resolve())
    valid = (
        "Trip_Id IS NOT NULL AND Trip_Duration >= 0 AND Start_Station_Id IS NOT NULL "
        "AND Start_Time IS NOT NULL AND End_Station_Id IS NOT NULL AND End_Time IS NOT NULL "
        "AND End_Time >= Start_Time"
    )
    relation = f"read_csv_auto({csv_path}, header=true)"
    with duckdb.connect() as connection:
        counts = connection.execute(
            f"""SELECT count(*), count(*) FILTER (WHERE {valid}),
            count(DISTINCT Trip_Id) FILTER (WHERE {valid}), min(Start_Time), max(Start_Time),
            (SELECT count(DISTINCT station_id) FROM (
                SELECT cast(Start_Station_Id AS BIGINT) station_id FROM {relation} WHERE {valid}
                UNION SELECT cast(End_Station_Id AS BIGINT) station_id FROM {relation} WHERE {valid}
            ))
            FROM {relation}"""
        ).fetchone()
        if counts is None:
            raise ValueError("official ridership source returned no quality counts")
        total, accepted, unique_trips, first_time, last_time, stations = counts
        duplicates = int(accepted) - int(unique_trips)
        if duplicates:
            raise ValueError(f"official ridership source contains {duplicates} duplicate trip IDs")
        already_exists = destination.exists()
        if not already_exists:
            temporary = destination.with_suffix(".parquet.tmp")
            query = f"""COPY (
                SELECT
                    cast(Trip_Id AS VARCHAR) AS trip_id,
                    timezone('America/Toronto', Start_Time) AS started_at,
                    timezone('America/Toronto', End_Time) AS ended_at,
                    cast(Trip_Duration AS BIGINT) AS duration_seconds,
                    cast(cast(Start_Station_Id AS BIGINT) AS VARCHAR) AS start_station_id,
                    Start_Station_Name AS start_station_name,
                    cast(cast(End_Station_Id AS BIGINT) AS VARCHAR) AS end_station_id,
                    End_Station_Name AS end_station_name,
                    Bike_Model AS bike_type,
                    User_Type AS user_type,
                    {_sql(metadata.source_name)} AS source_name,
                    {_sql(source.name)} AS source_file,
                    row_number() OVER () + 1 AS source_row_number,
                    {_sql(content_hash)} AS source_content_hash,
                    'toronto-ridership-2024' AS source_schema_version,
                    '1.0.0' AS adapter_version,
                    cast({_sql(retrieved)} AS TIMESTAMPTZ) AS retrieved_at,
                    {source_url} AS source_url,
                    {licence} AS licence
                FROM {relation} WHERE {valid}
            ) TO {_sql(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"""
            connection.execute(query)
            os.replace(temporary, destination)
    summary = SourceQualitySummary(
        dataset="ridership",
        source_name=metadata.source_name,
        source_schema_version="toronto-ridership-2024",
        adapter_version="1.0.0",
        source_content_hash=content_hash,
        total_rows=int(total),
        accepted_rows=int(accepted),
        rejected_rows=int(total) - int(accepted),
        duplicate_rows=0,
        first_observed_at=(
            first_time.replace(tzinfo=ZoneInfo("America/Toronto")).astimezone(UTC)
            if first_time
            else None
        ),
        last_observed_at=(
            last_time.replace(tzinfo=ZoneInfo("America/Toronto")).astimezone(UTC)
            if last_time
            else None
        ),
        distinct_station_count=int(stations),
        missing_optional_value_count=0,
    )
    document = (summary.model_dump_json(indent=2) + "\n").encode()
    quality_path = quality_dir / f"{hashlib.sha256(document).hexdigest()}.json"
    quality_path.write_bytes(document)
    return BulkImportResult(
        parquet_path=destination,
        quality_path=quality_path,
        rows_written=0 if already_exists else int(accepted),
        rejected_rows=int(total) - int(accepted),
        duplicates_ignored=int(accepted) if already_exists else 0,
    )
