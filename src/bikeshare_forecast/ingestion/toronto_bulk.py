"""Streaming import for official Bike Share Toronto ridership CSVs."""

from __future__ import annotations

import codecs
import hashlib
import os
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import duckdb

from bikeshare_forecast.contracts.historical import SourceQualitySummary
from bikeshare_forecast.ingestion.historical import SourceMetadata


@dataclass(frozen=True)
class BulkImportResult:
    parquet_path: Path
    quality_path: Path
    accepted_rows: int
    rows_written: int
    rejected_rows: int
    duplicates_ignored: int


OfficialRidershipSchema = Literal[
    "toronto-ridership-2022",
    "toronto-ridership-2023",
    "toronto-ridership-2024",
    "toronto-ridership-2025",
]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sql(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _hash_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


@contextmanager
def _readable_csv(source: Path) -> Iterator[tuple[Path, str, str]]:
    if source.suffix.lower() != ".zip":
        content_hash = _hash(source)
        if _is_utf8(source):
            yield source, source.name, content_hash
            return
        temporary_path = _transcode_to_utf8(source)
        try:
            yield temporary_path, source.name, content_hash
        finally:
            temporary_path.unlink(missing_ok=True)
        return
    with zipfile.ZipFile(source) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(
                f"expected exactly one CSV member in {source}, found {len(csv_members)}"
            )
        member = csv_members[0]
        contents = archive.read(member)
    with tempfile.NamedTemporaryFile(
        prefix=f"{source.stem}-", suffix=".csv", delete=False
    ) as temporary:
        temporary.write(contents)
        temporary_path = Path(temporary.name)
    try:
        yield temporary_path, f"{source.name}!{member}", _hash_bytes(contents)
    finally:
        temporary_path.unlink(missing_ok=True)


def _is_utf8(source: Path) -> bool:
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    try:
        with source.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                decoder.decode(block)
            decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def _transcode_to_utf8(source: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f"{source.stem}-utf8-", suffix=".csv", delete=False, mode="w", encoding="utf-8"
    ) as output:
        with source.open("r", encoding="cp1252", errors="replace", newline="") as input_file:
            for block in iter(lambda: input_file.read(1024 * 1024), ""):
                output.write(block)
        return Path(output.name)


def _schema_from_source(source: Path) -> OfficialRidershipSchema:
    name = source.name.lower()
    if "2022" in name:
        return "toronto-ridership-2022"
    if "2023" in name:
        return "toronto-ridership-2023"
    if "2024" in name:
        return "toronto-ridership-2024"
    if "2025" in name:
        return "toronto-ridership-2025"
    raise ValueError(f"cannot infer official Toronto ridership schema from {source.name}")


def import_toronto_ridership_official(
    source: Path,
    output_dir: Path,
    *,
    metadata: SourceMetadata,
    schema: OfficialRidershipSchema | None = None,
) -> BulkImportResult:
    """Validate and stream an official source into canonical historical Parquet."""
    source_schema_version = schema or _schema_from_source(source)
    with _readable_csv(source) as (csv_source, source_file, content_hash):
        return _import_readable_csv(
            csv_source,
            source_file,
            content_hash,
            output_dir,
            metadata=metadata,
            source_schema_version=source_schema_version,
        )


def import_toronto_ridership_2024(
    source: Path, output_dir: Path, *, metadata: SourceMetadata
) -> BulkImportResult:
    """Validate and stream the official 2024 source into canonical Parquet."""
    return import_toronto_ridership_official(
        source,
        output_dir,
        metadata=metadata,
        schema="toronto-ridership-2024",
    )


def _import_readable_csv(
    source: Path,
    source_file: str,
    content_hash: str,
    output_dir: Path,
    *,
    metadata: SourceMetadata,
    source_schema_version: OfficialRidershipSchema,
) -> BulkImportResult:
    parquet_dir = output_dir / "ridership" / "parquet"
    quality_dir = output_dir / "ridership" / "quality"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    destination = parquet_dir / f"part-{content_hash}.parquet"
    retrieved = metadata.retrieved_at.astimezone(UTC).isoformat()
    source_url = "NULL" if metadata.source_url is None else _sql(metadata.source_url)
    licence = "NULL" if metadata.licence is None else _sql(metadata.licence)
    csv_path = _sql(source.resolve())
    relation = f"read_csv_auto({csv_path}, header=true, all_varchar=true)"
    if source_schema_version in {"toronto-ridership-2022", "toronto-ridership-2023"}:
        trip_id = '"Trip Id"'
        duration = 'try_cast("Trip  Duration" AS BIGINT)'
        start_station_id = 'try_cast("Start Station Id" AS BIGINT)'
        start_station_name = '"Start Station Name"'
        start_time = "try_strptime(\"Start Time\", '%m/%d/%Y %H:%M')"
        end_station_id = 'try_cast("End Station Id" AS BIGINT)'
        end_station_name = '"End Station Name"'
        end_time = "try_strptime(\"End Time\", '%m/%d/%Y %H:%M')"
        bike_type = "NULL"
        user_type = '"User Type"'
    else:
        trip_id = "Trip_Id"
        duration = "try_cast(Trip_Duration AS BIGINT)"
        start_station_id = "try_cast(Start_Station_Id AS BIGINT)"
        start_station_name = "Start_Station_Name"
        start_time = "try_strptime(Start_Time, '%Y-%m-%d %H:%M:%S')"
        end_station_id = "try_cast(End_Station_Id AS BIGINT)"
        end_station_name = "End_Station_Name"
        end_time = "try_strptime(End_Time, '%Y-%m-%d %H:%M:%S')"
        bike_type = "Bike_Model"
        user_type = "User_Type"
    normalized = f"""(
        SELECT
            {trip_id} raw_trip_id,
            {duration} raw_duration,
            {start_station_id} raw_start_station_id,
            {start_time} raw_start_time,
            {start_station_name} raw_start_station_name,
            {end_station_id} raw_end_station_id,
            {end_time} raw_end_time,
            {end_station_name} raw_end_station_name,
            {bike_type} raw_bike_type,
            {user_type} raw_user_type
        FROM {relation}
    )"""
    valid = (
        "raw_trip_id IS NOT NULL AND raw_duration >= 0 AND raw_start_station_id IS NOT NULL "
        "AND raw_start_time IS NOT NULL AND raw_end_station_id IS NOT NULL "
        "AND raw_end_time IS NOT NULL AND raw_end_time >= raw_start_time"
    )
    with duckdb.connect() as connection:
        counts = connection.execute(
            f"""SELECT count(*), count(*) FILTER (WHERE {valid}),
            count(DISTINCT raw_trip_id) FILTER (WHERE {valid}),
            min(raw_start_time), max(raw_start_time),
            (SELECT count(DISTINCT station_id) FROM (
                SELECT raw_start_station_id station_id FROM {normalized} WHERE {valid}
                UNION SELECT raw_end_station_id station_id FROM {normalized} WHERE {valid}
            ))
            FROM {normalized}"""
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
                    cast(raw_trip_id AS VARCHAR) AS trip_id,
                    timezone('America/Toronto', raw_start_time) AS started_at,
                    timezone('America/Toronto', raw_end_time) AS ended_at,
                    cast(raw_duration AS BIGINT) AS duration_seconds,
                    cast(raw_start_station_id AS VARCHAR) AS start_station_id,
                    nullif(raw_start_station_name, 'NULL') AS start_station_name,
                    cast(raw_end_station_id AS VARCHAR) AS end_station_id,
                    nullif(raw_end_station_name, 'NULL') AS end_station_name,
                    raw_bike_type AS bike_type,
                    raw_user_type AS user_type,
                    {_sql(metadata.source_name)} AS source_name,
                    {_sql(source_file)} AS source_file,
                    row_number() OVER () + 1 AS source_row_number,
                    {_sql(content_hash)} AS source_content_hash,
                    {_sql(source_schema_version)} AS source_schema_version,
                    '1.0.0' AS adapter_version,
                    cast({_sql(retrieved)} AS TIMESTAMPTZ) AS retrieved_at,
                    {source_url} AS source_url,
                    {licence} AS licence
                FROM {normalized} WHERE {valid}
            ) TO {_sql(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"""
            connection.execute(query)
            os.replace(temporary, destination)
    summary = SourceQualitySummary(
        dataset="ridership",
        source_name=metadata.source_name,
        source_schema_version=source_schema_version,
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
        accepted_rows=int(accepted),
        rows_written=0 if already_exists else int(accepted),
        rejected_rows=int(total) - int(accepted),
        duplicates_ignored=int(accepted) if already_exists else 0,
    )
