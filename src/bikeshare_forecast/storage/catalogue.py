"""A small persistent DuckDB catalogue over append-only Parquet snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class DataSummary:
    row_count: int
    station_count: int
    first_source_at: datetime | None
    last_source_at: datetime | None
    duplicate_key_count: int


class DuckDBCatalogue:
    def __init__(self, database_path: Path, parquet_root: Path) -> None:
        self.database_path = database_path
        self.parquet_root = parquet_root

    def refresh(self) -> None:
        """Register the current immutable Parquet set as a queryable view."""
        files = sorted(self.parquet_root.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no Parquet snapshots in {self.parquet_root}")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        glob_path = str(self.parquet_root / "*.parquet").replace("'", "''")
        with duckdb.connect(str(self.database_path)) as connection:
            connection.execute(
                f"CREATE OR REPLACE VIEW station_snapshots AS "
                f"SELECT * FROM read_parquet('{glob_path}', union_by_name = true)"
            )

    def summary(self) -> DataSummary:
        self.refresh()
        with duckdb.connect(str(self.database_path), read_only=True) as connection:
            row = connection.execute(
                """
                SELECT
                    count(*) AS row_count,
                    count(DISTINCT station_id) AS station_count,
                    min(source_last_reported_at) AS first_source_at,
                    max(source_last_reported_at) AS last_source_at,
                    count(*) - count(DISTINCT (
                        station_id,
                        source_last_reported_at,
                        source_system_id
                    )) AS duplicate_key_count
                FROM station_snapshots
                """
            ).fetchone()
        if row is None:
            raise RuntimeError("DuckDB returned no summary row")
        return DataSummary(
            row_count=int(row[0]),
            station_count=int(row[1]),
            first_source_at=row[2],
            last_source_at=row[3],
            duplicate_key_count=int(row[4]),
        )

    def station_rows(self, station_id: str) -> list[tuple[object, ...]]:
        """Return ordered station observations through a parameterized query."""
        self.refresh()
        with duckdb.connect(str(self.database_path), read_only=True) as connection:
            return connection.execute(
                """
                SELECT station_id, source_last_reported_at, bikes_available, docks_available
                FROM station_snapshots
                WHERE station_id = ?
                ORDER BY source_last_reported_at
                """,
                [station_id],
            ).fetchall()
