from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from bikeshare_forecast.ingestion import SourceMetadata, import_toronto_ridership_2024

FIXTURE = (
    Path(__file__).parents[2] / "data" / "fixtures" / "historical" / "ridership_2024_official.csv"
)


def test_official_2024_import_streams_valid_rows_and_reports_rejections(tmp_path: Path) -> None:
    metadata = SourceMetadata(
        source_name="toronto-open-data",
        retrieved_at=datetime(2026, 9, 19, tzinfo=UTC),
    )
    first = import_toronto_ridership_2024(FIXTURE, tmp_path, metadata=metadata)
    second = import_toronto_ridership_2024(FIXTURE, tmp_path, metadata=metadata)
    frame = pl.read_parquet(first.parquet_path)

    assert first.rows_written == 1
    assert first.rejected_rows == 1
    assert second.rows_written == 0
    assert second.duplicates_ignored == 1
    assert frame["start_station_id"].to_list() == ["7001"]
    assert frame["end_station_id"].to_list() == ["7002"]
    assert str(frame.schema["started_at"]) == "Datetime(time_unit='us', time_zone='UTC')"
