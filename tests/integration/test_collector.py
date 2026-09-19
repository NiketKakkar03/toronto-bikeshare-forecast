import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from bikeshare_forecast.ingestion import collect_once

FIXTURES = Path(__file__).parents[2] / "data" / "fixtures" / "gbfs"


def _payload(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _client(
    *, fail_status_once: bool = False, missing_status: bool = False
) -> tuple[httpx.Client, list[str]]:
    calls: list[str] = []
    failures = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal failures
        calls.append(request.url.path)
        if request.url.path.endswith("gbfs.json"):
            discovery = _payload("gbfs.json")
            if missing_status:
                discovery["data"]["feeds"] = [discovery["data"]["feeds"][0]]
            return httpx.Response(200, json=discovery)
        if request.url.path.endswith("station_information.json"):
            return httpx.Response(200, json=_payload("station_information.json"))
        if request.url.path.endswith("station_status.json"):
            if fail_status_once and failures == 0:
                failures += 1
                return httpx.Response(503, json={"error": "temporary"})
            return httpx.Response(200, json=_payload("station_status.json"))
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_collection_runs_end_to_end_retries_and_is_idempotent(tmp_path: Path) -> None:
    client, calls = _client(fail_status_once=True)
    kwargs = {
        "discovery_url": "https://example.invalid/gbfs/v3.0/gbfs.json",
        "raw_dir": tmp_path / "raw",
        "silver_dir": tmp_path / "silver",
        "retrieved_at": datetime(2026, 9, 16, 12, tzinfo=UTC),
        "client": client,
    }

    first = collect_once(**kwargs)
    second = collect_once(**kwargs)

    assert first.snapshots_written == 1
    assert first.raw_objects_created == 2
    assert second.snapshots_written == 0
    assert second.duplicates_ignored == 1
    assert second.raw_objects_created == 0
    assert calls.count("/gbfs/v3.0/station_status.json") == 3
    frame = pl.read_parquet(next((tmp_path / "silver").glob("*.parquet")))
    assert frame.row(0, named=True)["station_id"] == "7001"
    assert len(list((tmp_path / "raw").rglob("*.manifest.json"))) == 2
    client.close()


def test_collection_rejects_discovery_without_required_feed_before_writes(tmp_path: Path) -> None:
    client, _ = _client(missing_status=True)
    with pytest.raises(ValueError, match="missing required feeds"):
        collect_once(
            discovery_url="https://example.invalid/gbfs/v3.0/gbfs.json",
            raw_dir=tmp_path / "raw",
            silver_dir=tmp_path / "silver",
            retrieved_at=datetime(2026, 9, 16, 12, tzinfo=UTC),
            client=client,
        )
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "silver").exists()
    client.close()
