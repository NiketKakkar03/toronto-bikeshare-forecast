import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest

from bikeshare_forecast.ingestion import collect_once
from bikeshare_forecast.ml import DatasetConfig, build_dataset, evaluate_run, train_models
from bikeshare_forecast.ml.common import read_json

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


def test_source_to_evaluation_pipeline_is_fully_offline_and_repeatable(tmp_path: Path) -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    discovery = _payload("gbfs.json")
    information = _payload("station_information.json")
    information["data"]["stations"][0]["capacity"] = 4

    for step in range(40):
        event_time = start + timedelta(minutes=15 * step)
        bikes = step % 5
        status = _payload("station_status.json")
        station = status["data"]["stations"][0]
        station["num_vehicles_available"] = bikes
        station["num_docks_available"] = 4 - bikes
        station["last_reported"] = event_time.isoformat().replace("+00:00", "Z")
        status["last_updated"] = event_time.isoformat().replace("+00:00", "Z")

        def handler(request: httpx.Request, status_payload: object = status) -> httpx.Response:
            if request.url.path.endswith("gbfs.json"):
                return httpx.Response(200, json=discovery)
            if request.url.path.endswith("station_information.json"):
                return httpx.Response(200, json=information)
            return httpx.Response(200, json=status_payload)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            collect_once(
                discovery_url="https://example.invalid/gbfs/v3.0/gbfs.json",
                raw_dir=tmp_path / "raw",
                silver_dir=tmp_path / "silver",
                retrieved_at=event_time + timedelta(seconds=10),
                client=client,
            )

    dataset = tmp_path / "dataset"
    models = tmp_path / "models"
    build_dataset(
        tmp_path / "silver",
        dataset,
        DatasetConfig(horizons_minutes=(15,)),
    )
    train_models(dataset, models)
    report = read_json(evaluate_run(dataset, models, tmp_path / "evaluation"))

    assert report["rows"] > 0
    assert set(report["metrics"]) == {"15m"}
    assert (models / "logistic-empty-15m.json").exists()
    assert (models / "logistic-full-15m.json").exists()
    assert len(list((tmp_path / "silver").glob("*.parquet"))) == 40
