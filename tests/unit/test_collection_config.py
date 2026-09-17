from datetime import timedelta
from pathlib import Path

from bikeshare_forecast.config import load_collection_config


def test_loads_typed_collection_config() -> None:
    path = Path(__file__).parents[2] / "configs" / "collection.toml"

    config = load_collection_config(path)

    assert config.gbfs.max_attempts == 3
    assert config.gbfs.poll_interval_seconds == 300
    assert config.storage.raw_dir == Path("data/raw/gbfs")
    assert config.validation_policy.freshness_threshold == timedelta(minutes=10)
    assert config.validation_policy.latitude_min == 43.4
