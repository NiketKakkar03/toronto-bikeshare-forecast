"""Print the feeds advertised by a configured GBFS discovery document."""

import argparse
from pathlib import Path

from bikeshare_forecast.config import load_collection_config
from bikeshare_forecast.ingestion.gbfs import fetch_discovery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collection.toml"),
        help="Path to the collection configuration file",
    )
    args = parser.parse_args()

    config = load_collection_config(args.config).gbfs
    discovery = fetch_discovery(
        str(config.discovery_url),
        timeout_seconds=config.request_timeout_seconds,
        max_attempts=config.max_attempts,
        backoff_seconds=config.retry_backoff_seconds,
    )

    print(f"GBFS version: {discovery.version}")
    for feed in discovery.data.feeds:
        print(f"{feed.name}: {feed.url}")


if __name__ == "__main__":
    main()
