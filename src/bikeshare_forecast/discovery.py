"""Print the feeds advertised by a configured GBFS discovery document."""

import argparse
import tomllib
from pathlib import Path
from typing import Any

from bikeshare_forecast.ingestion.gbfs import fetch_discovery


def load_gbfs_config(path: Path) -> dict[str, Any]:
    """Load the GBFS section from the collection configuration."""
    with path.open("rb") as config_file:
        config = tomllib.load(config_file)
    return dict(config["gbfs"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collection.toml"),
        help="Path to the collection configuration file",
    )
    args = parser.parse_args()

    config = load_gbfs_config(args.config)
    discovery = fetch_discovery(
        str(config["discovery_url"]),
        timeout_seconds=float(config["request_timeout_seconds"]),
    )

    print(f"GBFS version: {discovery.version}")
    for feed in discovery.data.feeds:
        print(f"{feed.name}: {feed.url}")


if __name__ == "__main__":
    main()
