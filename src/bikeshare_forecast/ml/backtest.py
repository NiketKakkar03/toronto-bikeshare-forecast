"""Expanding-window temporal backtests over a persisted dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from bikeshare_forecast.ml.common import read_json, sha256_file, write_json
from bikeshare_forecast.ml.evaluate import evaluate_run
from bikeshare_forecast.ml.train import train_models


def _mean_tree(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _mean_tree([value[key] for value in values]) for key in first}
    if isinstance(first, int | float):
        return sum(float(value) for value in values) / len(values)
    return first


def backtest_run(
    dataset_dir: Path,
    output_dir: Path,
    *,
    folds: int = 3,
    minimum_train_fraction: float = 0.5,
    evaluation_fraction: float = 0.1,
) -> Path:
    """Run deterministic expanding-window folds without touching the final source dataset."""
    if folds < 2:
        raise ValueError("at least two backtest folds are required")
    if not 0 < minimum_train_fraction < 1 or not 0 < evaluation_fraction < 1:
        raise ValueError("backtest fractions must be between zero and one")
    source_path = dataset_dir / "dataset.parquet"
    source_manifest = read_json(dataset_dir / "dataset-manifest.json")
    if source_manifest["dataset_sha256"] != sha256_file(source_path):
        raise ValueError("dataset hash does not match its manifest")
    frame = pl.read_parquet(source_path)
    times = sorted(frame["feature_time"].unique().to_list())
    evaluation_count = max(1, int(len(times) * evaluation_fraction))
    first_cutoff = max(1, int(len(times) * minimum_train_fraction))
    last_cutoff = len(times) - evaluation_count
    if first_cutoff >= last_cutoff:
        raise ValueError("dataset is too short for the requested backtest geometry")
    cutoffs = [
        round(first_cutoff + index * (last_cutoff - first_cutoff) / (folds - 1))
        for index in range(folds)
    ]
    reports: list[dict[str, Any]] = []
    for number, cutoff_index in enumerate(cutoffs, start=1):
        test_end_index = min(cutoff_index + evaluation_count, len(times))
        cutoff = times[cutoff_index]
        test_end = times[test_end_index] if test_end_index < len(times) else None
        in_test_window = pl.col("feature_time") >= cutoff
        if test_end is not None:
            in_test_window &= pl.col("feature_time") < test_end
        fold = frame.with_columns(
            pl.when(pl.col("feature_time") < cutoff)
            .then(pl.lit("train"))
            .when(in_test_window)
            .then(pl.lit("test"))
            .otherwise(pl.lit("unused"))
            .alias("split")
        )
        fold_root = output_dir / f"fold-{number:02d}"
        fold_dataset = fold_root / "dataset"
        fold_dataset.mkdir(parents=True, exist_ok=True)
        fold_path = fold_dataset / "dataset.parquet"
        fold.write_parquet(fold_path, compression="zstd")
        fold_manifest = {
            **source_manifest,
            "dataset_sha256": sha256_file(fold_path),
            "parent_dataset_sha256": source_manifest["dataset_sha256"],
            "backtest_fold": number,
            "backtest_train_end_exclusive": cutoff.isoformat(),
            "backtest_test_end_exclusive": test_end.isoformat() if test_end else None,
        }
        write_json(fold_dataset / "dataset-manifest.json", fold_manifest)
        model_dir = fold_root / "models"
        train_models(fold_dataset, model_dir)
        report = read_json(evaluate_run(fold_dataset, model_dir, fold_root / "evaluation"))
        reports.append(report)
    aggregate = {
        "schema_version": 1,
        "fold_count": folds,
        "parent_dataset_sha256": source_manifest["dataset_sha256"],
        "fold_reports": [
            f"fold-{number:02d}/evaluation/evaluation.json" for number in range(1, folds + 1)
        ],
        "mean_metrics": _mean_tree([report["metrics"] for report in reports]),
    }
    path = output_dir / "backtest.json"
    write_json(path, aggregate)
    return path
