"""Scalable deterministic ridge-regression station-demand training."""

from __future__ import annotations

from pathlib import Path

import duckdb

from bikeshare_forecast.ml.common import read_json, sha256_file, write_json
from bikeshare_forecast.ml.dataset import FEATURE_COLUMNS


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense linear system with partial-pivot Gaussian elimination."""
    size = len(vector)
    augmented = [[*row, vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("ridge normal equation is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[index][-1] for index in range(size)]


def _sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def train_models(dataset_dir: Path, output_dir: Path) -> Path:
    """Train departure and arrival ridge models from streaming sufficient statistics."""
    dataset_path = dataset_dir / "dataset.parquet"
    manifest = read_json(dataset_dir / "dataset-manifest.json")
    expected_hash = manifest.get("dataset_sha256")
    if expected_hash != sha256_file(dataset_path):
        raise ValueError("dataset hash does not match its manifest")
    source = f"read_parquet({_sql_path(dataset_path)})"
    columns = list(FEATURE_COLUMNS)
    with duckdb.connect() as connection:
        row_count = connection.execute(
            f"SELECT count(*) FROM {source} WHERE split='train'"
        ).fetchone()
        if row_count is None:
            raise ValueError("training row count is unavailable")
        training_rows = int(row_count[0])
        if training_rows == 0:
            raise ValueError("training split is empty")
        statistic_sql = ", ".join(f"avg({column}), stddev_pop({column})" for column in columns)
        statistics = connection.execute(
            f"SELECT {statistic_sql} FROM {source} WHERE split='train'"
        ).fetchone()
        if statistics is None:
            raise ValueError("training feature statistics are unavailable")
        means = [float(statistics[index * 2]) for index in range(len(columns))]
        scales = [max(float(statistics[index * 2 + 1]), 1e-12) for index in range(len(columns))]
        standardized = [
            f"(({column}-({means[index]}))/{scales[index]})" for index, column in enumerate(columns)
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, object]] = []
        for horizon_value in manifest["config"]["horizons_minutes"]:
            horizon = int(horizon_value)
            for target in ("departures", "arrivals"):
                label = f"target_{target}_{horizon}m"
                expressions = ["count(*)"]
                expressions.extend(f"sum({value})" for value in standardized)
                for left in range(len(columns)):
                    expressions.extend(
                        f"sum({standardized[left]}*{standardized[right]})"
                        for right in range(left, len(columns))
                    )
                expressions.append(f"sum({label})")
                expressions.extend(f"sum({value}*{label})" for value in standardized)
                values = connection.execute(
                    f"SELECT {', '.join(expressions)} FROM {source} WHERE split='train'"
                ).fetchone()
                if values is None:
                    raise ValueError(f"training statistics unavailable for {target} {horizon}m")
                dimension = len(columns) + 1
                gram = [[0.0] * dimension for _ in range(dimension)]
                gram[0][0] = float(values[0])
                cursor = 1
                for index in range(len(columns)):
                    gram[0][index + 1] = gram[index + 1][0] = float(values[cursor])
                    cursor += 1
                for left in range(len(columns)):
                    for right in range(left, len(columns)):
                        gram[left + 1][right + 1] = gram[right + 1][left + 1] = float(
                            values[cursor]
                        )
                        cursor += 1
                response = [float(values[cursor])]
                response.extend(float(value) for value in values[cursor + 1 :])
                for index in range(1, dimension):
                    gram[index][index] += 1e-4 * training_rows
                coefficients = _solve(gram, response)
                model = {
                    "schema_version": 2,
                    "model_type": "deterministic_ridge_regression",
                    "target": target,
                    "horizon_minutes": horizon,
                    "feature_columns": columns,
                    "standardization_means": means,
                    "standardization_scales": scales,
                    "weights": coefficients[1:],
                    "intercept": coefficients[0],
                    "training_rows": training_rows,
                    "dataset_sha256": expected_hash,
                }
                path = output_dir / f"ridge-{target}-{horizon}m.json"
                write_json(path, model)
                files.append({"path": path.name, "sha256": sha256_file(path)})
    path = output_dir / "model-manifest.json"
    write_json(
        path,
        {
            "schema_version": 2,
            "forecast_kind": "station_demand",
            "dataset_sha256": expected_hash,
            "models": files,
        },
    )
    return path
