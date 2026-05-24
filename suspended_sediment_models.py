from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 1000
MIN_DATASET_ROWS = 3


@dataclass
class DataSet:
    features: list[list[float]]
    targets: list[float]


@dataclass
class ModelResult:
    model_name: str
    mae: float
    rmse: float
    r2: float


class MeanRegressor:
    name = "MeanBaseline"

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, _features: Sequence[Sequence[float]], targets: Sequence[float]) -> None:
        self._mean = sum(targets) / len(targets)

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        return [self._mean] * len(features)


class LinearRegressionGD:
    name = "LinearRegressionGD"

    def __init__(
        self, learning_rate: float = DEFAULT_LEARNING_RATE, epochs: int = DEFAULT_EPOCHS
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._weights: list[float] = []
        self._means: list[float] = []
        self._stds: list[float] = []

    def _normalize(self, row: Sequence[float]) -> list[float]:
        normalized: list[float] = []
        for idx, value in enumerate(row):
            std = self._stds[idx]
            if std == 0:
                normalized.append(0.0)
            else:
                normalized.append((value - self._means[idx]) / std)
        return normalized

    def fit(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> None:
        if not features:
            raise ValueError("features cannot be empty")
        feature_count = len(features[0])
        self._means = [
            sum(row[idx] for row in features) / len(features) for idx in range(feature_count)
        ]
        self._stds = []
        for idx in range(feature_count):
            variance = sum((row[idx] - self._means[idx]) ** 2 for row in features) / len(features)
            self._stds.append(math.sqrt(variance))

        normalized_features = [self._normalize(row) for row in features]
        self._weights = [0.0] * (feature_count + 1)

        for _ in range(self.epochs):
            gradients = [0.0] * (feature_count + 1)
            sample_count = len(features)

            for row, target in zip(normalized_features, targets):
                prediction = self._weights[0]
                for idx, value in enumerate(row, start=1):
                    prediction += self._weights[idx] * value

                error = prediction - target
                gradients[0] += error
                for idx, value in enumerate(row, start=1):
                    gradients[idx] += error * value

            for idx in range(len(self._weights)):
                self._weights[idx] -= self.learning_rate * (gradients[idx] / sample_count)

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        predictions: list[float] = []
        for row in features:
            normalized_row = self._normalize(row)
            value = self._weights[0]
            for idx, feature in enumerate(normalized_row, start=1):
                value += self._weights[idx] * feature
            predictions.append(value)
        return predictions


class KNNRegressor:
    name = "KNNRegressor(k=3)"

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self._features: list[list[float]] = []
        self._targets: list[float] = []

    def fit(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> None:
        self._features = [list(row) for row in features]
        self._targets = list(targets)

    def _distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        predictions: list[float] = []
        for row in features:
            distances = sorted(
                (
                    (self._distance(row, training_row), target)
                    for training_row, target in zip(self._features, self._targets)
                ),
                key=lambda pair: pair[0],
            )
            nearest = [target for _, target in distances[: self.k]]
            predictions.append(sum(nearest) / len(nearest))
        return predictions


def read_csv_dataset(path: Path, target_column: str, feature_columns: Iterable[str] | None = None) -> DataSet:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file must include a header row")

        if target_column not in reader.fieldnames:
            raise ValueError(f"Target column '{target_column}' not found")

        selected_features = list(feature_columns) if feature_columns else [
            field for field in reader.fieldnames if field != target_column
        ]

        if not selected_features:
            raise ValueError("At least one feature column is required")

        missing = [field for field in selected_features if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"Feature columns not found: {', '.join(missing)}")

        features: list[list[float]] = []
        targets: list[float] = []
        for row in reader:
            features.append([float(row[field]) for field in selected_features])
            targets.append(float(row[target_column]))

    if len(features) < MIN_DATASET_ROWS:
        raise ValueError(
            f"CSV must contain at least {MIN_DATASET_ROWS} rows for train/test evaluation"
        )

    return DataSet(features=features, targets=targets)


def train_test_split(dataset: DataSet, test_ratio: float = 0.2, seed: int = 42) -> tuple[DataSet, DataSet]:
    if len(dataset.features) < 2:
        raise ValueError("dataset must contain at least 2 rows")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    indices = list(range(len(dataset.features)))
    random.Random(seed).shuffle(indices)

    test_count = max(1, int(len(indices) * test_ratio))
    if test_count >= len(indices):
        test_count = len(indices) - 1

    test_indices = set(indices[:test_count])

    train_features, train_targets = [], []
    test_features, test_targets = [], []

    for idx, (feature_row, target) in enumerate(zip(dataset.features, dataset.targets)):
        if idx in test_indices:
            test_features.append(feature_row)
            test_targets.append(target)
        else:
            train_features.append(feature_row)
            train_targets.append(target)

    return DataSet(train_features, train_targets), DataSet(test_features, test_targets)


def _validate_metric_inputs(actual: Sequence[float], predicted: Sequence[float]) -> None:
    if not actual or not predicted:
        raise ValueError("actual and predicted must be non-empty")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    _validate_metric_inputs(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    _validate_metric_inputs(actual, predicted)
    squared_error = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
    return math.sqrt(squared_error)


def r2(actual: Sequence[float], predicted: Sequence[float]) -> float:
    _validate_metric_inputs(actual, predicted)
    mean_actual = sum(actual) / len(actual)
    total_variance = sum((value - mean_actual) ** 2 for value in actual)
    if total_variance == 0:
        return 0.0
    residual_variance = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return 1 - (residual_variance / total_variance)


def compare_models(dataset: DataSet) -> list[ModelResult]:
    train_set, test_set = train_test_split(dataset)
    models = [MeanRegressor(), LinearRegressionGD(), KNNRegressor()]

    results: list[ModelResult] = []
    for model in models:
        model.fit(train_set.features, train_set.targets)
        predictions = model.predict(test_set.features)
        results.append(
            ModelResult(
                model_name=model.name,
                mae=mae(test_set.targets, predictions),
                rmse=rmse(test_set.targets, predictions),
                r2=r2(test_set.targets, predictions),
            )
        )

    return sorted(results, key=lambda result: result.rmse)


def _format_results(results: Sequence[ModelResult]) -> str:
    lines = ["Model,MAE,RMSE,R2"]
    for result in results:
        lines.append(
            f"{result.model_name},{result.mae:.6f},{result.rmse:.6f},{result.r2:.6f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare models for suspended sediment load prediction"
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column that stores suspended sediment load",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        help="Optional feature columns to use. Defaults to all columns except target",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = read_csv_dataset(args.data, args.target, args.features)
    results = compare_models(dataset)
    print(_format_results(results))


if __name__ == "__main__":
    main()
