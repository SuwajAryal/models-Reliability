import csv
import tempfile
import unittest
from pathlib import Path

from suspended_sediment_models import (
    DataSet,
    KNNRegressor,
    LinearRegressionGD,
    MeanRegressor,
    compare_models,
    read_csv_dataset,
    train_test_split,
)


class SuspendedSedimentModelTests(unittest.TestCase):
    def test_compare_models_returns_ranked_results(self):
        features = [[float(x)] for x in range(1, 21)]
        targets = [3 * row[0] + 2 for row in features]
        dataset = DataSet(features=features, targets=targets)

        results = compare_models(dataset)
        names = [result.model_name for result in results]

        self.assertEqual(len(results), 3)
        self.assertCountEqual(
            names, [KNNRegressor.name, LinearRegressionGD.name, MeanRegressor.name]
        )
        self.assertLessEqual(results[0].rmse, results[1].rmse)
        self.assertLessEqual(results[1].rmse, results[2].rmse)
        self.assertEqual(results[-1].model_name, MeanRegressor.name)
        self.assertLess(results[0].rmse, results[-1].rmse)

    def test_read_csv_dataset_supports_target_and_feature_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sediment.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["discharge", "rainfall", "suspended_load"])
                writer.writerow([10, 5, 30])
                writer.writerow([12, 8, 36])
                writer.writerow([14, 6, 42])

            dataset = read_csv_dataset(
                path=csv_path,
                target_column="suspended_load",
                feature_columns=["discharge", "rainfall"],
            )

            self.assertEqual(dataset.features, [[10.0, 5.0], [12.0, 8.0], [14.0, 6.0]])
            self.assertEqual(dataset.targets, [30.0, 36.0, 42.0])

    def test_train_test_split_rejects_zero_and_one_test_ratio(self):
        dataset = DataSet(features=[[1.0], [2.0], [3.0]], targets=[1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            train_test_split(dataset, test_ratio=0.0)

        with self.assertRaises(ValueError):
            train_test_split(dataset, test_ratio=1.0)


if __name__ == "__main__":
    unittest.main()
