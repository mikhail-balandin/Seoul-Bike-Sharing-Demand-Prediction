"""Проверки разбора ошибок финальной модели."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from error_analysis_07 import (
    build_calibration,
    build_top_errors,
    summarize_errors,
)


def make_results():
    """Создаёт маленькую таблицу ошибок для проверок."""
    actual = np.array([10, 20, 30, 40], dtype=float)
    prediction = np.array([5, 25, 40, 20], dtype=float)
    residual = actual - prediction

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"]
            ),
            "Hour": [8, 8, 18, 18],
            "Seasons": ["Winter", "Winter", "Summer", "Summer"],
            "actual": actual,
            "prediction": prediction,
            "residual": residual,
            "abs_error": np.abs(residual),
            "squared_error": residual**2,
            "Group": ["A", "A", "B", "B"],
        }
    )


class ErrorAnalysisTests(unittest.TestCase):
    def test_summarize_errors_calculates_metrics(self) -> None:
        results = make_results()

        with patch("error_analysis_07.MIN_GROUP_SIZE", 2):
            summary = summarize_errors(results, "Group")

        self.assertEqual(summary.loc["A", "Observations"], 2)
        self.assertAlmostEqual(summary.loc["A", "RMSE"], 5.0)
        self.assertAlmostEqual(summary.loc["A", "MAE"], 5.0)
        self.assertAlmostEqual(summary.loc["A", "Bias"], 0.0)

        self.assertAlmostEqual(
            summary.loc["B", "RMSE"],
            np.sqrt(250.0),
        )
        self.assertAlmostEqual(summary.loc["B", "MAE"], 15.0)
        self.assertAlmostEqual(summary.loc["B", "Bias"], 5.0)
        self.assertTrue(summary["Reliable"].all())

    def test_summarize_errors_marks_small_groups(self) -> None:
        results = make_results()

        with patch("error_analysis_07.MIN_GROUP_SIZE", 3):
            summary = summarize_errors(results, "Group")

        self.assertFalse(summary["Reliable"].any())

    def test_calibration_keeps_all_observations(self) -> None:
        prediction = np.arange(1, 21, dtype=float)
        actual = prediction + 1

        results = pd.DataFrame(
            {
                "actual": actual,
                "prediction": prediction,
                "residual": actual - prediction,
                "squared_error": (actual - prediction) ** 2,
            }
        )

        calibration = build_calibration(results, q=4)

        self.assertEqual(
            int(calibration["Observations"].sum()),
            len(results),
        )
        self.assertAlmostEqual(
            float(calibration["Bias"].mean()),
            1.0,
        )
        self.assertAlmostEqual(
            float(calibration["RMSE"].mean()),
            1.0,
        )

    def test_top_errors_are_sorted(self) -> None:
        results = make_results()
        top = build_top_errors(results, n=2)

        self.assertEqual(len(top), 2)
        self.assertTrue(
            top["abs_error"].is_monotonic_decreasing
        )
        self.assertEqual(top.iloc[0]["Date"], "2018-01-04")
        self.assertEqual(top.iloc[0]["abs_error"], 20.0)


if __name__ == "__main__":
    unittest.main()
