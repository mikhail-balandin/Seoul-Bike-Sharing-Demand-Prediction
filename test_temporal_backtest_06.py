"""Проверки независимого хронологического backtest."""

import unittest

from feature_engineering_02 import TARGET_COLUMN
from temporal_backtest_06 import chronological_split, load_frame, random_split


class TemporalBacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X, cls.y, cls.dates, cls.operating = load_frame()

    def test_target_is_not_a_feature(self) -> None:
        self.assertNotIn(TARGET_COLUMN, self.X.columns)

    def test_splits_are_disjoint_and_cover_every_row(self) -> None:
        for split in (random_split(self.dates), chronological_split(self.dates)):
            train = set(split.train.tolist())
            validation = set(split.validation.tolist())
            test = set(split.test.tolist())
            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)
            self.assertEqual(train | validation | test, set(range(len(self.X))))

    def test_chronological_split_keeps_time_order(self) -> None:
        split = chronological_split(self.dates)
        train_days = self.dates.iloc[split.train]
        validation_days = self.dates.iloc[split.validation]
        test_days = self.dates.iloc[split.test]
        self.assertLessEqual(train_days.max(), validation_days.min())
        self.assertLessEqual(validation_days.max(), test_days.min())

    def test_non_operating_targets_are_zero(self) -> None:
        self.assertEqual(int((self.y[~self.operating] != 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
