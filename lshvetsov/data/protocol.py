from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import RANDOM_SPLIT_SEED, TIMESTAMP, VALIDATION_SCHEMES, split_operating_hours


ROLLING = "rolling_cv"
MONTHLY = "monthly_80_10_10"
RANDOM = "random_80_10_10"
SCHEME_LABELS = {
    ROLLING: "Временные фолды",
    MONTHLY: "Случайное 80/10/10 внутри месяца",
    RANDOM: "Случайное 80/10/10 по году",
}
MEMBERSHIP_FILE = "split_membership.csv"
TEST_METRICS_FILE = "test_metrics.csv"
TEST_PREDICTIONS_FILE = "test_predictions.csv"
VALIDATION_MODELS_FILE = "validation_winners.joblib"


@dataclass(frozen=True)
class EvaluationSplit:
    folds: dict[str, tuple[pd.DataFrame, pd.DataFrame]]
    final_train: pd.DataFrame
    test: pd.DataFrame


def holdout_hours(count: int) -> int:
    return (count + 5) // 10


def random_split(opened: pd.DataFrame, *, within_month: bool) -> EvaluationSplit:
    rng = np.random.default_rng(RANDOM_SPLIT_SEED)
    groups = (
        [month for _, month in opened.groupby(opened[TIMESTAMP].dt.strftime("%Y-%m"), sort=True)]
        if within_month else [opened]
    )
    training, validation, test = [], [], []
    for group in groups:
        count = holdout_hours(len(group))
        if count == 0 or 2 * count >= len(group):
            raise ValueError("Each group needs open hours for train, validation and test.")
        shuffled = rng.permutation(group.index)
        validation.extend(shuffled[:count])
        test.extend(shuffled[count:2 * count])
        training.extend(shuffled[2 * count:])
    train = opened.loc[sorted(training)]
    return EvaluationSplit(
        {"holdout": (train, opened.loc[sorted(validation)])}, train, opened.loc[sorted(test)],
    )


def make_splits(hourly: pd.DataFrame) -> dict[str, EvaluationSplit]:
    opened = hourly.loc[hourly["Functioning Day"].eq("Yes")].sort_values(TIMESTAMP)
    count = holdout_hours(len(opened))
    if count == 0:
        raise ValueError("The time-based scheme requires a test period.")
    development, test = opened.iloc[:-count], opened.iloc[-count:]
    folds = {
        period.name: split_operating_hours(development, period)
        for period in VALIDATION_SCHEMES[ROLLING]
    }
    return {
        ROLLING: EvaluationSplit(folds, development, test),
        MONTHLY: random_split(opened, within_month=True),
        RANDOM: random_split(opened, within_month=False),
    }


def split_membership(splits: dict[str, EvaluationSplit]) -> pd.DataFrame:
    rolling = splits[ROLLING]
    hours = pd.concat([rolling.final_train[TIMESTAMP], rolling.test[TIMESTAMP]])
    table = hours.to_frame()
    for scheme, split in splits.items():
        table[scheme] = "train"
        table.loc[split.test.index, scheme] = "test"
        if scheme != ROLLING:
            table.loc[split.folds["holdout"][1].index, scheme] = "validation"
    table["rolling_validation_fold"] = ""
    for period, (_, validation) in rolling.folds.items():
        table.loc[validation.index, "rolling_validation_fold"] = period
    return table.reset_index(drop=True)


def validation_file(scheme: str) -> str:
    return f"validation_{scheme}.csv"
