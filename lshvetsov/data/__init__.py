from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..paths import DATA_PATH


SEOUL_TIMEZONE = "Asia/Seoul"
TIMESTAMP = "timestamp"
TARGET = "Rented Bike Count"
TEST_START = pd.Timestamp("2018-11-01", tz=SEOUL_TIMEZONE)
TEST_END = TEST_START + pd.offsets.MonthBegin()
RANDOM_SPLIT_SEED = 42


@dataclass(frozen=True)
class ValidationPeriod:
    name: str
    start: str
    stop: str


VALIDATION_SCHEMES = {
    "rolling_cv": (
        ValidationPeriod("2018_07", "2018-07-01", "2018-08-01"),
        ValidationPeriod("2018_08", "2018-08-01", "2018-09-01"),
        ValidationPeriod("2018_09", "2018-09-01", "2018-10-01"),
        ValidationPeriod("2018_10", "2018-10-01", "2018-11-01"),
    ),
}


SELECTION_SCHEMES = {
    "rolling_cv": "Временные фолды",
    "monthly_80_20": "80/20 внутри месяца",
    "random_80_20": "Случайное 80/20",
}


def load_hourly_data(path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="latin1")
    hours = data["Hour"]
    if not pd.api.types.is_integer_dtype(hours) or not hours.between(0, 23).all():
        raise ValueError("Hour must contain integers between 0 and 23.")

    dates = pd.to_datetime(data["Date"], format="%d/%m/%Y")
    data[TIMESTAMP] = (dates + pd.to_timedelta(hours, unit="h")).dt.tz_localize(
        SEOUL_TIMEZONE
    )
    data = data.sort_values(TIMESTAMP, ignore_index=True)
    hourly_steps = data[TIMESTAMP].diff().iloc[1:]
    if data.empty or data[TIMESTAMP].isna().any() or not hourly_steps.eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError("The dataset must contain a continuous hourly timeline.")
    if not data["Functioning Day"].isin(["Yes", "No"]).all():
        raise ValueError("Functioning Day must contain Yes or No.")
    if not data["Holiday"].isin(["Holiday", "No Holiday"]).all():
        raise ValueError("Holiday must contain Holiday or No Holiday.")

    return data


def split_operating_hours(
    data: pd.DataFrame, period: ValidationPeriod
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(period.start, tz=SEOUL_TIMEZONE)
    stop = pd.Timestamp(period.stop, tz=SEOUL_TIMEZONE)
    if not start < stop <= TEST_START:
        raise ValueError(f"Invalid validation boundaries for {period.name}.")

    opened = data["Functioning Day"].eq("Yes")
    timestamps = data[TIMESTAMP]
    train = data.loc[opened & timestamps.lt(start)]
    validation = data.loc[opened & timestamps.between(start, stop, inclusive="left")]
    if train.empty or validation.empty:
        raise ValueError(f"No operating hours for train or validation in {period.name}.")
    return train, validation


def split_november(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = data[TIMESTAMP]
    train = data.loc[data["Functioning Day"].eq("Yes") & timestamps.lt(TEST_START)]
    test = data.loc[timestamps.between(TEST_START, TEST_END, inclusive="left")]
    expected_hours = int((TEST_END - TEST_START) / pd.Timedelta(hours=1))
    if train.empty or len(test) != expected_hours:
        raise ValueError("The training period and all November test hours are required.")
    return train, test


def split_monthly_80_20(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = data.sort_values(TIMESTAMP)
    month = ordered[TIMESTAMP].dt.strftime("%Y-%m")
    groups = ordered.groupby(month, sort=False)
    cutoff = groups[TIMESTAMP].transform("size") * 4 // 5
    first_part = groups.cumcount().lt(cutoff)
    opened = ordered["Functioning Day"].eq("Yes")
    train = ordered.loc[opened & first_part]
    validation = ordered.loc[opened & ~first_part]
    if train.empty or validation.empty:
        raise ValueError("No operating hours for the pooled monthly 80/20 split.")
    return train, validation


def split_random_80_20(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool the first two parts of the reproducible shuffled 60/20/20 split."""
    opened = data.loc[data["Functioning Day"].eq("Yes")]
    train, remainder = train_test_split(
        opened, test_size=0.40, random_state=RANDOM_SPLIT_SEED
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=RANDOM_SPLIT_SEED
    )
    return pd.concat([train, validation]), test
