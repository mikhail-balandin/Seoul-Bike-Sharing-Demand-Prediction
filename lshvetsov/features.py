import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import TIMESTAMP
from .models import MedianResidualBoosting


CALENDAR_CATEGORIES = ["hour", "weekday", "weekend_hour"]
BASIC_WEATHER = (
    "Temperature(°C)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Rainfall(mm)",
)
ALL_WEATHER = BASIC_WEATHER + (
    "Humidity(%)",
    "Solar Radiation (MJ/m2)",
    "Snowfall (cm)",
)
FEATURE_SETS = {
    "calendar": (),
    "calendar_weather_basic": BASIC_WEATHER,
    "calendar_weather_all": ALL_WEATHER,
}
FEATURE_LABELS = {
    "calendar": "Календарь",
    "calendar_weather_basic": "Календарь + основная погода",
    "calendar_weather_all": "Календарь + вся погода",
    "baseline": "Почасовая медиана",
}


def build_features(data: pd.DataFrame, weather: tuple[str, ...]) -> pd.DataFrame:
    timestamps = data[TIMESTAMP].dt
    days_in_year = np.where(timestamps.is_leap_year, 366, 365)
    annual_phase = 2 * np.pi * (timestamps.day_of_year - 1) / days_in_year
    return data.loc[:, list(weather)].assign(
        hour=timestamps.hour,
        weekday=timestamps.day_of_week,
        weekend_hour=timestamps.hour.where(timestamps.day_of_week.ge(5), -1),
        annual_sin=np.sin(annual_phase),
        annual_cos=np.cos(annual_phase),
        is_holiday=data["Holiday"].eq("Holiday").astype("int8"),
    )


def make_model(regressor: RegressorMixin, weather: tuple[str, ...]) -> RegressorMixin:
    residual_model = isinstance(regressor, MedianResidualBoosting)
    estimator = regressor.regressor if residual_model else regressor
    needs_dense_input = isinstance(estimator, HistGradientBoostingRegressor)
    preprocessing = ColumnTransformer(
        [
            (
                "calendar",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not needs_dense_input),
                CALENDAR_CATEGORIES,
            ),
            ("numeric_calendar", "passthrough", ["is_holiday", "annual_sin", "annual_cos"]),
            ("weather", StandardScaler(), list(weather)),
        ]
    )
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", clone(estimator))])
    return MedianResidualBoosting(pipeline) if residual_model else pipeline
