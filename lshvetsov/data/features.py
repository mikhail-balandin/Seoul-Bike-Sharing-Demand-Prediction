import numpy as np
import pandas as pd

from . import TIMESTAMP


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
