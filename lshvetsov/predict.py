import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from math import isfinite
from pathlib import Path

import joblib
import pandas as pd

from .data import SEOUL_TIMEZONE, TIMESTAMP
from .features import FEATURE_SETS, build_features


MODEL_DIR = Path(__file__).resolve().parent / "results" / "models"


@dataclass(frozen=True)
class SavedModel:
    algorithm: str
    alpha: float | None
    november_rmse: float
    weather_columns: tuple[str, ...]
    path: Path
    parameters: Mapping[str, float | int | bool]
    selection_schemes: tuple[str, ...] = ()
    november_peak_rmse: float | None = None


def load_models(directory: Path = MODEL_DIR) -> dict[str, SavedModel]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    entries = {entry["feature_set"]: entry for entry in manifest["models"]}
    if entries.keys() != FEATURE_SETS.keys():
        raise ValueError("Манифест должен содержать три набора входов из эксперимента.")

    models = {}
    for feature_set, weather in FEATURE_SETS.items():
        entry = entries[feature_set]
        model_file = f"{feature_set}.joblib"
        if tuple(entry["weather_columns"]) != weather or entry["model_file"] != model_file:
            raise ValueError(f"Манифест модели {feature_set} не совпадает со схемой входов.")
        models[feature_set] = SavedModel(
            algorithm=entry["algorithm"],
            alpha=entry["alpha"],
            november_rmse=entry["november_rmse"],
            weather_columns=weather,
            path=directory / model_file,
            parameters=entry.get("parameters", {}),
            selection_schemes=tuple(entry.get("selection_schemes", ())),
            november_peak_rmse=entry.get("november_peak_rmse"),
        )
    return models


def predict_rentals(
    model: SavedModel,
    *,
    day: date,
    hour: int,
    is_holiday: bool,
    is_open: bool,
    weather: Mapping[str, float],
) -> float:
    if not is_open:
        return 0.0

    missing = [column for column in model.weather_columns if column not in weather]
    if missing:
        raise ValueError(f"Заполните погодные поля: {', '.join(missing)}.")
    values = {column: weather[column] for column in model.weather_columns}
    if not all(isfinite(value) for value in values.values()):
        raise ValueError("Погодные значения должны быть конечными числами.")

    timestamp = pd.Timestamp(datetime.combine(day, time(hour)), tz=SEOUL_TIMEZONE)
    inputs = pd.DataFrame(
        [{TIMESTAMP: timestamp, "Holiday": "Holiday" if is_holiday else "No Holiday", **values}]
    )
    features = build_features(inputs, model.weather_columns)
    pipeline = joblib.load(model.path)
    return float(pipeline.predict(features).clip(min=0)[0])
