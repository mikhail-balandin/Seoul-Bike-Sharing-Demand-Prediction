import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from math import isfinite
from pathlib import Path

import joblib
import pandas as pd

from ..data import SEOUL_TIMEZONE, TIMESTAMP
from ..data.features import FEATURE_SETS, build_features
from ..data.protocol import SCHEME_LABELS
from ..paths import MODEL_DIR


@dataclass(frozen=True)
class SavedModel:
    algorithm: str
    validation_rmse: float
    test_rmse: float
    test_peak_rmse: float
    test_mae: float
    test_r2: float
    test_hours: int
    weather_columns: tuple[str, ...]
    path: Path
    parameters: Mapping[str, float | int | bool]


def load_model_schemes(directory: Path = MODEL_DIR) -> dict[str, dict[str, SavedModel]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["selection"] != "validation_rmse_per_scheme":
        raise ValueError("Пересоберите манифест командой finalize из README проекта.")
    entries = {(entry["scheme"], entry["feature_set"]): entry for entry in manifest["models"]}
    if len(entries) != len(manifest["models"]) or entries.keys() != {
        (scheme, feature_set) for scheme in SCHEME_LABELS for feature_set in FEATURE_SETS
    }:
        raise ValueError("Манифест должен содержать все сценарии каждой схемы.")

    models = {scheme: {} for scheme in SCHEME_LABELS}
    for (scheme, feature_set), entry in entries.items():
        weather = FEATURE_SETS[feature_set]
        model_file = entry["model_file"]
        if tuple(entry["weather_columns"]) != weather:
            raise ValueError(f"Манифест модели {feature_set} не совпадает со схемой входов.")
        if Path(model_file).name != model_file or not model_file.endswith(".joblib"):
            raise ValueError(f"Некорректный файл модели {feature_set}.")
        models[scheme][feature_set] = SavedModel(
            algorithm=entry["algorithm"],
            validation_rmse=entry["validation_rmse"],
            test_rmse=entry["test_rmse"],
            test_peak_rmse=entry["test_peak_rmse"],
            test_mae=entry["test_mae"],
            test_r2=entry["test_r2"],
            test_hours=entry["test_hours"],
            weather_columns=weather,
            path=directory / model_file,
            parameters=entry["parameters"],
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
