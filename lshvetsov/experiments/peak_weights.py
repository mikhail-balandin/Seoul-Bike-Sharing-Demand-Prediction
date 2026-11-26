import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import (
    TARGET, TIMESTAMP, VALIDATION_SCHEMES, load_hourly_data, split_november, split_operating_hours,
)
from ..data.features import ALL_WEATHER, build_features
from ..paths import RESULTS_DIR
from ..validation.models import make_model
from ..validation.train import PEAK_QUANTILE, candidate_models, score_predictions, write_scores


BASE_CONFIGURATION = "hgb_200_24_0.1"
RESULT_FILE = "peak_weight_metrics.csv"
RECENCY_HALF_LIVES = (30, 60, 120)
PEAK_MULTIPLIERS = (2, 4)


def recency_weights(train: pd.DataFrame, half_life_days: int) -> np.ndarray:
    timestamps = train[TIMESTAMP]
    age_days = (timestamps.max() - timestamps) / pd.Timedelta(days=1)
    return np.exp2(-age_days.to_numpy() / half_life_days)


def peak_weights(train: pd.DataFrame, multiplier: int) -> np.ndarray:
    demand = train[TARGET].to_numpy()
    threshold = np.quantile(demand, PEAK_QUANTILE)
    return np.where(demand >= threshold, float(multiplier), 1.0)


def compare_weights(hourly: pd.DataFrame) -> pd.DataFrame:
    folds = {
        period.name: split_operating_hours(hourly, period)
        for period in VALIDATION_SCHEMES["rolling_cv"]
    }
    rolling_periods = tuple(folds)
    train, november = split_november(hourly)
    folds["november"] = (train, november.loc[november["Functioning Day"].eq("Yes")])
    estimator = candidate_models()[BASE_CONFIGURATION]
    actual = {period: check[TARGET].to_numpy() for period, (_, check) in folds.items()}
    predictions: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    for period, (train, check) in folds.items():
        train_features = build_features(train, ALL_WEATHER)
        check_features = build_features(check, ALL_WEATHER)
        variants = [
            ("uniform", 1, None),
            *(("recency", days, recency_weights(train, days)) for days in RECENCY_HALF_LIVES),
            *(("peak", factor, peak_weights(train, factor)) for factor in PEAK_MULTIPLIERS),
        ]
        for scheme, setting, weights in variants:
            model = make_model(estimator, ALL_WEATHER)
            model.fit(train_features, train[TARGET], model__sample_weight=weights)
            forecast = model.predict(check_features).clip(min=0)
            predictions.setdefault((scheme, setting), {})[period] = forecast

    rows = []
    rolling_actual = np.concatenate([actual[period] for period in rolling_periods])
    for (scheme, setting), predicted in predictions.items():
        rolling_predictions = np.concatenate([predicted[period] for period in rolling_periods])
        evaluations = [("rolling_cv", rolling_actual, rolling_predictions)]
        evaluations.extend((period, actual[period], forecast) for period, forecast in predicted.items())
        for period, observed, forecast in evaluations:
            rows.append({
                "configuration": BASE_CONFIGURATION,
                "weighting": scheme,
                "setting": setting,
                "period": period,
                **score_predictions(observed, forecast),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare training weights for the full-weather boosting model.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    scores = compare_weights(load_hourly_data())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / RESULT_FILE
    write_scores(scores, path)
    columns = ["weighting", "setting", "period", "rmse", "peak_rmse", "validation_hours"]
    print(scores.loc[scores["period"].isin(["rolling_cv", "november"]), columns].to_string(index=False))
    print(path)


if __name__ == "__main__":
    main()
