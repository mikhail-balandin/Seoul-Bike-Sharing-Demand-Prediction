import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .data import (
    SELECTION_SCHEMES,
    TARGET,
    TEST_START,
    TIMESTAMP,
    VALIDATION_SCHEMES,
    load_hourly_data,
    split_monthly_80_20,
    split_operating_hours,
    split_random_80_20,
)
from .features import FEATURE_SETS, build_features, make_model
from .models import HourlyMedian, MedianResidualBoosting


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
LASSO_ALPHAS = (0.1, 1.0, 10.0, 100.0)
LASSO_MAX_ITER = 20_000
BOOSTING_GRID = {
    "max_iter": (180, 200),
    "max_leaf_nodes": (24, 31),
    "learning_rate": (0.05, 0.1),
}
BOOSTING_FIXED = {"min_samples_leaf": 20, "early_stopping": False, "random_state": 42}
PEAK_QUANTILE = 0.9


def candidate_models() -> dict[str, RegressorMixin]:
    candidates = {"linear": LinearRegression()}
    candidates.update({f"ridge_{alpha:g}": Ridge(alpha=alpha) for alpha in RIDGE_ALPHAS})
    candidates.update(
        {f"lasso_{alpha:g}": Lasso(alpha=alpha, max_iter=LASSO_MAX_ITER) for alpha in LASSO_ALPHAS}
    )
    for iterations, leaves, rate in product(*BOOSTING_GRID.values()):
        settings = dict(zip(BOOSTING_GRID, (iterations, leaves, rate)))
        boosting = HistGradientBoostingRegressor(**settings, **BOOSTING_FIXED)
        suffix = f"{iterations}_{leaves}_{rate:g}"
        candidates[f"hgb_{suffix}"] = boosting
        candidates[f"median_hgb_{suffix}"] = MedianResidualBoosting(boosting)
    return candidates


def model_parameters(estimator: RegressorMixin) -> dict[str, int | float | bool]:
    if isinstance(estimator, MedianResidualBoosting):
        estimator = estimator.regressor
    params = estimator.get_params()
    if isinstance(estimator, HistGradientBoostingRegressor):
        return {key: params[key] for key in (*BOOSTING_GRID, *BOOSTING_FIXED)}
    if isinstance(estimator, Ridge):
        return {"alpha": params["alpha"], "fit_intercept": params["fit_intercept"]}
    if isinstance(estimator, Lasso):
        return {key: params[key] for key in ("alpha", "fit_intercept", "max_iter")}
    return {"fit_intercept": params["fit_intercept"]}


def baseline_predict(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    model = HourlyMedian().fit(build_features(train, ()), train[TARGET])
    return model.predict(build_features(validation, ()))


def score_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    peaks = actual >= np.quantile(actual, PEAK_QUANTILE)
    return {
        "rmse": root_mean_squared_error(actual, predicted),
        "mae": mean_absolute_error(actual, predicted),
        "r2": r2_score(actual, predicted),
        "peak_rmse": root_mean_squared_error(actual[peaks], predicted[peaks]),
        "validation_hours": len(actual),
    }


def summarize_folds(
    actual: dict[str, np.ndarray], predictions: dict[str, np.ndarray]
) -> dict[str, float | int]:
    summary = score_predictions(
        np.concatenate(list(actual.values())),
        np.concatenate([predictions[name] for name in actual]),
    )
    if len(actual) > 1:
        for period, observed in actual.items():
            summary[f"rmse_{period}"] = root_mean_squared_error(observed, predictions[period])
    return summary


def evaluate_scheme(folds: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    actual = {name: check[TARGET].to_numpy() for name, (_, check) in folds.items()}
    baseline = {
        name: baseline_predict(train, check) for name, (train, check) in folds.items()
    }
    rows = [
        {
            "feature_set": "baseline",
            "algorithm": "HourlyMedian",
            "configuration": "baseline",
            "parameters": "{}",
            **summarize_folds(actual, baseline),
        }
    ]
    candidates = candidate_models()
    for feature_set, weather in FEATURE_SETS.items():
        feature_folds = {
            name: (build_features(train, weather), train[TARGET], build_features(check, weather))
            for name, (train, check) in folds.items()
        }
        for configuration, estimator in candidates.items():
            predictions = {}
            for name, (train_features, target, valid_features) in feature_folds.items():
                model = make_model(estimator, weather)
                model.fit(train_features, target)
                predictions[name] = model.predict(valid_features).clip(min=0)
            rows.append(
                {
                    "feature_set": feature_set,
                    "algorithm": type(estimator).__name__,
                    "configuration": configuration,
                    "parameters": json.dumps(model_parameters(estimator)),
                    **summarize_folds(actual, predictions),
                }
            )
    return pd.DataFrame(rows).sort_values(["rmse", "configuration"], ignore_index=True)


def write_scores(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, float_format="%.8f", lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hourly rental forecasts before November.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    hourly = load_hourly_data()
    development = hourly.loc[hourly[TIMESTAMP].lt(TEST_START)]
    schemes = {
        name: {
            period.name: split_operating_hours(development, period)
            for period in periods
        }
        for name, periods in VALIDATION_SCHEMES.items()
    }
    schemes["monthly_80_20"] = {"month_tails": split_monthly_80_20(development)}
    schemes["random_80_20"] = {"shuffled_hours": split_random_80_20(development)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scheme in SELECTION_SCHEMES:
        results = evaluate_scheme(schemes[scheme])
        path = args.output_dir / f"{scheme}.csv"
        write_scores(results, path)
        print(f"{scheme} — {len(results)} configurations — {path}", flush=True)


if __name__ == "__main__":
    main()
