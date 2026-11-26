import argparse
import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from ..data import TARGET, load_hourly_data
from ..data.features import FEATURE_SETS, build_features
from ..data.protocol import (
    MEMBERSHIP_FILE, ROLLING, VALIDATION_MODELS_FILE, make_splits, split_membership, validation_file,
)
from ..paths import RESULTS_DIR
from .models import HourlyMedian, MedianResidualBoosting, make_model


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


def evaluate_scheme(
    folds: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    retained: dict[str, tuple[str, RegressorMixin]] | None = None,
) -> pd.DataFrame:
    if retained is not None and len(folds) != 1:
        raise ValueError("A fitted winner can only be retained for a single validation split.")
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
        best_key = (float("inf"), "")
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
            metrics = summarize_folds(actual, predictions)
            key = (float(metrics["rmse"]), configuration)
            if retained is not None and key < best_key:
                best_key = key
                retained[feature_set] = (configuration, model)
            rows.append(
                {
                    "feature_set": feature_set,
                    "algorithm": type(estimator).__name__,
                    "configuration": configuration,
                    "parameters": json.dumps(model_parameters(estimator)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["rmse", "configuration"], ignore_index=True)


def write_scores(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, float_format="%.8f", lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select models using validation hours in each split.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    splits = make_splits(load_hourly_data())
    winners = {}
    results = {}
    for scheme, split in splits.items():
        retained = {} if scheme != ROLLING else None
        results[scheme] = evaluate_scheme(split.folds, retained=retained)
        if retained is not None:
            winners[scheme] = retained
        print(f"{scheme} — {len(results[scheme])} configurations validated", flush=True)

    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for scheme, scores in results.items():
        write_scores(scores, args.output_dir / validation_file(scheme))
    write_scores(split_membership(splits), args.output_dir / MEMBERSHIP_FILE)
    joblib.dump(winners, models_dir / VALIDATION_MODELS_FILE, compress=3)


if __name__ == "__main__":
    main()
