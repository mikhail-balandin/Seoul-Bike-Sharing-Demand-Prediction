# Независимый backtest: сравниваем случайное и хронологическое
# разбиение. Учимся только на Functioning Day = Yes, нерабочие
# часы получают прогноз 0. Считаем operating- и full-demand
# метрики и отдельно rolling TimeSeriesSplit.


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from eda_01 import load_prepared_data
from feature_engineering_02 import TARGET_COLUMN, build_features

RANDOM_STATE = 42


@dataclass(frozen=True)
class Split:
    name: str
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def load_frame() -> tuple[pd.DataFrame, np.ndarray, pd.Series, np.ndarray]:
    """Возвращает признаки, target, даты и маску рабочих дней."""
    raw = load_prepared_data().sort_values(["Date", "Hour"]).reset_index(drop=True)
    features = build_features(raw).drop(columns=[TARGET_COLUMN])
    target = raw[TARGET_COLUMN].to_numpy()
    dates = raw["Date"]
    operating = raw["Functioning Day"].eq("Yes").to_numpy()

    assert len(features) == len(target) == len(dates)
    assert not features.isna().any().any()
    return features, target, dates, operating


def random_split(dates: pd.Series) -> Split:
    """Случайное разбиение 60/20/20 по всем строкам."""
    indices = np.arange(len(dates))
    train, remainder = train_test_split(
        indices,
        test_size=0.4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return Split("random", np.sort(train), np.sort(validation), np.sort(test))


def chronological_split(dates: pd.Series) -> Split:
    """Делит целые дни по времени: 60% / 20% / 20%."""
    unique_days = np.sort(pd.unique(dates))
    train_end = int(len(unique_days) * 0.6)
    validation_end = int(len(unique_days) * 0.8)

    train_days = unique_days[:train_end]
    validation_days = unique_days[train_end:validation_end]
    test_days = unique_days[validation_end:]

    train = np.flatnonzero(dates.isin(train_days).to_numpy())
    validation = np.flatnonzero(dates.isin(validation_days).to_numpy())
    test = np.flatnonzero(dates.isin(test_days).to_numpy())
    return Split("chronological", train, validation, test)


def _fit_predict(
    model_name: str,
    train_rows: np.ndarray,
    eval_rows: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    operating: np.ndarray,
) -> np.ndarray:
    """Прогнозирует строки; для нерабочих часов возвращает 0."""
    predictions = np.zeros(len(eval_rows), dtype=float)
    train_operating = train_rows[operating[train_rows]]
    eval_operating = eval_rows[operating[eval_rows]]

    if len(train_operating) == 0 or len(eval_operating) == 0:
        return predictions

    x_train = X.iloc[train_operating]
    y_train = y[train_operating]
    x_eval = X.iloc[eval_operating]

    if model_name == "baseline":
        operating_predictions = np.full(len(eval_operating), y_train.mean())
    elif model_name == "linear":
        model = LinearRegression().fit(x_train, y_train)
        operating_predictions = np.clip(model.predict(x_eval), 0.0, None)
    elif model_name == "linear-log":
        model = LinearRegression().fit(x_train, np.log1p(y_train))
        operating_predictions = np.expm1(model.predict(x_eval))
    elif model_name == "hist-gradient-boosting":
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ).fit(x_train, y_train)
        operating_predictions = np.clip(model.predict(x_eval), 0.0, None)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    predictions[operating[eval_rows]] = operating_predictions
    return predictions


def _metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "bias": float(np.mean(y_true - predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def evaluate_split(
    split: Split,
    X: pd.DataFrame,
    y: np.ndarray,
    operating: np.ndarray,
    model_name: str,
) -> dict[str, object]:
    test_predictions = _fit_predict(model_name, split.train, split.test, X, y, operating)
    test_operating = operating[split.test]
    y_test = y[split.test]

    full = _metrics(y_test, test_predictions)
    operating_metrics = _metrics(y_test[test_operating], test_predictions[test_operating])

    return {
        "split": split.name,
        "model": model_name,
        "test_rows": len(split.test),
        "test_operating": int(test_operating.sum()),
        "test_non_operating": int((~test_operating).sum()),
        "full_rmse": full["rmse"],
        "full_mae": full["mae"],
        "full_bias": full["bias"],
        "full_r2": full["r2"],
        "operating_rmse": operating_metrics["rmse"],
        "operating_mae": operating_metrics["mae"],
        "operating_bias": operating_metrics["bias"],
        "negative_predictions": int((test_predictions < 0).sum()),
    }


def rolling_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    dates: pd.Series,
    operating: np.ndarray,
    model_names: list[str],
) -> pd.DataFrame:
    """TimeSeriesSplit по первым 80% времени."""
    chronological = chronological_split(dates)
    development = np.sort(np.concatenate([chronological.train, chronological.validation]))
    cv = TimeSeriesSplit(n_splits=5)

    rows = []
    for fold, (train_position, validation_position) in enumerate(cv.split(development), start=1):
        train_rows = development[train_position]
        validation_rows = development[validation_position]
        for model_name in model_names:
            predictions = _fit_predict(model_name, train_rows, validation_rows, X, y, operating)
            validation_operating = operating[validation_rows]
            y_validation = y[validation_rows]
            metrics = _metrics(y_validation, predictions)
            operating_metrics = _metrics(
                y_validation[validation_operating],
                predictions[validation_operating],
            )
            rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "full_rmse": metrics["rmse"],
                    "full_mae": metrics["mae"],
                    "operating_rmse": operating_metrics["rmse"],
                    "operating_mae": operating_metrics["mae"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    X, y, dates, operating = load_frame()
    print(f"Rows: {len(X)}; operating: {int(operating.sum())}; non-operating: {int((~operating).sum())}")
    print(f"Non-operating rows with non-zero target: {int((y[~operating] != 0).sum())}")
    print(f"Features: {X.shape[1]}")

    splits = [random_split(dates), chronological_split(dates)]
    model_names = ["baseline", "linear", "linear-log", "hist-gradient-boosting"]

    result_rows = []
    for split in splits:
        print(
            f"\n{split.name}: train={len(split.train)}, "
            f"validation={len(split.validation)}, test={len(split.test)}"
        )
        for model_name in model_names:
            result_rows.append(evaluate_split(split, X, y, operating, model_name))

    result = pd.DataFrame(result_rows)
    display_columns = [
        "split",
        "model",
        "test_rows",
        "test_operating",
        "test_non_operating",
        "full_rmse",
        "full_mae",
        "full_bias",
        "full_r2",
        "operating_rmse",
        "operating_mae",
        "operating_bias",
        "negative_predictions",
    ]
    print("\nHoldout comparison:")
    print(result[display_columns].to_string(index=False, float_format=lambda value: f"{value:,.3f}"))

    cv = rolling_cv(X, y, dates, operating, ["linear", "linear-log", "hist-gradient-boosting"])
    cv_summary = (
        cv.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            full_rmse=("full_rmse", "mean"),
            full_mae=("full_mae", "mean"),
            operating_rmse=("operating_rmse", "mean"),
            operating_mae=("operating_mae", "mean"),
        )
        .sort_values("operating_rmse")
    )
    print("\nRolling TimeSeriesSplit (mean over 5 folds):")
    print(cv_summary.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))


if __name__ == "__main__":
    main()
