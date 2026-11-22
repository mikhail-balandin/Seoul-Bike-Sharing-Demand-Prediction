import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from feature_engineering_02 import X, y
from modeling_utils import (
    calculate_metrics,
    split_data,
    train_linear_model,
)

def main():
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    # Train + validation используем для кросс-валидации
    X_development = pd.concat(
        [X_train, X_validation]
    )

    y_development = pd.concat(
        [y_train, y_validation]
    )

    print(f"Development: {X_development.shape}")
    print(f"Reserved test: {X_test.shape}")

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    experiment_rows = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        cv.split(X_development),
        start=1,
    ):
        X_train_fold = X_development.iloc[train_indices]
        X_validation_fold = X_development.iloc[
            validation_indices
        ]

        y_train_fold = y_development.iloc[train_indices]
        y_validation_fold = y_development.iloc[
            validation_indices
        ]

        # среднее по train-фолду
        baseline_predictions = np.full(
            shape=len(y_validation_fold),
            fill_value=y_train_fold.mean(),
        )

        # LinearRegression + clipping
        _, raw_clipped_predictions = train_linear_model(
            X_train=X_train_fold,
            y_train=y_train_fold,
            X_evaluation=X_validation_fold,
            clip=True,
        )

        # LinearRegression. log1p(y)
        _, log_predictions = train_linear_model(
            X_train=X_train_fold,
            y_train=y_train_fold,
            X_evaluation=X_validation_fold,
            log_target=True,
        )

        # HistGradientBoosting
        boosting_model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=42,
        )

        boosting_model.fit(
            X_train_fold,
            y_train_fold,
        )

        boosting_predictions = boosting_model.predict(
            X_validation_fold
        )

        boosting_clipped_predictions = np.clip(
            boosting_predictions,
            a_min=0,
            a_max=None,
        )

        fold_predictions = {
            "Baseline": baseline_predictions,
            "Linear raw + clipping": raw_clipped_predictions,
            "Linear log target": log_predictions,
            "HistGradientBoosting": boosting_predictions,
            "HistGradientBoosting + clipping": (
                boosting_clipped_predictions
            ),
        }

        for model_name, predictions in fold_predictions.items():
            metrics = calculate_metrics(
                y_true=y_validation_fold,
                predictions=predictions,
                peak_reference=y_train_fold,
            )

            experiment_rows.append(
                {
                    "Fold": fold_number,
                    "Model": model_name,
                    **metrics,
                }
            )

    cv_results = pd.DataFrame(experiment_rows)

    print("\nResults by fold:")
    print(
        cv_results.round(2).to_string(index=False)
    )

    # Сводная таблица: среднее и разброс между фолдами
    summary = (
        cv_results
        .groupby("Model")
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            Bias_mean=("Bias", "mean"),
            Peak_RMSE_mean=("Peak RMSE", "mean"),
            Negative_mean=("Negative", "mean"),
            Min_prediction=("Min prediction", "min"),
        )
        .sort_values("RMSE_mean")
    )

    print("\nCross-validation summary:")
    print(summary.round(2))

if __name__ == "__main__":
    main()