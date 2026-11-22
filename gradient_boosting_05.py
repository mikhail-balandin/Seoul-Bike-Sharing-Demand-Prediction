import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

from feature_engineering_02 import X, y
from modeling_utils import (
    calculate_metrics,
    split_data,
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

    X_development = pd.concat(
        [X_train, X_validation]
    )

    y_development = pd.concat(
        [y_train, y_validation]
    )

    print(f"Development: {X_development.shape}")
    print(f"Test: {X_test.shape}")

    # Параметры полностью совпадают с кросс-валидацией
    final_model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=42,
    )

    # Test в обучении не участвует
    final_model.fit(
        X_development,
        y_development,
    )

    raw_test_predictions = final_model.predict(X_test)

    final_test_predictions = np.clip(
        raw_test_predictions,
        a_min=0,
        a_max=None,
    )

    final_metrics = calculate_metrics(
        y_true=y_test,
        predictions=final_test_predictions,
        peak_reference=y_development,
    )

    print("\nFINAL TEST: HistGradientBoosting + clipping")

    for metric_name, value in final_metrics.items():
        print(f"{metric_name}: {value:.2f}")

    print(
        f"\nNegative predictions before clipping: "
        f"{(raw_test_predictions < 0).sum()}"
    )
    print(
        f"Minimum prediction before clipping: "
        f"{raw_test_predictions.min():.2f}"
    )

    peak_threshold = y_development.quantile(0.90)
    test_peak_count = (
        y_test >= peak_threshold
    ).sum()

    print(f"\nPeak threshold: {peak_threshold:.2f}")
    print(f"Test peak observations: {test_peak_count}")

if __name__ == "__main__":
    main()