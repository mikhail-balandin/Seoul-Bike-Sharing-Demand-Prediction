import numpy as np
import pandas as pd

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

    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_validation.shape}")
    print(f"Test: {X_test.shape}")


    baseline_predictions = np.full(
        shape=len(y_validation),
        fill_value=y_train.mean(),
    )

    raw_model, raw_predictions = train_linear_model(
        X_train=X_train,
        y_train=y_train,
        X_evaluation=X_validation,
    )

    _, raw_clipped_predictions = train_linear_model(
        X_train=X_train,
        y_train=y_train,
        X_evaluation=X_validation,
        clip=True,
    )

    log_model, log_predictions = train_linear_model(
        X_train=X_train,
        y_train=y_train,
        X_evaluation=X_validation,
        log_target=True,
    )

    model_predictions = {
        "Baseline": baseline_predictions,
        "Raw": raw_predictions,
        "Raw + clipping": raw_clipped_predictions,
        "Log target": log_predictions,
    }

    comparison_rows = []

    for model_name, predictions in model_predictions.items():
        metrics = calculate_metrics(
            y_true=y_validation,
            predictions=predictions,
            peak_reference=y_train,
        )

        comparison_rows.append(
            {
                "Model": model_name,
                **metrics,
            }
        )

    comparison = (
        pd.DataFrame(comparison_rows)
        .set_index("Model")
    )

    peak_threshold = y_train.quantile(0.90)
    validation_peak_count = (
        y_validation >= peak_threshold
    ).sum()

    print(f"\nPeak threshold: {peak_threshold:.2f}")
    print(
        f"Validation peak observations: "
        f"{validation_peak_count}"
    )

    print("\nValidation comparison:")
    print(comparison.round(2))

    print(f"\nRaw intercept: {raw_model.intercept_:.2f}")
    print(f"Log intercept: {log_model.intercept_:.2f}")

if __name__ == "__main__":
    main()