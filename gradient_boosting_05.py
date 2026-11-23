import numpy as np
import pandas as pd

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import HistGradientBoostingRegressor

from feature_engineering_02 import X, y, operating_df
from modeling_utils import (
    calculate_metrics,
    split_data,
)

def summarize_errors(data, group_column):
    """Считает RMSE, MAE и bias по группе наблюдений."""
    summary = data.groupby(group_column).agg(
        Observations=("residual", "size"),
        MeanSquaredError=("squared_error", "mean"),
        MAE=("abs_error", "mean"),
        Bias=("residual", "mean"),
    )

    summary["RMSE"] = np.sqrt(
        summary["MeanSquaredError"]
    )

    return summary[
        ["Observations", "RMSE", "MAE", "Bias"]
    ]

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

    final_model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=42,
    )

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

    diagnostic_columns = [
        "Date",
        "Hour",
        "Seasons",
        "Temperature(°C)",
        "Rainfall(mm)",
        "Snowfall (cm)",
    ]

    error_results = operating_df.loc[
        X_test.index,
        diagnostic_columns,
    ].copy()

    error_results["actual"] = y_test.to_numpy()
    error_results["prediction"] = final_test_predictions

    error_results["residual"] = (
        error_results["actual"]
        - error_results["prediction"]
    )

    error_results["abs_error"] = (
        error_results["residual"].abs()
    )

    error_results["squared_error"] = (
        error_results["residual"] ** 2
    )

    # Ошибки по часам и сезонам
    hourly_errors = summarize_errors(
        error_results,
        "Hour",
    )

    seasonal_errors = summarize_errors(
        error_results,
        "Seasons",
    )

    seasonal_errors = seasonal_errors.reindex(
        ["Winter", "Spring", "Summer", "Autumn"]
    )

    print("\nErrors by hour:")
    print(hourly_errors.round(2))

    print("\nErrors by season:")
    print(seasonal_errors.round(2))

    error_results["Demand level"] = np.where(
        error_results["actual"] >= peak_threshold,
        "Peak",
        "Regular",
    )

    demand_errors = summarize_errors(
        error_results,
        "Demand level",
    ).reindex(["Regular", "Peak"])

    print("\nErrors by demand level:")
    print(demand_errors.round(2))

    largest_errors = error_results.nlargest(
        n=10,
        columns="abs_error",
    )

    print("\nLargest absolute errors:")
    print(
        largest_errors[
            [
                "Date",
                "Hour",
                "Seasons",
                "actual",
                "prediction",
                "residual",
                "abs_error",
            ]
        ]
        .round(
            {
                "prediction": 2,
                "residual": 2,
                "abs_error": 2,
            }
        )
        .to_string(index=False)
    )

    figures_dir = (
        Path(__file__).resolve().parent / "figures"
    )
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # фактические значения и прогнозы
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.scatterplot(
        data=error_results,
        x="actual",
        y="prediction",
        s=20,
        alpha=0.35,
        ax=ax,
    )

    upper_limit = max(
        error_results["actual"].max(),
        error_results["prediction"].max(),
    )

    ax.plot(
        [0, upper_limit],
        [0, upper_limit],
        color="red",
        linestyle="--",
        label="Ideal prediction",
    )

    ax.set(
        xlabel="Actual rentals",
        ylabel="Predicted rentals",
        title="Final model: actual vs predicted",
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        figures_dir / "08_final_actual_vs_predicted.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    # остатки и прогнозы
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=error_results,
        x="prediction",
        y="residual",
        s=20,
        alpha=0.35,
        ax=ax,
    )

    ax.axhline(
        y=0,
        color="red",
        linestyle="--",
    )

    ax.set(
        xlabel="Predicted rentals",
        ylabel="Residual: actual - prediction",
        title="Final model residuals",
    )

    fig.tight_layout()
    fig.savefig(
        figures_dir / "09_final_residuals.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    # RMSE по часам и сезонам
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 5),
    )

    sns.barplot(
        data=hourly_errors.reset_index(),
        x="Hour",
        y="RMSE",
        color="#15668b",
        ax=axes[0],
    )
    axes[0].set_title("RMSE by hour")

    sns.barplot(
        data=seasonal_errors.reset_index(),
        x="Seasons",
        y="RMSE",
        order=["Winter", "Spring", "Summer", "Autumn"],
        color="#15668b",
        ax=axes[1],
    )
    axes[1].set_title("RMSE by season")

    fig.tight_layout()
    fig.savefig(
        figures_dir / "10_final_errors_by_time.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()