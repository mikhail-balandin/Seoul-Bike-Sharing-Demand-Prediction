import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from feature_engineering_02 import X, operating_df, y

# Гарантируем хронологический порядок наблюдений
chronological_index = (
    operating_df
    .sort_values(["Date", "Hour"])
    .index
)

X = X.loc[chronological_index]
y = y.loc[chronological_index]
operating_df = operating_df.loc[chronological_index]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)


print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# прогнозируем среднее из train
baseline_predictions = np.full(
    shape=len(y_test),
    fill_value=y_train.mean(),
)

# Линейная регрессия
model = LinearRegression()
model.fit(X_train, y_train)

linear_predictions = model.predict(X_test)

def print_metrics(model_name, y_true, predictions):
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)

    print(f"\n{model_name}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

print_metrics("Baseline: train mean", y_test, baseline_predictions)
print_metrics("Linear regression", y_test, linear_predictions)

print(f"\nIntercept: {model.intercept_:.2f}")
print(
    f"Negative predictions: "
    f"{(linear_predictions < 0).sum()}"
)
print(
    f"Minimum prediction: "
   f"{linear_predictions.min():.2f}"
)


# 2 Этап. 

# Таблица для анализа ошибок
diagnostic_columns = [
    "Date",
    "Hour",
    "Seasons",
    "Temperature(°C)",
    "Humidity(%)",
    "Rainfall(mm)",
    "Snowfall (cm)",
]

results = operating_df.loc[
    X_test.index,
    diagnostic_columns,
].copy()

# to_numpy() сохраняет порядок строк X_test/y_test
results["actual"] = y_test.to_numpy()
results["prediction"] = linear_predictions

results["residual"] = (
    results["actual"] - results["prediction"]
)
results["abs_error"] = results["residual"].abs()

# Анализ отрицательных прогнозов
negative_predictions = results.loc[
    results["prediction"] < 0
]

negative_share = (
    len(negative_predictions) / len(results)
)

print(f"\nNegative prediction share: {negative_share:.1%}")

print("\nActual rentals for negative predictions:")
print(negative_predictions["actual"].describe())

print("\nNegative predictions by season:")
print(negative_predictions["Seasons"].value_counts())

print("\nNegative predictions by hour:")
print(
    negative_predictions["Hour"]
    .value_counts()
    .head(10)
)

print("\nMean weather values for negative predictions:")
print(
    negative_predictions[
        [
            "Temperature(°C)",
            "Humidity(%)",
            "Rainfall(mm)",
            "Snowfall (cm)",
        ]
    ]
    .mean()
    .round(2)
)

# Сравнение с обрезкой прогнозов нулём
clipped_predictions = np.clip(
    linear_predictions,
    a_min=0,
    a_max=None,
)

print_metrics(
    "Clipped linear regression",
    y_test,
    clipped_predictions,
)

# Наблюдения с наибольшими абсолютными ошибками
largest_errors = results.nlargest(
    n=10,
    columns="abs_error",
)

print("\nLargest absolute errors:")
print(
    largest_errors[
        [
            "Date",
            "Hour",
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

# 3 этап
# добавим валидацию

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    shuffle=True,
    random_state=42,
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    shuffle=True,
    random_state=42,
)

print(f"Train: {X_train.shape}")
print(f"Validation: {X_validation.shape}")
print(f"Test: {X_test.shape}")


baseline_predictions = np.full(
    shape=len(y_validation),
    fill_value=y_train.mean(),
)

# Модель на исходной цели
raw_model = LinearRegression()
raw_model.fit(X_train, y_train)

raw_predictions = raw_model.predict(X_validation)

raw_clipped_predictions = np.clip(
    raw_predictions,
    a_min=0,
    a_max=None,
)

# Модель на log1p(y)
log_model = LinearRegression()
log_model.fit(
    X_train,
    np.log1p(y_train),
)

log_predictions = np.expm1(
    log_model.predict(X_validation)
)

# Порог пикового спроса определяем только по train
peak_threshold = y_train.quantile(0.90)

actual = y_validation.to_numpy()
peak_mask = actual >= peak_threshold

def calculate_metrics(model_name, predictions):
    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    mae = mean_absolute_error(
        actual,
        predictions,
    )

    # положительный bias — средний underfitting
    # отрицательный bias — средний overfitting
    bias = np.mean(actual - predictions)

    peak_rmse = np.sqrt(
        mean_squared_error(
            actual[peak_mask],
            predictions[peak_mask],
        )
    )

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "Peak RMSE": peak_rmse,
        "Negative": int((predictions < 0).sum()),
        "Min prediction": predictions.min(),
    }

comparison = pd.DataFrame(
    [
        calculate_metrics(
            "Baseline",
            baseline_predictions,
        ),
        calculate_metrics(
            "Raw",
            raw_predictions,
        ),
        calculate_metrics(
            "Raw + clipping",
            raw_clipped_predictions,
        ),
        calculate_metrics(
            "Log target",
            log_predictions,
        ),
    ]
).set_index("Model")

print(f"\nPeak threshold: {peak_threshold:.2f}")
print(f"Validation peak observations: {peak_mask.sum()}")

print("\nValidation comparison:")
print(comparison.round(2))
