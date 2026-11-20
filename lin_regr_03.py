import numpy as np

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