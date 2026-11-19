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

# Простой baseline: всегда прогнозируем среднее из train
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