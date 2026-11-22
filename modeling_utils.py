import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def split_data(X, y, random_state=42):
    """Разделяет данные на 60% train, 20% validation и 20% test."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        shuffle=True,
        random_state=random_state,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        shuffle=True,
        random_state=random_state,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )

def train_linear_model(
    X_train,
    y_train,
    X_evaluation,
    log_target=False,
    clip=False,
):
    """Обучает LinearRegression и возвращает модель и прогнозы."""
    model = LinearRegression()

    target = (
        np.log1p(y_train)
        if log_target
        else y_train
    )

    model.fit(X_train, target)
    predictions = model.predict(X_evaluation)

    if log_target:
        predictions = np.expm1(predictions)

    if clip:
        predictions = np.clip(
            predictions,
            a_min=0,
            a_max=None,
        )

    return model, predictions

def calculate_metrics(
    y_true,
    predictions,
    peak_reference=None,
):
    """Считает метрики в исходных единицах целевой переменной."""
    actual = np.asarray(y_true)
    predictions = np.asarray(predictions)

    metrics = {
        "RMSE": np.sqrt(
            mean_squared_error(actual, predictions)
        ),
        "MAE": mean_absolute_error(actual, predictions),
        "Bias": np.mean(actual - predictions),
        "Negative": int((predictions < 0).sum()),
        "Min prediction": predictions.min(),
    }

    if peak_reference is not None:
        peak_threshold = np.quantile(
            peak_reference,
            0.90,
        )
        peak_mask = actual >= peak_threshold

        metrics["Peak RMSE"] = np.sqrt(
            mean_squared_error(
                actual[peak_mask],
                predictions[peak_mask],
            )
        )

    return metrics