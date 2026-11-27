import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..data.features import CALENDAR_CATEGORIES


class HourlyMedian(RegressorMixin, BaseEstimator):
    @staticmethod
    def _groups(features: pd.DataFrame) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays(
            [features["hour"], features["weekday"].ge(5)], names=["hour", "weekend"]
        )

    def fit(self, X: pd.DataFrame, y):
        target = pd.Series(np.asarray(y), index=self._groups(X))
        self.medians_ = target.groupby(level=["hour", "weekend"]).median()
        self.fallback_ = target.median()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        return self.medians_.reindex(self._groups(X)).fillna(self.fallback_).to_numpy()


class MedianResidualBoosting(RegressorMixin, BaseEstimator):
    """Predict the difference from the training median for the hour and day type."""

    def __init__(self, regressor: RegressorMixin):
        self.regressor = regressor

    def fit(self, X: pd.DataFrame, y):
        median = HourlyMedian().fit(X, y)
        residuals = np.asarray(y) - median.predict(X)
        self.regressor_ = clone(self.regressor).fit(X, residuals)
        self.median_ = median
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        return self.median_.predict(X) + self.regressor_.predict(X)


def make_model(regressor: RegressorMixin, weather: tuple[str, ...]) -> RegressorMixin:
    residual_model = isinstance(regressor, MedianResidualBoosting)
    estimator = regressor.regressor if residual_model else regressor
    needs_dense_input = isinstance(estimator, HistGradientBoostingRegressor)
    preprocessing = ColumnTransformer(
        [
            (
                "calendar",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not needs_dense_input),
                CALENDAR_CATEGORIES,
            ),
            ("numeric_calendar", "passthrough", ["is_holiday", "annual_sin", "annual_cos"]),
            ("weather", StandardScaler(), list(weather)),
        ]
    )
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", clone(estimator))])
    return MedianResidualBoosting(pipeline) if residual_model else pipeline
