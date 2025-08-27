"""Implementation of statistical models."""

from abc import ABC, abstractmethod

import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import QuantileTransformer


def _gauss_transform(x: np.ndarray):
    """Transform x to be (marginally) gaussian."""
    assert x.ndim == 2
    qt = QuantileTransformer(n_quantiles=len(x), output_distribution="normal")
    qt.fit(x)
    return qt.transform(x), qt


class PredictionModel(ABC):
    """Abstract base class for prediction models."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        assert isinstance(n_features, int) and n_features > 0
        assert isinstance(n_targets, int) and n_targets > 0

        self.n_features = n_features
        self.n_targets = n_targets
        self.trained = False  # whether model has been trained yet.

    def fit(self, x, y):
        """Fit model using training data."""
        assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)) == 0
        assert x.shape == (y.shape[0], self.n_features)
        assert y.shape == (x.shape[0], self.n_targets)
        self._fit(x, y)
        self.trained = True

    def predict(self, x):
        """Predict y given x."""
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained
        return self._predict(x).reshape(x.shape[0], self.n_targets)

    @abstractmethod
    def _fit(self, x, y):
        pass

    @abstractmethod
    def _predict(self, x):
        pass


class MultiCAM(PredictionModel):
    """MultiCAM model described in our first paper."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        super().__init__(n_features, n_targets)

        # additional metadata that needs to be saved for prediction.
        self.qt_x = None
        self.qt_y = None
        self.qt_pred = None
        self.x_train = None
        self.y_train = None

        # setup linear regression model
        self.reg = linear_model.LinearRegression()

    def _fit(self, x, y) -> None:
        """Fit model using training data"""
        assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)) == 0
        assert x.shape == (y.shape[0], self.n_features)
        assert y.shape == (x.shape[0], self.n_targets)

        # need to save training data to predict from ranks later.
        self.x_train = x.copy()
        self.y_train = y.copy()

        # transform ranks to be (marginally) gaussian.
        x_gauss, self.qt_x = _gauss_transform(x)
        y_gauss, self.qt_y = _gauss_transform(y)

        # then fit a linear regression model to the transformed data.
        self.reg.fit(x_gauss, y_gauss)

        # get quantile transformer of prediction to (marginal) normal using training data.
        y_pred = self.reg.predict(x_gauss)
        self.qt_pred = QuantileTransformer(
            n_quantiles=len(y_pred), output_distribution="normal"
        )
        self.qt_pred.fit(y_pred)

    def _predict(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained

        # transform ranks to be (marginally) gaussian.
        x_gauss = self.qt_x.transform(x)

        # predict on transformed ranks.
        y_not_gauss = self.reg.predict(x_gauss)

        # get quantile transformer of prediction to (marginal) normal.
        y_gauss = self.qt_pred.transform(y_not_gauss)
        y_pred = self.qt_y.inverse_transform(y_gauss)
        return y_pred
