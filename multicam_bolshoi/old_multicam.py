"""Implementation of statistical models."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import rankdata
from sklearn import linear_model
from sklearn.preprocessing import QuantileTransformer


def _value_at_rank(x, ranks):
    """Get value at ranks of multidimensional array."""
    assert x.shape[1] == ranks.shape[1]
    assert ranks.dtype == int
    n, m = ranks.shape
    y = np.zeros((n, m), dtype=float)
    for ii in range(m):
        y[:, ii] = np.take(x[:, ii], ranks[:, ii])
    return y


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
        self.qt_xr = None
        self.qt_yr = None
        self.qt_pred = None
        self.rank_lookup = {}
        self.x_train = None
        self.y_train = None

        # setup linear regression model
        self.reg = linear_model.LinearRegression()

    def _fit(self, x, y):
        """Fit model using training data"""
        assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)) == 0
        assert x.shape == (y.shape[0], self.n_features)
        assert y.shape == (x.shape[0], self.n_targets)

        # need to save training data to predict from ranks later.
        self.x_train = x.copy()
        self.y_train = y.copy()

        # first get ranks of features and targets.
        xr = rankdata(x, axis=0, method="ordinal")
        yr = rankdata(y, axis=0, method="ordinal")

        # transform ranks to be (marginally) gaussian.
        x_gauss, self.qt_xr = _gauss_transform(xr)
        y_gauss, self.qt_yr = _gauss_transform(yr)

        # then fit a linear regression model to the transformed data.
        self.reg.fit(x_gauss, y_gauss)

        # get quantile transformer of prediction to (marginal) normal using training data.
        y_pred = self.reg.predict(x_gauss)
        self.qt_pred = QuantileTransformer(
            n_quantiles=len(y_pred), output_distribution="normal"
        )
        self.qt_pred.fit(y_pred)

        # finally, create lookup table for low and high ranks of each feature.
        for jj in range(self.n_features):
            x_train_jj = np.sort(self.x_train[:, jj])
            u, c = np.unique(x_train_jj, return_counts=True)
            lranks = np.cumsum(c) - c + 1
            hranks = np.cumsum(c)
            self.rank_lookup[jj] = (u, lranks, hranks)

        return x_gauss, y_gauss

    def _get_ranks(self, x, mode="middle"):
        assert mode in {"middle", "random"}

        # get ranks of test data (based on training data)
        xr = np.zeros_like(x) * np.nan
        for jj in range(self.n_features):
            x_jj = x[:, jj]
            x_train_jj = np.sort(self.x_train[:, jj])
            uniq, lranks, hranks = self.rank_lookup[jj]
            xr[:, jj] = np.searchsorted(x_train_jj, x_jj) + 1  # indices to ranks

            # if value is in training data, get middle or random rank
            in_train = np.isin(x_jj, uniq)
            u_indices = np.searchsorted(uniq, x_jj[in_train])
            lr, hr = lranks[u_indices], hranks[u_indices]  # repeat appropriately
            xr[in_train, jj] = (
                np.random.randint(lr, hr + 1) if mode == "random" else (lr + hr) / 2
            )

        assert np.sum(np.isnan(xr)) == 0

        return xr

    def _predict(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained

        xr = self._get_ranks(x, mode="middle")

        # transform ranks to be (marginally) gaussian.
        x_gauss = self.qt_xr.transform(xr)

        # predict on transformed ranks.
        y_not_gauss = self.reg.predict(x_gauss)

        # get quantile transformer of prediction to (marginal) normal.
        y_gauss = self.qt_pred.transform(y_not_gauss)
        yr = self.qt_yr.inverse_transform(y_gauss).astype(int)
        yr -= 1  # ranks are 1-indexed, so subtract 1 to get 0-indexed.

        # predictions are points in train data corresponding to ranks predicted
        y_train_sorted = np.sort(self.y_train, axis=0)
        y_pred = _value_at_rank(y_train_sorted, yr)

        return y_pred
