from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae # noqa
from sklearn.metrics import r2_score    # noqa

import numpy as np

from .const import EPS


def ape(y, p):
    """Absolute Percentage Error (APE).
    Args:
        y (float): target
        p (float): prediction

    Returns:
        e (float): APE
    """

    assert np.abs(y) > EPS
    return np.abs(1 - p / y)


def mape(y, p):
    """Mean Absolute Percentage Error (MAPE).
    Args:
        y (numpy.array): target
        p (numpy.array): prediction

    Returns:
        e (numpy.float64): MAPE
    """

    filt = np.abs(y) > EPS
    return np.mean(np.abs(1 - p[filt] / y[filt]))


def rmse(y, p):
    """Root Mean Squared Error (RMSE).
    Args:
        y (numpy.array): target
        p (numpy.array): prediction

    Returns:
        e (numpy.float64): RMSE
    """

    # check and get number of samples
    assert y.shape == p.shape

    return np.sqrt(mse(y, p))


def gini(y, p):
    """Normalized Gini Coefficient.

    Args:
        y (numpy.array): target
        p (numpy.array): prediction

    Returns:
        e (numpy.float64): normalized Gini coefficient
    """

    # check and get number of samples
    assert y.shape == p.shape

    n_samples = y.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y, p]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    l_true = np.cumsum(true_order) / np.sum(true_order)
    l_pred = np.cumsum(pred_order) / np.sum(pred_order)
    l_ones = np.linspace(1/n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    g_true = np.sum(l_ones - l_true)
    g_pred = np.sum(l_ones - l_pred)

    # normalize to true Gini coefficient
    return g_pred / g_true
