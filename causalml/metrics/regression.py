from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

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