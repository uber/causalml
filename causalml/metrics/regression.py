import logging
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae  # noqa
from sklearn.metrics import r2_score  # noqa

from .const import EPS


logger = logging.getLogger("causalml")


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


def smape(y, p):
    """Symmetric Mean Absolute Percentage Error (sMAPE).
    Args:
        y (numpy.array): target
        p (numpy.array): prediction

    Returns:
        e (numpy.float64): sMAPE
    """
    return 2.0 * np.mean(np.abs(y - p) / (np.abs(y) + np.abs(p)))


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
    l_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    g_true = np.sum(l_ones - l_true)
    g_pred = np.sum(l_ones - l_pred)

    # normalize to true Gini coefficient
    return g_pred / g_true


def regression_metrics(
    y, p, w=None, metrics={"RMSE": rmse, "sMAPE": smape, "Gini": gini}
):
    """Log metrics for regressors.

    Args:
        y (numpy.array): target
        p (numpy.array): prediction
        w (numpy.array, optional): a treatment vector (1 or True: treatment, 0 or False: control). If given, log
            metrics for the treatment and control group separately
        metrics (dict, optional): a dictionary of the metric names and functions
    """
    assert metrics
    assert y.shape[0] == p.shape[0]

    for name, func in metrics.items():
        if w is not None:
            assert y.shape[0] == w.shape[0]
            if w.dtype != bool:
                w = w == 1
            logger.info("{:>8s}   (Control): {:10.4f}".format(name, func(y[~w], p[~w])))
            logger.info("{:>8s} (Treatment): {:10.4f}".format(name, func(y[w], p[w])))
        else:
            logger.info("{:>8s}: {:10.4f}".format(name, func(y, p)))
