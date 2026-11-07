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
    if not filt.any():
        raise ValueError(
            "MAPE is undefined when every target is zero or smaller than EPS "
            f"({EPS}); no element survives the filter."
        )
    return np.mean(np.abs(1 - p[filt] / y[filt]))


def smape(y, p):
    """Symmetric Mean Absolute Percentage Error (sMAPE).
    Args:
        y (numpy.array): target
        p (numpy.array): prediction

    Returns:
        e (numpy.float64): sMAPE
    """
    denominator = np.abs(y) + np.abs(p)
    # Where the target and the prediction are both zero the prediction is exact,
    # and the 0 / 0 term is defined as 0 rather than propagating a NaN.
    ratio = np.divide(
        np.abs(y - p),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > EPS,
    )
    return 2.0 * np.mean(ratio)


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

    y = np.asarray(y, dtype=float)
    p = np.asarray(p)
    n_samples = y.shape[0]
    l_ones = np.linspace(1 / n_samples, 1, n_samples)

    def _gini_sorted_by(key):
        # sort y on key (from largest to smallest)
        order = np.argsort(-key, kind="stable")
        y_sorted = y[order]
        # rows with tied keys have no order between them, so give each
        # row in a tie the group's mean y; otherwise the result depends
        # on the input row order (e.g. a constant prediction scores 1.0)
        _, inverse, counts = np.unique(
            key[order], return_inverse=True, return_counts=True
        )
        y_sorted = np.bincount(inverse, weights=y_sorted)[inverse] / counts[inverse]

        # get Lorenz curve and Gini coefficient (area between curves)
        lorenz = np.cumsum(y_sorted) / np.sum(y_sorted)
        return np.sum(l_ones - lorenz)

    # normalize to true Gini coefficient
    return _gini_sorted_by(p) / _gini_sorted_by(y)


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
