import logging
from sklearn.metrics import log_loss, roc_auc_score

from .const import EPS
from .regression import regression_metrics


logger = logging.getLogger("causalml")


def logloss(y, p):
    """Bounded log loss error.
    Args:
        y (numpy.array): target
        p (numpy.array): prediction
    Returns:
        bounded log loss error
    """

    p[p < EPS] = EPS
    p[p > 1 - EPS] = 1 - EPS
    return log_loss(y, p)


def classification_metrics(
    y, p, w=None, metrics={"AUC": roc_auc_score, "Log Loss": logloss}
):
    """Log metrics for classifiers.

    Args:
        y (numpy.array): target
        p (numpy.array): prediction
        w (numpy.array, optional): a treatment vector (1 or True: treatment, 0 or False: control). If given, log
            metrics for the treatment and control group separately
        metrics (dict, optional): a dictionary of the metric names and functions
    """
    regression_metrics(y=y, p=p, w=w, metrics=metrics)
