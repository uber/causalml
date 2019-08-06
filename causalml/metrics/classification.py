from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import log_loss, roc_auc_score

from .const import EPS


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
