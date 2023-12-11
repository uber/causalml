from maq import MAQ, get_ipw_scores


def get_mq_ipw_scores(Y, W, W_hat=None):
    """Construct evaluation scores via inverse-propensity weighting.
    See https://github.com/grf-labs/maq/blob/master/python-package/maq/maq.py
    """

    return get_ipw_scores(Y, W, W_hat)


class MultiQini(MAQ):
    """Fit a Multi-Armed Qini.
    See https://github.com/grf-labs/maq/blob/master/python-package/maq/maq.py
    """

    pass
