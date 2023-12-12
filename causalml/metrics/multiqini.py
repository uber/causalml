from maq import MAQ, get_ipw_scores

#  See https://github.com/grf-labs/maq/blob/master/python-package/maq/maq.py


def get_mq_ipw_scores(Y, W, W_hat=None):
    return get_ipw_scores(Y, W, W_hat)


get_mq_ipw_scores.__doc__ = get_ipw_scores.__doc__


class MultiQini(MAQ):
    pass


MultiQini.__doc__ = MAQ.__doc__
