import numpy as np
from causalml.inference.meta.utils import get_weighted_variance


def test_weighted_variance():
    x = np.array([1, 2, 3, 4, 5])
    sample_weight_equal = np.ones(len(x))

    var_x = get_weighted_variance(x, sample_weight_equal)
    # should get the same variance with equal sample_weight
    assert var_x == x.var()

    x1 = np.array([1, 2, 3, 4, 4, 5, 5])
    sample_weight_equal = np.ones(len(x1))
    sample_weight = [1, 1, 1, 2, 2]
    var_x2 = get_weighted_variance(x, sample_weight)
    var_x1 = get_weighted_variance(x1, sample_weight_equal)

    # should get the same variance by duplicate the observation based on the sample weight
    assert var_x1 == var_x2
