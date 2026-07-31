import inspect

import numpy as np

from causalml.inference.meta.utils import get_weighted_variance
from causalml.inference.tree import CausalTreeRegressor
from causalml.inference.tree.utils import timeit


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


def test_timeit_preserves_the_wrapped_signature():
    """``timeit`` must not erase the identity of what it decorates.

    Without ``functools.wraps`` the returned closure reports its own
    ``(*args, **kw)`` signature and the name ``"wrapped"``, which breaks
    anything that introspects the method — ``help()``, Sphinx autodoc, and
    scikit-learn's parameter discovery among them.
    """

    @timeit()
    def example(X, treatment, y, sample_weight=None):
        """Docstring that must survive decoration."""
        return X

    assert example.__name__ == "example"
    assert example.__doc__ == "Docstring that must survive decoration."
    assert list(inspect.signature(example).parameters) == [
        "X",
        "treatment",
        "y",
        "sample_weight",
    ]


def test_timeit_preserves_bootstrap_pool_signature():
    """``CausalTreeRegressor.bootstrap_pool`` is the decorator's only caller."""
    assert CausalTreeRegressor.bootstrap_pool.__name__ == "bootstrap_pool"
    params = list(inspect.signature(CausalTreeRegressor.bootstrap_pool).parameters)
    assert params[:4] == ["self", "X", "treatment", "y"]


def test_timeit_still_returns_the_wrapped_result():
    """Guard the decorator's actual job while changing its metadata handling."""

    @timeit(exclude_kwargs=("secret",))
    def add(a, b, secret=None):
        return a + b

    assert add(2, 3, secret="hidden") == 5
