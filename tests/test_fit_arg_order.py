"""Tests for the #854 fit-family argument-order deprecation shim.

CausalML meta-learners historically take ``fit(X, treatment, y, ...)``, which puts
``y`` third and breaks ``sklearn.pipeline.Pipeline`` (it calls the final
estimator's ``fit(X, y)`` positionally). For scikit-learn compatibility the
positional order becomes ``fit(X, y, treatment, ...)`` in v1.0.

This is a two-step deprecation: the positional order is UNCHANGED for now (so no
existing call silently breaks), but passing ``treatment``/``y`` positionally emits
a ``FutureWarning`` steering callers to keyword arguments, which are
order-independent and therefore safe across the v1.0 flip. These tests pin that
behavior; they should be updated (not deleted) when the signatures are reordered.
"""

import inspect
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from causalml.inference.meta import (
    BaseSRegressor,
    BaseTRegressor,
    BaseXRegressor,
    BaseRRegressor,
    BaseDRRegressor,
    XGBTRegressor,
)

# One representative regressor per meta-learner family; all default to
# ``control_name=0`` which matches the 0/1 treatment from ``synthetic_data``.
REGRESSORS = [
    BaseSRegressor,
    BaseTRegressor,
    BaseXRegressor,
    BaseRRegressor,
    BaseDRRegressor,
]


def _order_warnings(record):
    """Filter a warning record down to the #854 arg-order FutureWarnings."""
    return [
        w
        for w in record
        if issubclass(w.category, FutureWarning) and "argument order" in str(w.message)
    ]


@pytest.mark.parametrize("learner_cls", REGRESSORS)
def test_positional_fit_warns(generate_regression_data, learner_cls):
    y, X, treatment, _, _, _ = generate_regression_data()
    learner = learner_cls(learner=LinearRegression())
    with pytest.warns(FutureWarning, match="argument order"):
        learner.fit(X, treatment, y)


@pytest.mark.parametrize("learner_cls", REGRESSORS)
def test_keyword_fit_does_not_warn(generate_regression_data, learner_cls):
    y, X, treatment, _, _, _ = generate_regression_data()
    learner = learner_cls(learner=LinearRegression())
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        learner.fit(X=X, y=y, treatment=treatment)
    assert _order_warnings(record) == []


def test_positional_and_keyword_are_equivalent(generate_regression_data):
    """The shim must never silently swap y/treatment: both call styles must fit
    identically. Guards against a values-based reorder that could corrupt data."""
    y, X, treatment, _, _, _ = generate_regression_data()

    positional = BaseTRegressor(learner=LinearRegression())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        positional.fit(X, treatment, y)

    keyword = BaseTRegressor(learner=LinearRegression())
    keyword.fit(X=X, y=y, treatment=treatment)

    np.testing.assert_allclose(positional.predict(X), keyword.predict(X))


def test_fit_predict_warns_once(generate_regression_data):
    """fit_predict -> fit delegation must not double-count the warning."""
    y, X, treatment, _, _, _ = generate_regression_data()
    learner = BaseTRegressor(learner=LinearRegression())
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        learner.fit_predict(X, treatment, y)
    assert len(_order_warnings(record)) == 1


def test_estimate_ate_warns_once(generate_regression_data):
    """estimate_ate -> (fit_predict ->) fit delegation must warn exactly once."""
    y, X, treatment, _, _, _ = generate_regression_data()
    learner = BaseSRegressor(learner=LinearRegression())
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        learner.estimate_ate(X, treatment, y)
    assert len(_order_warnings(record)) == 1


def test_subclass_super_fit_warns_once(generate_regression_data):
    """A subclass fit that delegates to super().fit (XGBTRegressor) warns once."""
    y, X, treatment, _, _, _ = generate_regression_data()
    learner = XGBTRegressor()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        learner.fit(X, treatment, y)
    assert len(_order_warnings(record)) == 1


@pytest.mark.parametrize("learner_cls", REGRESSORS)
def test_fit_signature_is_preserved(learner_cls):
    """functools.wraps keeps the real (X, treatment, y, ...) signature visible."""
    learner = learner_cls(learner=LinearRegression())
    params = list(inspect.signature(learner.fit).parameters)
    assert params[:3] == ["X", "treatment", "y"]
