"""Tests for the #854 argument-order deprecation shim.

CausalML learners historically take ``fit(X, treatment, y, ...)``, which puts
``y`` third and breaks ``sklearn.pipeline.Pipeline`` (it calls the final
estimator's ``fit(X, y)`` positionally). For scikit-learn compatibility the
positional order becomes ``(X, y, treatment, ...)`` in v1.0 — across the whole
package (meta-learners, causal/uplift trees and forests, IV, and the TF/Torch/
JAX estimators), and for ``predict`` as well as the fit family, so the same two
arguments never sit in opposite orders on one class.

This is a two-step deprecation: the positional order is UNCHANGED for now (so no
existing call silently breaks), but passing ``treatment``/``y`` positionally emits
a ``FutureWarning`` steering callers to keyword arguments, which are
order-independent and therefore safe across the v1.0 flip. These tests pin that
behavior; they should be updated (not deleted) when the signatures are reordered.
"""

import importlib
import inspect
import pkgutil
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from causalml.inference._arg_order import (
    SHIMMED_METHODS,
    _positional_params,
    v1_order,
)
from causalml.inference.iv import BaseDRIVRegressor, IVRegressor
from causalml.inference.meta import (
    BaseSRegressor,
    BaseTRegressor,
    BaseXRegressor,
    BaseRRegressor,
    BaseDRRegressor,
    XGBTRegressor,
    BaseSClassifier,
    BaseTClassifier,
    BaseXClassifier,
    BaseRClassifier,
)
from causalml.inference.tree import CausalTreeRegressor

from .const import RANDOM_SEED

# One representative regressor per meta-learner family; all default to
# ``control_name=0`` which matches the 0/1 treatment from ``synthetic_data``.
REGRESSORS = [
    BaseSRegressor,
    BaseTRegressor,
    BaseXRegressor,
    BaseRRegressor,
    BaseDRRegressor,
]

# The R-learner's ``predict(X, p, return_components)`` already omits treatment/y,
# so it is unaffected by the flip. Derive the list instead of hard-coding it, so
# it stays correct if a predict signature changes.
PREDICT_REORDERED = [
    cls
    for cls in REGRESSORS
    if "treatment"
    in _positional_params(getattr(cls.predict, "__wrapped__", cls.predict))
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


# --- v1.0 target ordering ---------------------------------------------------


@pytest.mark.parametrize(
    "current, expected",
    [
        # Meta-learners: treatment and y simply swap.
        (["X", "treatment", "y", "p"], ["X", "y", "treatment", "p"]),
        # predict: same rule, so fit and predict agree after the flip.
        (
            ["X", "treatment", "y", "p", "return_components", "verbose"],
            ["X", "y", "treatment", "p", "return_components", "verbose"],
        ),
        # IV: the instrument keeps its relative position and shifts right.
        (["X", "treatment", "y", "w"], ["X", "y", "treatment", "w"]),
        # DRIV: `assignment` lands fourth, after X, y and treatment.
        (
            ["X", "assignment", "treatment", "y", "p", "pZ"],
            ["X", "y", "treatment", "assignment", "p", "pZ"],
        ),
        # A signature with neither is left completely alone.
        (["X", "sample_weight"], ["X", "sample_weight"]),
    ],
)
def test_v1_order(current, expected):
    """X first, then y, then treatment; everything else keeps its relative order."""
    assert v1_order(current) == expected


# --- coverage across the whole package --------------------------------------


def _shimmable_methods():
    """Yield (qualname, method) for every learner method taking treatment and y.

    Walks the installed package so a newly added learner is picked up
    automatically. Modules whose optional backend (tf/torch/jax) is missing are
    skipped, so this covers them only in the backend CI lanes.
    """
    import causalml

    for info in pkgutil.walk_packages(causalml.__path__, prefix="causalml."):
        try:
            module = importlib.import_module(info.name)
        except Exception:  # optional backend absent, or an import-time failure
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not cls.__module__.startswith("causalml."):
                continue
            for name in SHIMMED_METHODS:
                method = cls.__dict__.get(name)
                if not callable(method) or getattr(
                    method, "__isabstractmethod__", False
                ):
                    continue
                params = _positional_params(getattr(method, "__wrapped__", method))
                if "treatment" in params and "y" in params:
                    yield f"{cls.__module__}.{cls.__qualname__}.{name}", method


def test_every_treatment_y_method_is_shimmed():
    """Completeness guard for the deprecation window.

    A deprecation window is one-shot: anything whose positional order changes at
    v1.0 must warn in this release or it needs a second cycle. This fails if a
    learner method takes ``treatment`` and ``y`` positionally without the shim.
    """
    unshimmed = [
        qualname
        for qualname, method in _shimmable_methods()
        if not getattr(method, "_arg_order_shimmed", False)
    ]
    assert unshimmed == [], f"missing #854 shim on: {unshimmed}"


def test_shim_covers_more_than_the_meta_learners():
    """Guards against the hook silently regressing to BaseLearner-only scope."""
    modules = {qualname.rsplit(".", 2)[0] for qualname, _ in _shimmable_methods()}
    assert any("inference.tree" in m for m in modules), modules
    assert any("inference.iv" in m for m in modules), modules


# --- behaviour outside the meta-learner family ------------------------------


def test_causal_tree_fit_warns_positionally(generate_regression_data):
    """The trees inherit sklearn estimator bases, so they need the flip too."""
    y, X, treatment, _, _, _ = generate_regression_data()
    with pytest.warns(FutureWarning, match="argument order"):
        CausalTreeRegressor().fit(X, treatment, y)


def test_causal_tree_fit_keyword_is_silent(generate_regression_data):
    y, X, treatment, _, _, _ = generate_regression_data()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        CausalTreeRegressor().fit(X=X, y=y, treatment=treatment)
    assert _order_warnings(record) == []


def test_iv_regressor_fit_warns_positionally():
    """IVRegressor's message must name its own signature, instrument included."""
    rng = np.random.RandomState(RANDOM_SEED)
    n = 200
    w = rng.binomial(1, 0.5, n).astype(float)
    treatment = w * rng.binomial(1, 0.8, n)
    X = rng.normal(size=(n, 2))
    y = treatment + X[:, 0] + rng.normal(size=n)
    with pytest.warns(FutureWarning, match=r"becomes \(X, y, treatment, w\)"):
        IVRegressor().fit(X, treatment, y, w)


def test_driv_message_puts_assignment_fourth():
    """Pins the v1.0 order chosen for DRIV: (X, y, treatment, assignment, ...)."""
    params = _positional_params(BaseDRIVRegressor.fit.__wrapped__)
    assert v1_order(params)[:4] == ["X", "y", "treatment", "assignment"]


# --- predict joins the deprecation ------------------------------------------


@pytest.mark.parametrize("learner_cls", PREDICT_REORDERED)
def test_predict_warns_positionally(generate_regression_data, learner_cls):
    """After the flip fit and predict must agree, so predict warns too."""
    y, X, treatment, _, _, _ = generate_regression_data()
    learner = learner_cls(learner=LinearRegression())
    learner.fit(X=X, y=y, treatment=treatment)
    with pytest.warns(FutureWarning, match="argument order"):
        learner.predict(X, treatment, y)


@pytest.mark.parametrize("learner_cls", PREDICT_REORDERED)
def test_predict_keyword_is_silent(generate_regression_data, learner_cls):
    y, X, treatment, _, _, _ = generate_regression_data()
    learner = learner_cls(learner=LinearRegression())
    learner.fit(X=X, y=y, treatment=treatment)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        learner.predict(X, treatment=treatment, y=y)
    assert _order_warnings(record) == []


def test_fit_predict_warns_once_across_fit_and_predict(generate_regression_data):
    """fit_predict delegates to both fit and predict; still one warning."""
    y, X, treatment, _, _, _ = generate_regression_data()
    learner = BaseTRegressor(learner=LinearRegression())
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        learner.fit_predict(X, treatment, y)
    assert len(_order_warnings(record)) == 1


# --- classifier fit overrides are wrapped independently ---------------------


@pytest.mark.parametrize(
    "learner_cls", [BaseSClassifier, BaseTClassifier, BaseXClassifier, BaseRClassifier]
)
def test_classifier_fit_is_shimmed(learner_cls):
    """The classifier variants define their own fit, so each is wrapped separately."""
    assert getattr(learner_cls.fit, "_arg_order_shimmed", False)
    assert _positional_params(learner_cls.fit)[:3] == ["X", "treatment", "y"]
