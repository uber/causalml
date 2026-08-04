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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from causalml.inference._arg_order import (
    MIGRATION_GUIDE_URL,
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
from causalml.inference.tree import (
    CausalRandomForestRegressor,
    CausalTreeRegressor,
    UpliftRandomForestClassifier,
    UpliftTreeClassifier,
)
from causalml.metrics.sensitivity import Sensitivity

from .const import (
    RANDOM_SEED,
    CONTROL_NAME,
    CONVERSION,
    TREATMENT_NAMES,
    TREATMENT_COL,
    OUTCOME_COL,
    SCORE_COL,
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
        # UpliftTreeClassifier: the validation triple is reordered in place too,
        # so it stays consistent with the main one (#980 open question 1).
        (
            ["X", "treatment", "y", "X_val", "treatment_val", "y_val", "sample_weight"],
            ["X", "y", "treatment", "X_val", "y_val", "treatment_val", "sample_weight"],
        ),
        # Sensitivity helpers are in scope, and `p` shifts right (#980 q2).
        (["X", "p", "treatment", "y"], ["X", "y", "treatment", "p"]),
        # A signature with neither is left completely alone.
        (["X", "sample_weight"], ["X", "sample_weight"]),
        # Only `y`, no `treatment`: not this deprecation, so untouched. Keeps the
        # rule off the vendored sklearn tree builders, where hoisting `y` would
        # put it ahead of `X`.
        (["tree", "X", "y", "sample_weight"], ["tree", "X", "y", "sample_weight"]),
        (["X", "w", "y", "initial_taus"], ["X", "w", "y", "initial_taus"]),
    ],
)
def test_v1_order(current, expected):
    """X first, then y, then treatment; everything else keeps its relative order."""
    assert v1_order(current) == expected


# --- coverage across the whole package --------------------------------------


def _shimmable_methods():
    """Yield (qualname, method) for every public method whose order changes at v1.0.

    Scans by **signature**, not by method name. An earlier name-based allowlist
    silently missed eleven public methods (#981) — `bootstrap`,
    `fit_bootstrap_ensemble`, `bootstrap_pool`, `prune`, `fill` and the three
    `Sensitivity.get_*` helpers — so this guard must not be narrowed back to a
    fixed set of names.

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
            for name, method in vars(cls).items():
                if (
                    name.startswith("_")
                    or isinstance(method, (classmethod, staticmethod))
                    or not callable(method)
                    or getattr(method, "__isabstractmethod__", False)
                ):
                    continue
                try:
                    params = _positional_params(getattr(method, "__wrapped__", method))
                except (TypeError, ValueError):  # not introspectable
                    continue
                if v1_order(params) != params:
                    yield f"{cls.__module__}.{cls.__qualname__}.{name}", method


def test_every_reordered_method_is_shimmed():
    """Completeness guard for the deprecation window.

    A deprecation window is one-shot: anything whose positional order changes at
    v1.0 must warn in this release or it needs a second cycle. This fails if any
    public method's order would change without the shim.
    """
    unshimmed = [
        qualname
        for qualname, method in _shimmable_methods()
        if not getattr(method, "_arg_order_shimmed", False)
    ]
    assert unshimmed == [], f"missing #854 shim on: {unshimmed}"


def test_shim_reaches_past_the_fit_family():
    """#981: the shim must not regress to a fixed list of method names."""
    names = {qualname.rsplit(".", 1)[1] for qualname, _ in _shimmable_methods()}
    assert {"bootstrap", "fit_bootstrap_ensemble", "get_prediction"} <= names, names


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


# --- #981: coverage beyond the fit family -----------------------------------


def test_bootstrap_warns_positionally(generate_regression_data):
    """`bootstrap` is public API on the most-used learners and changes at v1.0."""
    y, X, treatment, _, _, _ = generate_regression_data()
    learner = BaseTRegressor(learner=LinearRegression())
    learner.fit(X=X, y=y, treatment=treatment)
    with pytest.warns(FutureWarning, match="argument order"):
        learner.bootstrap(X, treatment, y, None, 200)


def test_sensitivity_get_prediction_message_moves_p_right():
    """Sensitivity takes (X, p, treatment, y) — a third shape (#980 q2)."""
    params = _positional_params(Sensitivity.get_prediction.__wrapped__)
    assert params == ["X", "p", "treatment", "y"]
    assert v1_order(params) == ["X", "y", "treatment", "p"]


def test_uplift_tree_validation_triple_is_reordered():
    """#980 q1: X_val/y_val/treatment_val stays consistent with the main triple."""
    params = _positional_params(UpliftTreeClassifier.fit.__wrapped__)
    assert v1_order(params) == [
        "X",
        "y",
        "treatment",
        "X_val",
        "y_val",
        "treatment_val",
        "sample_weight",
        "check_input",
    ]


# --- the warning must point at a page that actually exists ------------------


def test_warning_links_to_the_migration_guide(generate_regression_data):
    """The link a user follows out of the warning has to reach instructions.

    It originally pointed at #854, which is a bug report -- it tells a reader
    what went wrong for someone else, not what to type.
    """
    y, X, treatment, _, _, _ = generate_regression_data()
    learner = BaseTRegressor(learner=LinearRegression())

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        learner.fit(X, treatment, y)

    message = str(_order_warnings(record)[0].message)
    assert MIGRATION_GUIDE_URL in message
    # The actionable instruction must be in the warning itself, not only behind
    # the link -- a user with no network still needs to know what to change.
    assert "fit(X, y=y, treatment=treatment)" in message


def test_migration_guide_url_matches_the_docs_source():
    """Pin the URL to the file and label it points at.

    Sphinx derives ``migration.html#fit-argument-order`` from the filename and
    the ``.. _fit-argument-order:`` label. Renaming either would leave the
    warning pointing at a 404 that nothing else would catch, because the
    warning text is a plain string and the docs build would stay green.
    """
    page, _, anchor = MIGRATION_GUIDE_URL.rpartition("/")[2].partition("#")
    guide = Path(__file__).resolve().parents[1] / "docs" / page.replace(".html", ".rst")

    assert guide.is_file(), f"{guide.name} is missing; the warning links to it"
    assert f".. _{anchor}:" in guide.read_text(), (
        f"label `{anchor}` is missing from {guide.name}; "
        "the warning's anchor would 404"
    )


def test_migration_guide_is_in_the_docs_toctree():
    """An unlinked page is unreachable by browsing and warns at build time."""
    docs = Path(__file__).resolve().parents[1] / "docs"
    assert "\n    migration\n" in (docs / "index.rst").read_text()


# --- the library's own internal calls must not warn the caller --------------


def _arg_order_warnings(record):
    """The shim's warnings only — not unrelated FutureWarnings from deps."""
    return [
        w
        for w in record
        if issubclass(w.category, FutureWarning) and "argument order" in str(w.message)
    ]


def test_causal_forest_keyword_fit_does_not_warn(generate_regression_data):
    """A forest fits each of its trees internally, and that must stay invisible.

    The shim's re-entrancy guard (``_in_arg_order_call``) is stored on the
    instance, so it does not span the forest -> tree hop: the flag is set on the
    forest while the warning fires on each tree. Until the internal calls were
    made keyword, an all-keyword ``fit`` emitted one FutureWarning per tree
    (default ``n_estimators=100``) telling the caller to do what they had
    already done, with no way to silence it.
    """
    y, X, treatment, _, _, _ = generate_regression_data()
    forest = CausalRandomForestRegressor(n_estimators=3, random_state=RANDOM_SEED)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        forest.fit(X=X, y=y, treatment=treatment)

    assert _arg_order_warnings(record) == []


def test_uplift_forest_keyword_fit_does_not_warn(generate_classification_data):
    """Same forest -> tree hop in the uplift ensemble (default n_estimators=10)."""
    df, x_names = generate_classification_data()
    forest = UpliftRandomForestClassifier(
        n_estimators=3,
        min_samples_leaf=50,
        control_name=CONTROL_NAME,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        forest.fit(
            X=df[x_names].values,
            y=df[CONVERSION].values,
            treatment=df["treatment_group_key"].values,
        )

    assert _arg_order_warnings(record) == []


def test_sensitivity_analysis_does_not_warn(generate_regression_data):
    """`sensitivity_analysis` takes no `treatment`/`y`, so it must never warn.

    Same instance-scoped re-entrancy gap as the forests, one module over:
    `sensitivity_analysis` is not itself shimmed (it has neither parameter), so
    it sets no guard flag, and the `get_prediction` / `get_ate_ci` /
    `get_potential_outcome_predictions` calls it makes internally each warned.
    The caller passed a DataFrame and column names -- there is no positional
    argument for them to fix, and no way to silence it.
    """
    y, X, treatment, _, _, e = generate_regression_data()
    features = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    df[TREATMENT_COL] = treatment
    df[OUTCOME_COL] = y
    df[SCORE_COL] = e

    sens = Sensitivity(
        df=df,
        inference_features=features,
        p_col=SCORE_COL,
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_COL,
        learner=BaseXRegressor(learner=LinearRegression()),
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sens.sensitivity_analysis(
            methods=["Random Cause", "Subset Data", "Random Replace"],
            sample_size=0.5,
        )

    assert _arg_order_warnings(record) == []


def test_timeit_preserves_signature():
    """`bootstrap_pool` is behind @timeit; without functools.wraps its signature
    reads (*args, **kw), which hid it from the signature-based shim."""
    assert CausalTreeRegressor.bootstrap_pool.__name__ == "bootstrap_pool"
    params = _positional_params(CausalTreeRegressor.bootstrap_pool.__wrapped__)
    assert params[:3] == ["X", "treatment", "y"]
