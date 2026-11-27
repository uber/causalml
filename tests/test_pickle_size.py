"""Regression guard for the serialized (pickled) size of fitted learners.

A fitted learner should only keep what it needs to predict. If a change starts
storing training data, per-sample arrays or other large intermediates on the
estimator, the pickle grows silently and every saved model gets bigger. These
tests fit each learner on a small fixed dataset, pickle it, and check the size
against a budget so that kind of change shows up as a failing test.

Part of the v1.0 roadmap, M1 "serialization hygiene + a pickle-size regression
guard in CI" (https://github.com/uber/causalml/discussions/938).
"""

import pickle

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalml.inference.meta import (
    BaseDRRegressor,
    BaseRClassifier,
    BaseRRegressor,
    BaseSClassifier,
    BaseSRegressor,
    BaseTClassifier,
    BaseTRegressor,
    BaseXClassifier,
    BaseXRegressor,
)
from causalml.inference.tree import (
    CausalRandomForestRegressor,
    CausalTreeRegressor,
    UpliftRandomForestClassifier,
    UpliftTreeClassifier,
)

from .const import CONTROL_NAME, CONVERSION, RANDOM_SEED

# Maximum pickle size in bytes for each learner, fitted on the conftest
# datasets (N_SAMPLE rows) with the settings in the builders below. Each budget
# is about 2x the size measured when it was set (shown in the comment), so that
# small changes from library versions or new attributes pass, while anything
# that scales with the training data fails.
#
# If a change makes a learner legitimately bigger, re-measure with
#     len(pickle.dumps(fitted_learner))
# and set the budget to about 2x the new size, rounded up. Mention the change
# in the PR so reviewers can confirm the growth is expected.
PICKLE_SIZE_BUDGETS = {
    "BaseSRegressor": 2_000,  # 920
    "BaseTRegressor": 3_000,  # 1,283
    "BaseXRegressor": 4_500,  # 2,074
    "BaseRRegressor": 4_500,  # 2,031
    "BaseDRRegressor": 7_000,  # 3,267
    "BaseSClassifier": 2_500,  # 1,110
    "BaseTClassifier": 3_000,  # 1,518
    "BaseXClassifier": 5_000,  # 2,451
    "BaseRClassifier": 4_500,  # 2,147
    "CausalTreeRegressor": 4_500,  # 2,112
    "CausalRandomForestRegressor": 14_000,  # 6,950
    "UpliftTreeClassifier": 9_000,  # 4,256
    "UpliftRandomForestClassifier": 40_000,  # 20,010
}


def _regression_learners():
    return {
        "BaseSRegressor": lambda: BaseSRegressor(learner=LinearRegression()),
        "BaseTRegressor": lambda: BaseTRegressor(learner=LinearRegression()),
        "BaseXRegressor": lambda: BaseXRegressor(learner=LinearRegression()),
        "BaseRRegressor": lambda: BaseRRegressor(
            learner=LinearRegression(), random_state=RANDOM_SEED
        ),
        "BaseDRRegressor": lambda: BaseDRRegressor(learner=LinearRegression()),
        "CausalTreeRegressor": lambda: CausalTreeRegressor(
            max_depth=4, random_state=RANDOM_SEED
        ),
        "CausalRandomForestRegressor": lambda: CausalRandomForestRegressor(
            n_estimators=5, max_depth=4, n_jobs=1, random_state=RANDOM_SEED
        ),
    }


def _binary_outcome_learners():
    return {
        "BaseSClassifier": lambda: BaseSClassifier(
            learner=LogisticRegression(max_iter=1000)
        ),
        "BaseTClassifier": lambda: BaseTClassifier(
            learner=LogisticRegression(max_iter=1000)
        ),
        "BaseXClassifier": lambda: BaseXClassifier(
            outcome_learner=LogisticRegression(max_iter=1000),
            effect_learner=LinearRegression(),
        ),
        "BaseRClassifier": lambda: BaseRClassifier(
            outcome_learner=LogisticRegression(max_iter=1000),
            effect_learner=LinearRegression(),
            random_state=RANDOM_SEED,
        ),
    }


def _uplift_learners():
    return {
        "UpliftTreeClassifier": lambda: UpliftTreeClassifier(
            control_name=CONTROL_NAME,
            max_depth=4,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        ),
        "UpliftRandomForestClassifier": lambda: UpliftRandomForestClassifier(
            control_name=CONTROL_NAME,
            n_estimators=5,
            max_depth=4,
            min_samples_leaf=20,
            n_jobs=1,
            random_state=RANDOM_SEED,
        ),
    }


# Learners that take a propensity score in fit(); passing the true one skips
# fitting a propensity model, which keeps the test fast.
_TAKES_PROPENSITY = {
    "BaseXRegressor",
    "BaseRRegressor",
    "BaseDRRegressor",
    "BaseXClassifier",
    "BaseRClassifier",
}


def _assert_within_budget(name, learner):
    size = len(pickle.dumps(learner))
    budget = PICKLE_SIZE_BUDGETS[name]
    assert size <= budget, (
        f"{name} pickles to {size:,} bytes, over its budget of {budget:,} bytes. "
        "If the growth is expected, update PICKLE_SIZE_BUDGETS in "
        "tests/test_pickle_size.py (see the comment there)."
    )


def test_every_budgeted_learner_is_tested():
    tested = {
        *_regression_learners(),
        *_binary_outcome_learners(),
        *_uplift_learners(),
    }
    assert tested == set(PICKLE_SIZE_BUDGETS)


@pytest.mark.parametrize("name", sorted(_regression_learners()))
def test_regression_learner_pickle_size(name, generate_regression_data):
    y, X, treatment, _, _, e = generate_regression_data()
    learner = _regression_learners()[name]()
    if name in _TAKES_PROPENSITY:
        learner.fit(X=X, treatment=treatment, y=y, p=e)
    else:
        learner.fit(X=X, treatment=treatment, y=y)
    _assert_within_budget(name, learner)


@pytest.mark.parametrize("name", sorted(_binary_outcome_learners()))
def test_binary_outcome_learner_pickle_size(name, generate_regression_data):
    y, X, treatment, _, _, e = generate_regression_data()
    y_binary = (y > y.mean()).astype(int)
    learner = _binary_outcome_learners()[name]()
    if name in _TAKES_PROPENSITY:
        learner.fit(X=X, treatment=treatment, y=y_binary, p=e)
    else:
        learner.fit(X=X, treatment=treatment, y=y_binary)
    _assert_within_budget(name, learner)


@pytest.mark.parametrize("name", sorted(_uplift_learners()))
def test_uplift_learner_pickle_size(name, generate_classification_data):
    df, x_names = generate_classification_data()
    learner = _uplift_learners()[name]()
    learner.fit(
        X=df[x_names].values,
        treatment=df["treatment_group_key"].values,
        y=df[CONVERSION].values,
    )
    _assert_within_budget(name, learner)
