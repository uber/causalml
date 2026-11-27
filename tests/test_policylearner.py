import numpy as np
import pytest

from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeClassifier

from causalml.optimize.policylearner import PolicyLearner

from tests.const import RANDOM_SEED


class FixedPolicyLearner:
    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        return np.tile([0.25, 0.75], (len(X), 1))


@pytest.mark.parametrize(
    "dhat, expected_classes, treat_column",
    [
        ([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], 2),
        ([1.0] * 6, [1.0], 0),
        ([-1.0] * 6, [-1.0], None),
        ([-1.0, 1.0] * 3, [-1.0, 1.0], 1),
    ],
)
def test_predict_proba_returns_probability_of_treat_class(
    dhat, expected_classes, treat_column
):
    X = np.arange(12).reshape(6, 2)
    treatment = np.array([0, 1] * 3)
    y = np.zeros(6)
    learner = PolicyLearner(
        outcome_learner=DummyRegressor(strategy="constant", constant=0),
        policy_learner=DecisionTreeClassifier(random_state=RANDOM_SEED),
        random_state=RANDOM_SEED,
    )

    learner.fit(
        X,
        treatment=treatment,
        y=y,
        p=np.full(6, 0.5),
        dhat=np.asarray(dhat),
    )

    np.testing.assert_array_equal(learner.model_pi.classes_, expected_classes)
    probabilities = learner.model_pi.predict_proba(X)
    expected = (
        probabilities[:, treat_column] if treat_column is not None else np.zeros(6)
    )
    result = learner.predict_proba(X)

    np.testing.assert_array_equal(result, expected)
    treat_rows = np.asarray(dhat) == 1
    np.testing.assert_array_equal(result[treat_rows], 1.0)
    if np.any(~treat_rows):
        np.testing.assert_array_equal(result[~treat_rows], 0.0)


def test_predict_proba_without_classes_returns_second_column():
    X = np.arange(12).reshape(6, 2)
    treatment = np.array([0, 1] * 3)
    y = np.zeros(6)
    learner = PolicyLearner(
        outcome_learner=DummyRegressor(strategy="constant", constant=0),
        policy_learner=FixedPolicyLearner(),
        random_state=RANDOM_SEED,
    )

    learner.fit(
        X,
        treatment=treatment,
        y=y,
        p=np.full(6, 0.5),
        dhat=np.ones(6),
    )

    np.testing.assert_array_equal(learner.predict_proba(X), np.full(6, 0.75))
