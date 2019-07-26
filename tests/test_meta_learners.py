import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from causalml.dataset import synthetic_data
from causalml.inference.meta import LRSLearner
from causalml.inference.meta import XGBTLearner, MLPTLearner
from causalml.inference.meta import BaseXLearner, BaseRLearner, BaseSLearner, BaseTLearner
from causalml.metrics import ape, mape


from const import RANDOM_SEED, N_SAMPLE, ERROR_THRESHOLD


@pytest.fixture
def generate_data():

    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = synthetic_data(mode=1, n=N_SAMPLE, p=8, sigma=.1)

        return data

    yield _generate_data


def test_synthetic_data(generate_data):
    y, X, treatment, tau, b, e = generate_data()

    assert y.shape[0] == X.shape[0]
    assert y.shape[0] == treatment.shape[0]
    assert y.shape[0] == tau.shape[0]
    assert y.shape[0] == b.shape[0]
    assert y.shape[0] == e.shape[0]


def test_LRSLearner(generate_data):
    y, X, treatment, tau, b, e = generate_data()

    learner = LRSLearner()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseTLearner(generate_data):
    y, X, treatment, tau, b, e = generate_data()

    learner = BaseTLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_MLPTLearner(generate_data):
    y, X, treatment, tau, b, e = generate_data()

    learner = MLPTLearner()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_XGBTLearner(generate_data):
    y, X, treatment, tau, b, e = generate_data()

    learner = XGBTLearner()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseXLearner(generate_data):
    y, X, treatment, tau, b, e = generate_data()

    learner = BaseXLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, p=e, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseRLearner(generate_data):
    y, X, treatment, tau, b, e = generate_data()

    learner = BaseRLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, p=e, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD
