from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from causalml.dataset import synthetic_data
from causalml.inference.meta import BaseSLearner, LRSLearner
from causalml.inference.meta import BaseTLearner, XGBTLearner, MLPTLearner
from causalml.inference.meta import BaseXLearner, BaseRLearner
from causalml.metrics import ape, gini


from .const import N_SAMPLE, ERROR_THRESHOLD


def test_synthetic_data():
    y, X, treatment, tau, b, e = synthetic_data(mode=1, n=N_SAMPLE, p=8, sigma=.1)

    assert (y.shape[0] == X.shape[0] and y.shape[0] == treatment.shape[0] and
            y.shape[0] == tau.shape[0] and y.shape[0] == b.shape[0] and
            y.shape[0] == e.shape[0])

    y, X, treatment, tau, b, e = synthetic_data(mode=2, n=N_SAMPLE, p=8, sigma=.1)

    assert (y.shape[0] == X.shape[0] and y.shape[0] == treatment.shape[0] and
            y.shape[0] == tau.shape[0] and y.shape[0] == b.shape[0] and
            y.shape[0] == e.shape[0])

    y, X, treatment, tau, b, e = synthetic_data(mode=3, n=N_SAMPLE, p=8, sigma=.1)

    assert (y.shape[0] == X.shape[0] and y.shape[0] == treatment.shape[0] and
            y.shape[0] == tau.shape[0] and y.shape[0] == b.shape[0] and
            y.shape[0] == e.shape[0])

    y, X, treatment, tau, b, e = synthetic_data(mode=4, n=N_SAMPLE, p=8, sigma=.1)

    assert (y.shape[0] == X.shape[0] and y.shape[0] == treatment.shape[0] and
            y.shape[0] == tau.shape[0] and y.shape[0] == b.shape[0] and
            y.shape[0] == e.shape[0])


def test_BaseSLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseSLearner(learner=LinearRegression())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_LRSLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = LRSLearner()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseTLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseTLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10)
    assert gini(tau, cate_p.flatten()) > .5


def test_MLPTLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = MLPTLearner()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10)
    assert gini(tau, cate_p.flatten()) > .5


def test_XGBTLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = XGBTLearner()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10)
    assert gini(tau, cate_p.flatten()) > .5


def test_BaseXLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseXLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, p=e, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(X=X, p=e, treatment=treatment, y=y, return_ci=True, n_bootstraps=10)
    assert gini(tau, cate_p.flatten()) > .5


def test_BaseRLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseRLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, p=e, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(X=X, p=e, treatment=treatment, y=y, return_ci=True, n_bootstraps=10)
    assert gini(tau, cate_p.flatten()) > .5
