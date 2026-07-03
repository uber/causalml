import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor

from causalml.dataset import synthetic_data
from causalml.inference.meta import (
    BaseSLearner,
    BaseSRegressor,
    BaseSClassifier,
    LRSRegressor,
)
from causalml.inference.meta import (
    BaseTLearner,
    BaseTRegressor,
    BaseTClassifier,
    XGBTRegressor,
    MLPTRegressor,
)
from causalml.inference.meta import BaseXLearner, BaseXClassifier, BaseXRegressor
from causalml.inference.meta import (
    BaseRLearner,
    BaseRClassifier,
    BaseRRegressor,
    XGBRRegressor,
)
from causalml.inference.meta import TMLELearner
from causalml.inference.meta import BaseDRLearner
from causalml.inference.meta import BaseDRRegressor
from causalml.inference.meta import BaseDRClassifier
from causalml.inference.meta.explainer import Explainer
from causalml.metrics import ape, auuc_score

from .const import RANDOM_SEED, N_SAMPLE, ERROR_THRESHOLD, CONTROL_NAME, CONVERSION


class ReadOnlyLinearRegression:
    """Minimal regressor that marks input arrays read-only like CatBoost."""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        X.flags.writeable = False
        return self

    def predict(self, X):
        result = self.model.predict(X)
        X.flags.writeable = False
        return result


def test_synthetic_data():
    y, X, treatment, tau, b, e = synthetic_data(mode=1, n=N_SAMPLE, p=8, sigma=0.1)

    assert (
        y.shape[0] == X.shape[0]
        and y.shape[0] == treatment.shape[0]
        and y.shape[0] == tau.shape[0]
        and y.shape[0] == b.shape[0]
        and y.shape[0] == e.shape[0]
    )

    y, X, treatment, tau, b, e = synthetic_data(mode=2, n=N_SAMPLE, p=8, sigma=0.1)

    assert (
        y.shape[0] == X.shape[0]
        and y.shape[0] == treatment.shape[0]
        and y.shape[0] == tau.shape[0]
        and y.shape[0] == b.shape[0]
        and y.shape[0] == e.shape[0]
    )

    y, X, treatment, tau, b, e = synthetic_data(mode=3, n=N_SAMPLE, p=8, sigma=0.1)

    assert (
        y.shape[0] == X.shape[0]
        and y.shape[0] == treatment.shape[0]
        and y.shape[0] == tau.shape[0]
        and y.shape[0] == b.shape[0]
        and y.shape[0] == e.shape[0]
    )

    y, X, treatment, tau, b, e = synthetic_data(mode=4, n=N_SAMPLE, p=8, sigma=0.1)

    assert (
        y.shape[0] == X.shape[0]
        and y.shape[0] == treatment.shape[0]
        and y.shape[0] == tau.shape[0]
        and y.shape[0] == b.shape[0]
        and y.shape[0] == e.shape[0]
    )


def test_BaseSLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseSLearner(learner=LinearRegression())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, return_ci=True)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, return_ci=True, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)


def test_BaseSLearner_predict_with_readonly_arrays(generate_regression_data):
    y, X, treatment, _, _, _ = generate_regression_data()
    X_readonly = np.array(X, copy=True)
    X_readonly.flags.writeable = False

    learner = BaseSLearner(learner=ReadOnlyLinearRegression())

    # Exercise both fit() and predict() with read-only array behavior.
    learner.fit(X=X_readonly, treatment=treatment, y=y)
    cate = learner.predict(X=X_readonly)

    assert cate.shape == (X.shape[0], 1)
    assert not X_readonly.flags.writeable


def test_BaseSRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseSRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
    )
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5


def test_LRSRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = LRSRegressor()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)


def test_BaseTLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseTLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5

    # test of using control_learner and treatment_learner
    learner = BaseTLearner(
        learner=XGBRegressor(),
        control_learner=RandomForestRegressor(),
        treatment_learner=RandomForestRegressor(),
    )
    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseTRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseTRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5


def test_MLPTRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = MLPTRegressor()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5


def test_XGBTRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = XGBTRegressor()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5


def test_BaseXLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseXLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, p=e, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, p=e, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5

    # basic test of using outcome_learner and effect_learner
    learner = BaseXLearner(
        learner=XGBRegressor(),
        control_outcome_learner=RandomForestRegressor(),
        treatment_outcome_learner=RandomForestRegressor(),
        control_effect_learner=RandomForestRegressor(),
        treatment_effect_learner=RandomForestRegressor(),
    )
    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseXRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseXRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, p=e, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, p=e, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5


def test_BaseXLearner_without_p(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseXLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, pretrain=True
    )
    assert ate_p_pt == ate_p
    assert lb_pt == lb
    assert ub_pt == ub

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5


def test_BaseXRegressor_without_p(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseXRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5


def test_BaseRLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseRLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, p=e, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, p=e, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5

    # basic test of using outcome_learner and effect_learner
    learner = BaseRLearner(
        learner=XGBRegressor(),
        outcome_learner=RandomForestRegressor(),
        effect_learner=RandomForestRegressor(),
    )
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert (
        ape(tau.mean(), ate_p) < ERROR_THRESHOLD * 5
    )  # might need to look into higher ape


def test_BaseRRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseRRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, p=e, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, p=e, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5


def test_explainer_auto_importance_catboost(generate_regression_data):
    catboost = pytest.importorskip("catboost")

    y, X, treatment, tau, b, e = generate_regression_data()

    model_tau = catboost.CatBoostRegressor(
        iterations=10,
        verbose=False,
        random_seed=RANDOM_SEED,
    )

    explainer = Explainer(
        method="auto",
        control_name=CONTROL_NAME,
        X=X,
        tau=tau,
        classes={CONTROL_NAME: 0},
        model_tau=model_tau,
    )

    importance = explainer.get_importance()

    assert len(importance) == 1
    assert CONTROL_NAME in importance
    assert isinstance(importance[CONTROL_NAME], pd.Series)
    assert len(importance[CONTROL_NAME]) == X.shape[1]


def test_BaseRLearner_without_p(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseRLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5


def test_BaseRRegressor_without_p(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseRRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pre-train model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5


def test_TMLELearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = TMLELearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, p=e, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseSClassifier(generate_classification_data):
    np.random.seed(RANDOM_SEED)

    df, x_names = generate_classification_data()

    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    uplift_model = BaseSClassifier(learner=XGBClassifier())

    uplift_model.fit(
        X=df_train[x_names].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
    )

    tau_pred = uplift_model.predict(
        X=df_test[x_names].values, treatment=df_test["treatment_group_key"].values
    )

    auuc_metrics = pd.DataFrame(
        {
            "tau_pred": tau_pred.flatten(),
            "W": df_test["treatment_group_key"].values,
            CONVERSION: df_test[CONVERSION].values,
            "treatment_effect_col": df_test["treatment_effect"].values,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col=CONVERSION,
        treatment_col="W",
        treatment_effect_col="treatment_effect_col",
        normalize=True,
    )
    assert auuc["tau_pred"] > 0.5


def test_BaseTClassifier(generate_classification_data):
    np.random.seed(RANDOM_SEED)

    df, x_names = generate_classification_data()

    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    uplift_model = BaseTClassifier(learner=LogisticRegression())

    uplift_model.fit(
        X=df_train[x_names].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
    )

    tau_pred = uplift_model.predict(
        X=df_test[x_names].values, treatment=df_test["treatment_group_key"].values
    )

    auuc_metrics = pd.DataFrame(
        {
            "tau_pred": tau_pred.flatten(),
            "W": df_test["treatment_group_key"].values,
            CONVERSION: df_test[CONVERSION].values,
            "treatment_effect_col": df_test["treatment_effect"].values,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col=CONVERSION,
        treatment_col="W",
        treatment_effect_col="treatment_effect_col",
        normalize=True,
    )
    assert auuc["tau_pred"] > 0.5


def test_BaseXClassifier(generate_classification_data):
    np.random.seed(RANDOM_SEED)

    df, x_names = generate_classification_data()

    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )

    propensity_model = LogisticRegression()
    propensity_model.fit(X=df[x_names].values, y=df["treatment_group_key"].values)
    df["propensity_score"] = propensity_model.predict_proba(df[x_names].values)[:, 1]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    # specify all 4 learners
    uplift_model = BaseXClassifier(
        control_outcome_learner=XGBClassifier(),
        control_effect_learner=XGBRegressor(),
        treatment_outcome_learner=XGBClassifier(),
        treatment_effect_learner=XGBRegressor(),
    )

    uplift_model.fit(
        X=df_train[x_names].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
    )

    tau_pred = uplift_model.predict(
        X=df_test[x_names].values, p=df_test["propensity_score"].values
    )

    # specify 2 learners
    uplift_model = BaseXClassifier(
        outcome_learner=XGBClassifier(), effect_learner=XGBRegressor()
    )

    uplift_model.fit(
        X=df_train[x_names].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
    )

    tau_pred = uplift_model.predict(
        X=df_test[x_names].values, p=df_test["propensity_score"].values
    )

    # calculate metrics
    auuc_metrics = pd.DataFrame(
        {
            "tau_pred": tau_pred.flatten(),
            "W": df_test["treatment_group_key"].values,
            CONVERSION: df_test[CONVERSION].values,
            "treatment_effect_col": df_test["treatment_effect"].values,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col=CONVERSION,
        treatment_col="W",
        treatment_effect_col="treatment_effect_col",
        normalize=True,
    )
    assert auuc["tau_pred"] > 0.5


def test_BaseRClassifier(generate_classification_data):
    np.random.seed(RANDOM_SEED)

    df, x_names = generate_classification_data()

    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )

    propensity_model = LogisticRegression()
    propensity_model.fit(X=df[x_names].values, y=df["treatment_group_key"].values)
    df["propensity_score"] = propensity_model.predict_proba(df[x_names].values)[:, 1]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    uplift_model = BaseRClassifier(
        outcome_learner=XGBClassifier(), effect_learner=XGBRegressor()
    )

    uplift_model.fit(
        X=df_train[x_names].values,
        p=df_train["propensity_score"].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
    )

    tau_pred = uplift_model.predict(X=df_test[x_names].values)

    auuc_metrics = pd.DataFrame(
        {
            "tau_pred": tau_pred.flatten(),
            "W": df_test["treatment_group_key"].values,
            CONVERSION: df_test[CONVERSION].values,
            "treatment_effect_col": df_test["treatment_effect"].values,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col=CONVERSION,
        treatment_col="W",
        treatment_effect_col="treatment_effect_col",
        normalize=True,
    )
    assert auuc["tau_pred"] > 0.5


def test_BaseRClassifier_with_sample_weights(generate_classification_data):
    np.random.seed(RANDOM_SEED)

    df, x_names = generate_classification_data()

    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )
    df["sample_weights"] = np.random.randint(low=1, high=3, size=df.shape[0])

    propensity_model = LogisticRegression()
    propensity_model.fit(X=df[x_names].values, y=df["treatment_group_key"].values)
    df["propensity_score"] = propensity_model.predict_proba(df[x_names].values)[:, 1]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    uplift_model = BaseRClassifier(
        outcome_learner=XGBClassifier(), effect_learner=XGBRegressor()
    )

    uplift_model.fit(
        X=df_train[x_names].values,
        p=df_train["propensity_score"].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
        sample_weight=df_train["sample_weights"],
    )

    tau_pred = uplift_model.predict(X=df_test[x_names].values)

    auuc_metrics = pd.DataFrame(
        {
            "tau_pred": tau_pred.flatten(),
            "W": df_test["treatment_group_key"].values,
            CONVERSION: df_test[CONVERSION].values,
            "treatment_effect_col": df_test["treatment_effect"].values,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col=CONVERSION,
        treatment_col="W",
        treatment_effect_col="treatment_effect_col",
        normalize=True,
    )
    assert auuc["tau_pred"] > 0.5


def test_XGBRegressor_with_sample_weights(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    weights = np.random.rand(y.shape[0])

    # Check if XGBRRegressor successfully produces treatment effect estimation
    # when sample_weight is passed
    uplift_model = XGBRRegressor()
    uplift_model.fit(X=X, p=e, treatment=treatment, y=y, sample_weight=weights)
    tau_pred = uplift_model.predict(X=X)
    assert len(tau_pred) == len(weights)


def test_pandas_input(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()
    # convert to pandas types
    y = pd.Series(y)
    X = pd.DataFrame(X)
    treatment = pd.Series(treatment)

    try:
        learner = BaseSLearner(learner=LinearRegression())
        ate_p, lb, ub = learner.estimate_ate(
            X=X, treatment=treatment, y=y, return_ci=True
        )
    except AttributeError:
        assert False
    try:
        learner = BaseTLearner(learner=LinearRegression())
        ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    except AttributeError:
        assert False
    try:
        learner = BaseXLearner(learner=LinearRegression())
        ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    except AttributeError:
        assert False
    try:
        learner = BaseRLearner(learner=LinearRegression())
        ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    except AttributeError:
        assert False
    try:
        learner = TMLELearner(learner=LinearRegression())
        ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    except AttributeError:
        assert False


def test_BaseDRLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseDRLearner(
        learner=XGBRegressor(), treatment_effect_learner=LinearRegression()
    )

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check pretrain model
    ate_p_pt, lb_pt, ub_pt = learner.estimate_ate(
        X=X, treatment=treatment, y=y, p=e, pretrain=True
    )
    assert (ate_p_pt == ate_p) and (lb_pt == lb) and (ub_pt == ub)

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, p=e, return_ci=True, n_bootstraps=10
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    # Check if the normalized AUUC score of model's prediction is higher than random (0.5).
    auuc = auuc_score(
        auuc_metrics,
        outcome_col="y",
        treatment_col="W",
        treatment_effect_col="tau",
        normalize=True,
    )
    assert auuc["cate_p"] > 0.5


def test_BaseDRLearner_estimate_ate_bootstrap(generate_regression_data):
    """Regression test for issue #857: estimate_ate with bootstrap_ci=True
    raised TypeError due to stray seed argument passed to bootstrap()."""
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseDRRegressor(learner=LinearRegression(), control_name=0)

    # This call raised TypeError before the fix
    ate, lb, ub = learner.estimate_ate(
        X=X,
        treatment=treatment,
        y=y,
        p=e,
        bootstrap_ci=True,
        n_bootstraps=10,
        bootstrap_size=200,
        seed=RANDOM_SEED,
    )

    # Verify results are valid
    assert np.all(np.isfinite(ate))
    assert np.all(np.isfinite(lb))
    assert np.all(np.isfinite(ub))

    # Verify same seed produces identical bootstrap CI bounds
    learner2 = BaseDRRegressor(learner=LinearRegression(), control_name=0)
    ate2, lb2, ub2 = learner2.estimate_ate(
        X=X,
        treatment=treatment,
        y=y,
        p=e,
        bootstrap_ci=True,
        n_bootstraps=10,
        bootstrap_size=200,
        seed=RANDOM_SEED,
    )
    np.testing.assert_array_equal(lb, lb2)
    np.testing.assert_array_equal(ub, ub2)

    # fit_predict() should also honor seed for bootstrap reproducibility.
    learner_fp1 = BaseDRRegressor(learner=LinearRegression(), control_name=0)
    _, te_lb1, te_ub1 = learner_fp1.fit_predict(
        X=X,
        treatment=treatment,
        y=y,
        p=e,
        return_ci=True,
        n_bootstraps=10,
        bootstrap_size=200,
        seed=RANDOM_SEED,
    )
    learner_fp2 = BaseDRRegressor(learner=LinearRegression(), control_name=0)
    _, te_lb2, te_ub2 = learner_fp2.fit_predict(
        X=X,
        treatment=treatment,
        y=y,
        p=e,
        return_ci=True,
        n_bootstraps=10,
        bootstrap_size=200,
        seed=RANDOM_SEED,
    )
    np.testing.assert_array_equal(te_lb1, te_lb2)
    np.testing.assert_array_equal(te_ub1, te_ub2)

    # Verify seed=None still returns valid results
    learner3 = BaseDRRegressor(learner=LinearRegression(), control_name=0)
    ate3, lb3, ub3 = learner3.estimate_ate(
        X=X,
        treatment=treatment,
        y=y,
        p=e,
        bootstrap_ci=True,
        n_bootstraps=10,
        bootstrap_size=200,
    )
    assert np.all(np.isfinite(ate3))
    assert np.all(np.isfinite(lb3))
    assert np.all(np.isfinite(ub3))

    # Verify global RNG state is not leaked by seeded bootstrap
    np.random.seed(99)
    _ = np.random.random()
    state_before = np.random.get_state()
    learner4 = BaseDRRegressor(learner=LinearRegression(), control_name=0)
    learner4.estimate_ate(
        X=X,
        treatment=treatment,
        y=y,
        p=e,
        bootstrap_ci=True,
        n_bootstraps=10,
        bootstrap_size=200,
        seed=RANDOM_SEED,
    )
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_BaseDRClassifier(generate_classification_data):
    np.random.seed(RANDOM_SEED)

    df, X_names = generate_classification_data()

    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )

    # Extract features and outcome
    y = df[CONVERSION].values
    X = df[X_names].values
    treatment = df["treatment_group_key"].values

    learner = BaseDRClassifier(
        learner=LogisticRegression(), treatment_effect_learner=LinearRegression()
    )

    # Test fit and predict
    te = learner.fit_predict(X=X, treatment=treatment, y=y)

    # Check that treatment effects are returned
    assert te.shape[0] == X.shape[0]
    assert te.shape[1] == len(np.unique(treatment[treatment != 0]))

    # Test with return_components
    te, yhat_cs, yhat_ts = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_components=True
    )

    # Check that components are returned as probabilities
    for group in learner.t_groups:
        assert np.all((yhat_cs[group] >= 0) & (yhat_cs[group] <= 1))
        assert np.all((yhat_ts[group] >= 0) & (yhat_ts[group] <= 1))

    # Test separate outcome and effect learners
    learner_separate = BaseDRClassifier(
        control_outcome_learner=LogisticRegression(),
        treatment_outcome_learner=LogisticRegression(),
        treatment_effect_learner=LinearRegression(),
    )

    te_separate = learner_separate.fit_predict(X=X, treatment=treatment, y=y)
    assert te_separate.shape == te.shape


def test_BaseTLearner_predict_return_ci(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseTRegressor(learner=LinearRegression(), control_name=0)

    # Test 1: store_bootstraps=True then predict with return_ci=True
    learner.fit(
        X,
        treatment,
        y,
        store_bootstraps=True,
        n_bootstraps=50,
        bootstrap_size=500,
        random_state=RANDOM_SEED,
    )
    tau_pred, lb, ub = learner.predict(X, return_ci=True)

    assert tau_pred.shape == (X.shape[0], len(learner.t_groups))
    assert lb.shape == tau_pred.shape
    assert ub.shape == tau_pred.shape
    assert (lb <= ub).all()

    # Test 2: ValueError without store_bootstraps
    learner2 = BaseTRegressor(learner=LinearRegression(), control_name=0)
    learner2.fit(X, treatment, y)
    with pytest.raises(ValueError):
        learner2.predict(X, return_ci=True)

    # Test 3: ValueError when return_ci and return_components both True
    with pytest.raises(ValueError):
        learner.predict(X, return_ci=True, return_components=True)

    # Test 4: old API unchanged
    tau_plain = learner.predict(X)
    assert tau_plain.shape == (X.shape[0], len(learner.t_groups))

    # Test 5: reproducibility via random_state
    learner3 = BaseTRegressor(learner=LinearRegression(), control_name=0)
    learner3.fit(
        X,
        treatment,
        y,
        store_bootstraps=True,
        n_bootstraps=50,
        bootstrap_size=500,
        random_state=RANDOM_SEED,
    )
    tau2, lb2, ub2 = learner3.predict(X, return_ci=True)
    np.testing.assert_array_equal(lb, lb2)
    np.testing.assert_array_equal(ub, ub2)

    # Test 6: parallel execution (n_jobs=2) produces same result as serial
    learner_parallel = BaseTRegressor(learner=LinearRegression(), control_name=0)
    learner_parallel.fit(
        X,
        treatment,
        y,
        store_bootstraps=True,
        n_bootstraps=50,
        bootstrap_size=500,
        random_state=RANDOM_SEED,
        n_jobs=2,
    )
    tau_p, lb_p, ub_p = learner_parallel.predict(X, return_ci=True)
    assert tau_p.shape == (X.shape[0], len(learner_parallel.t_groups))
    assert (lb_p <= ub_p).all()

    # Test 7: different random_state produces different bounds
    learner_diff = BaseTRegressor(learner=LinearRegression(), control_name=0)
    learner_diff.fit(
        X,
        treatment,
        y,
        store_bootstraps=True,
        n_bootstraps=50,
        bootstrap_size=500,
        random_state=RANDOM_SEED + 1,
    )
    _, lb_diff, ub_diff = learner_diff.predict(X, return_ci=True)
    assert not np.array_equal(
        lb, lb_diff
    ), "Different seeds should produce different bounds"


def test_multi_treatment_learners():
    """Comprehensive multi-treatment (N=3) contract test for all meta-learners.

    Verifies three classes of invariants:
      1. Common API contracts — return types and shapes for every public method.
      2. Structural post-fit invariants — shared-reference dicts, attribute presence.
      3. Optimisation correctness — control models trained once and shared.

    Covers: BaseTLearner, BaseXLearner, BaseSLearner, BaseDRLearner, BaseRLearner.

    Shared return-type contracts (regression learners below):
      - ``fit(...)`` → ``self`` (sklearn convention; enables method chaining)
      - ``predict(...)`` → ``np.ndarray`` of shape ``(n_samples, n_treatment_groups)``
      - ``predict(..., return_components=True)`` → ``tuple`` of length 3 ``(te, comp_a, comp_b)``
        (not implemented for R-learner; its ``predict`` only returns CATE).
      - ``fit_predict(..., return_ci=False)`` → CATE ``np.ndarray`` only (not a tuple)
      - ``fit_predict(..., return_ci=True)`` → ``tuple`` ``(te, lb, ub)`` of three ndarrays
      - ``estimate_ate(...)`` → ``tuple`` ``(ate, lb, ub)`` with each vector of shape
        ``(n_treatment_groups,)`` for T/X/R/DR by default; **BaseSLearner** returns only
        ``ate`` unless ``return_ci=True`` (then same triple as the others).
    """
    np.random.seed(RANDOM_SEED)
    n, p, n_groups = 600, 5, 3
    X = np.random.randn(n, p)
    # Three treatment groups (1, 2, 3) plus control (0), ~150 obs each.
    treatment = np.tile([0, 1, 2, 3], n // 4)
    tau = np.where(
        treatment == 1,
        1.0,
        np.where(treatment == 2, 2.0, np.where(treatment == 3, 3.0, 0.0)),
    )
    y = X[:, 0] + tau + 0.1 * np.random.randn(n)
    # Flat propensity scores for learners that require them (X, R).
    p_scores = {g: np.full(n, 1.0 / (n_groups + 1)) for g in [1, 2, 3]}

    # ── Shared assertion helpers ───────────────────────────────────────────────

    def _assert_fit_attrs(lrn, name):
        """t_groups must be a sorted ndarray; _classes must map each group to 0..N-1."""
        assert hasattr(lrn, "t_groups"), f"{name}: missing t_groups after fit"
        assert isinstance(lrn.t_groups, np.ndarray), f"{name}: t_groups must be ndarray"
        assert lrn.t_groups.shape == (
            n_groups,
        ), f"{name}: t_groups shape {lrn.t_groups.shape}"
        np.testing.assert_array_equal(
            lrn.t_groups,
            np.sort(lrn.t_groups),
            err_msg=f"{name}: t_groups must be sorted",
        )
        assert hasattr(lrn, "_classes") and isinstance(lrn._classes, dict)
        assert set(lrn._classes.keys()) == set(lrn.t_groups)
        assert set(lrn._classes.values()) == set(range(n_groups))

    def _assert_te(te, name, method):
        """te must be ndarray (n, n_groups) of finite values."""
        assert isinstance(te, np.ndarray), f"{name}.{method}: te must be ndarray"
        assert te.shape == (n, n_groups), f"{name}.{method}: te.shape={te.shape}"
        assert np.all(np.isfinite(te)), f"{name}.{method}: te has non-finite values"

    def _assert_components(yhat_cs, yhat_ts, t_groups, name):
        """yhat_cs and yhat_ts must be dicts of finite (n,) arrays covering all groups."""
        for label, d in [("yhat_cs", yhat_cs), ("yhat_ts", yhat_ts)]:
            assert isinstance(d, dict), f"{name}: {label} must be dict, got {type(d)}"
            assert set(d.keys()) == set(t_groups), f"{name}: {label} keys != t_groups"
            for g in t_groups:
                assert isinstance(d[g], np.ndarray) and d[g].shape == (n,)
                assert np.all(np.isfinite(d[g])), f"{name}: {label}[{g}] has non-finite"

    def _assert_ate(result, name):
        """estimate_ate must return (ate, lb, ub) — finite ndarrays of shape (n_groups,), lb<=ub."""
        assert (
            isinstance(result, tuple) and len(result) == 3
        ), f"{name}.estimate_ate: expected 3-tuple, got {type(result)}"
        ate, lb, ub = result
        for arr, label in [(ate, "ate"), (lb, "lb"), (ub, "ub")]:
            assert isinstance(
                arr, np.ndarray
            ), f"{name}.estimate_ate {label} must be ndarray"
            assert arr.shape == (
                n_groups,
            ), f"{name}.estimate_ate {label}.shape={arr.shape}"
            assert np.all(
                np.isfinite(arr)
            ), f"{name}.estimate_ate {label} has non-finite"
        assert np.all(lb <= ub), f"{name}.estimate_ate: lb > ub"

    def _assert_ci_triple(result, name, method):
        """fit_predict(return_ci=True) must return (te, lb, ub), each (n, n_groups)."""
        assert isinstance(result, tuple) and len(result) == 3
        for arr, label in zip(result, ["te", "lb", "ub"]):
            _assert_te(arr, name, f"{method}[{label}]")

    def _assert_shared_ref_dict(d, single_obj, keys, name, attr):
        """Every value in d must be the same Python object as single_obj."""
        assert isinstance(d, dict), f"{name}: {attr} must be dict"
        assert set(d.keys()) == set(keys), f"{name}: {attr} keys mismatch"
        assert all(
            d[g] is single_obj for g in keys
        ), f"{name}: all {attr} values must be shared refs to the single fitted model"

    def _assert_fit_returns_self(result, learner, name):
        """fit() must return self (sklearn convention enabling method chaining)."""
        assert result is learner, f"{name}.fit(): expected self, got {type(result)}"

    def _assert_plain_fit_predict(result, name):
        """fit_predict(return_ci=False) must return a single ndarray (CATE), not a tuple."""
        assert isinstance(
            result, np.ndarray
        ), f"{name}.fit_predict(return_ci=False): expected ndarray, got {type(result)}"

    # ── T-Learner ─────────────────────────────────────────────────────────────
    name = "BaseTLearner"
    tl = BaseTLearner(learner=LinearRegression())
    _assert_fit_returns_self(tl.fit(X=X, treatment=treatment, y=y), tl, name)

    _assert_fit_attrs(tl, name)
    assert hasattr(tl, "model_c"), f"{name}: missing model_c"
    _assert_shared_ref_dict(tl.models_c, tl.model_c, tl.t_groups, name, "models_c")
    assert hasattr(tl, "models_t") and isinstance(tl.models_t, dict)
    assert set(tl.models_t.keys()) == set(tl.t_groups)
    # Treatment models must be distinct objects (trained on different per-group data).
    assert all(
        tl.models_t[g1] is not tl.models_t[g2]
        for g1, g2 in zip(tl.t_groups[:-1], tl.t_groups[1:])
    ), f"{name}: models_t must be distinct objects per group"

    te = tl.predict(X=X)
    _assert_te(te, name, "predict()")

    out_pc = tl.predict(X=X, return_components=True)
    assert (
        isinstance(out_pc, tuple) and len(out_pc) == 3
    ), f"{name}.predict(return_components=True) must return (te, yhat_cs, yhat_ts)"
    te2, yhat_cs, yhat_ts = out_pc
    np.testing.assert_array_equal(te, te2, err_msg=f"{name}: predict inconsistency")
    _assert_components(yhat_cs, yhat_ts, tl.t_groups, name)
    assert all(
        yhat_cs[g] is yhat_cs[tl.t_groups[0]] for g in tl.t_groups
    ), f"{name}: yhat_cs values must share the same underlying array"

    fp_plain = tl.fit_predict(X=X, treatment=treatment, y=y)
    _assert_plain_fit_predict(fp_plain, name)
    _assert_te(fp_plain, name, "fit_predict()")
    _assert_ci_triple(
        tl.fit_predict(
            X=X,
            treatment=treatment,
            y=y,
            return_ci=True,
            n_bootstraps=5,
            bootstrap_size=150,
        ),
        name,
        "fit_predict",
    )
    _assert_ate(tl.estimate_ate(X=X, treatment=treatment, y=y), name)
    _assert_ate(tl.estimate_ate(X=X, treatment=treatment, y=y, pretrain=True), name)

    # ── X-Learner ─────────────────────────────────────────────────────────────
    name = "BaseXLearner"
    xl = BaseXLearner(learner=LinearRegression())
    _assert_fit_returns_self(
        xl.fit(X=X, treatment=treatment, y=y, p=p_scores), xl, name
    )

    _assert_fit_attrs(xl, name)
    assert hasattr(xl, "model_mu_c"), f"{name}: missing model_mu_c"
    _assert_shared_ref_dict(
        xl.models_mu_c, xl.model_mu_c, xl.t_groups, name, "models_mu_c"
    )
    assert (
        hasattr(xl, "var_c") and np.isscalar(xl.var_c) and np.isfinite(xl.var_c)
    ), f"{name}: var_c must be a finite scalar"
    assert hasattr(xl, "vars_c") and isinstance(xl.vars_c, dict)
    assert all(
        xl.vars_c[g] == xl.var_c for g in xl.t_groups
    ), f"{name}: vars_c values must all equal var_c"
    for attr in ("models_mu_t", "models_tau_c", "models_tau_t", "vars_t"):
        assert hasattr(xl, attr) and isinstance(
            getattr(xl, attr), dict
        ), f"{name}: missing {attr}"
        assert set(getattr(xl, attr).keys()) == set(
            xl.t_groups
        ), f"{name}: {attr} keys mismatch"

    te = xl.predict(X=X, p=p_scores)
    _assert_te(te, name, "predict()")

    out_pc = xl.predict(X=X, p=p_scores, return_components=True)
    assert (
        isinstance(out_pc, tuple) and len(out_pc) == 3
    ), f"{name}.predict(return_components=True) must return (te, dhat_cs, dhat_ts)"
    te2, dhat_cs, dhat_ts = out_pc
    np.testing.assert_array_equal(te, te2, err_msg=f"{name}: predict inconsistency")
    for label, d in [("dhat_cs", dhat_cs), ("dhat_ts", dhat_ts)]:
        assert isinstance(d, dict) and set(d.keys()) == set(
            xl.t_groups
        ), f"{name}: {label} mismatch"

    fp_plain_x = xl.fit_predict(X=X, treatment=treatment, y=y, p=p_scores)
    _assert_plain_fit_predict(fp_plain_x, name)
    _assert_te(fp_plain_x, name, "fit_predict()")
    _assert_ci_triple(
        xl.fit_predict(
            X=X,
            treatment=treatment,
            y=y,
            p=p_scores,
            return_ci=True,
            n_bootstraps=5,
            bootstrap_size=150,
        ),
        name,
        "fit_predict",
    )
    _assert_ate(xl.estimate_ate(X=X, treatment=treatment, y=y, p=p_scores), name)
    _assert_ate(
        xl.estimate_ate(X=X, treatment=treatment, y=y, p=p_scores, pretrain=True), name
    )

    # ── S-Learner ─────────────────────────────────────────────────────────────
    name = "BaseSLearner"
    sl = BaseSLearner(learner=LinearRegression())
    _assert_fit_returns_self(sl.fit(X=X, treatment=treatment, y=y), sl, name)

    _assert_fit_attrs(sl, name)
    assert hasattr(sl, "models") and isinstance(
        sl.models, dict
    ), f"{name}: missing models dict"
    assert set(sl.models.keys()) == set(sl.t_groups)
    # Each group's model is trained on different data so must be a distinct object.
    assert all(
        sl.models[g1] is not sl.models[g2]
        for g1, g2 in zip(sl.t_groups[:-1], sl.t_groups[1:])
    ), f"{name}: models must be distinct per group"

    te = sl.predict(X=X)
    _assert_te(te, name, "predict()")

    out_pc = sl.predict(X=X, return_components=True)
    assert isinstance(out_pc, tuple) and len(out_pc) == 3
    te2, yhat_cs, yhat_ts = out_pc
    np.testing.assert_array_equal(te, te2, err_msg=f"{name}: predict inconsistency")
    _assert_components(yhat_cs, yhat_ts, sl.t_groups, name)

    fp_plain_s = sl.fit_predict(X=X, treatment=treatment, y=y)
    _assert_plain_fit_predict(fp_plain_s, name)
    _assert_te(fp_plain_s, name, "fit_predict()")
    _assert_ci_triple(
        sl.fit_predict(
            X=X,
            treatment=treatment,
            y=y,
            return_ci=True,
            n_bootstraps=5,
            bootstrap_size=150,
        ),
        name,
        "fit_predict",
    )
    ate_only = sl.estimate_ate(X=X, treatment=treatment, y=y, return_ci=False)
    assert isinstance(ate_only, np.ndarray) and ate_only.shape == (
        n_groups,
    ), f"{name}.estimate_ate(return_ci=False) must be shape (n_groups,)"
    _assert_ate(sl.estimate_ate(X=X, treatment=treatment, y=y, return_ci=True), name)
    _assert_ate(
        sl.estimate_ate(X=X, treatment=treatment, y=y, return_ci=True, pretrain=True),
        name,
    )

    # ── DR-Learner ────────────────────────────────────────────────────────────
    name = "BaseDRLearner"
    dr = BaseDRLearner(
        learner=LinearRegression(), treatment_effect_learner=LinearRegression()
    )
    _assert_fit_returns_self(dr.fit(X=X, treatment=treatment, y=y), dr, name)

    _assert_fit_attrs(dr, name)
    # models_mu_c: list of 3 fold models (fold-specific, NOT per-group).
    assert hasattr(dr, "models_mu_c") and isinstance(
        dr.models_mu_c, list
    ), f"{name}: models_mu_c must be a list"
    assert len(dr.models_mu_c) == 3, f"{name}: models_mu_c must have 3 fold models"
    # Per-group outcome and effect models: each a list of 3 fold models.
    for attr in ("models_mu_t", "models_tau"):
        assert hasattr(dr, attr) and isinstance(getattr(dr, attr), dict)
        assert set(getattr(dr, attr).keys()) == set(dr.t_groups)
        for g in dr.t_groups:
            val = getattr(dr, attr)[g]
            assert (
                isinstance(val, list) and len(val) == 3
            ), f"{name}: {attr}[{g}] must be list of 3 fold models"

    te = dr.predict(X=X)
    _assert_te(te, name, "predict()")

    out_pc = dr.predict(X=X, return_components=True)
    assert isinstance(out_pc, tuple) and len(out_pc) == 3
    te2, yhat_cs, yhat_ts = out_pc
    np.testing.assert_array_equal(te, te2, err_msg=f"{name}: predict inconsistency")
    _assert_components(yhat_cs, yhat_ts, dr.t_groups, name)
    # yhat_cs must be a shared-reference dict (one fold-averaged control prediction).
    assert all(
        yhat_cs[g] is yhat_cs[dr.t_groups[0]] for g in dr.t_groups
    ), f"{name}: yhat_cs values must share the same underlying array"

    fp_plain_dr = dr.fit_predict(X=X, treatment=treatment, y=y)
    _assert_plain_fit_predict(fp_plain_dr, name)
    _assert_te(fp_plain_dr, name, "fit_predict()")
    _assert_ci_triple(
        dr.fit_predict(
            X=X,
            treatment=treatment,
            y=y,
            return_ci=True,
            n_bootstraps=5,
            bootstrap_size=150,
        ),
        name,
        "fit_predict",
    )
    _assert_ate(dr.estimate_ate(X=X, treatment=treatment, y=y), name)
    _assert_ate(dr.estimate_ate(X=X, treatment=treatment, y=y, pretrain=True), name)

    # ── R-Learner ─────────────────────────────────────────────────────────────
    name = "BaseRLearner"
    rl = BaseRLearner(
        learner=LinearRegression(),
        effect_learner=LinearRegression(),
        cv_n_jobs=1,
    )
    _assert_fit_returns_self(
        rl.fit(X=X, treatment=treatment, y=y, p=p_scores, verbose=False), rl, name
    )

    _assert_fit_attrs(rl, name)
    # R-learner: single shared outcome model fitted once via cross-validation.
    assert hasattr(rl, "model_mu"), f"{name}: missing model_mu"
    assert hasattr(rl, "models_tau") and isinstance(rl.models_tau, dict)
    assert set(rl.models_tau.keys()) == set(rl.t_groups)
    assert all(
        rl.models_tau[g1] is not rl.models_tau[g2]
        for g1, g2 in zip(rl.t_groups[:-1], rl.t_groups[1:])
    ), f"{name}: models_tau must be distinct per group"
    for attr in ("vars_c", "vars_t"):
        assert hasattr(rl, attr) and isinstance(getattr(rl, attr), dict)
        assert set(getattr(rl, attr).keys()) == set(rl.t_groups)

    # R-learner: predict(return_components=True) returns
    # (te, yhat, propensity) where yhat and propensity are
    # the nuisance components used by the R-learner.
    te = rl.predict(X=X, p=p_scores)
    _assert_te(te, name, "predict()")

    out_pc = rl.predict(X=X, p=p_scores, return_components=True)
    assert isinstance(out_pc, tuple) and len(out_pc) == 3

    te2, yhat, p = out_pc

    np.testing.assert_array_equal(
        te,
        te2,
        err_msg=f"{name}: predict inconsistency",
    )

    assert isinstance(yhat, np.ndarray)
    assert yhat.shape == (n,)
    assert np.all(np.isfinite(yhat))

    assert isinstance(p, dict)
    assert set(p.keys()) == set(rl.t_groups)

    for g in rl.t_groups:
        assert isinstance(p[g], np.ndarray)
        assert p[g].shape == (n,)
        assert np.all(np.isfinite(p[g]))

    fp_plain_r = rl.fit_predict(
        X=X,
        treatment=treatment,
        y=y,
        p=p_scores,
        verbose=False,
    )

    _assert_plain_fit_predict(fp_plain_r, name)
    _assert_te(fp_plain_r, name, "fit_predict()")

    fp_components = rl.fit_predict(
        X=X,
        treatment=treatment,
        y=y,
        p=p_scores,
        return_components=True,
        verbose=False,
    )

    assert isinstance(fp_components, tuple)
    assert len(fp_components) == 3

    te_fp, yhat_fp, p_fp = fp_components

    _assert_te(te_fp, name, "fit_predict(return_components=True)")

    assert isinstance(yhat_fp, np.ndarray)
    assert yhat_fp.shape == (n,)
    assert np.all(np.isfinite(yhat_fp))

    assert isinstance(p_fp, dict)
    assert set(p_fp.keys()) == set(rl.t_groups)

    for g in rl.t_groups:
        assert isinstance(p_fp[g], np.ndarray)
        assert p_fp[g].shape == (n,)
        assert np.all(np.isfinite(p_fp[g]))
    with pytest.raises(ValueError):
        rl.fit_predict(
            X=X,
            treatment=treatment,
            y=y,
            p=p_scores,
            return_ci=True,
            return_components=True,
            verbose=False,
        )
    with pytest.raises(ValueError):
        rl.predict(
            X=X,
            p=p_scores,
            return_ci=True,
            return_components=True,
        )
    _assert_ci_triple(
        rl.fit_predict(
            X=X,
            treatment=treatment,
            y=y,
            p=p_scores,
            return_ci=True,
            n_bootstraps=5,
            bootstrap_size=150,
            verbose=False,
        ),
        name,
        "fit_predict",
    )
    _assert_ate(rl.estimate_ate(X=X, treatment=treatment, y=y, p=p_scores), name)
    _assert_ate(
        rl.estimate_ate(X=X, treatment=treatment, y=y, p=p_scores, pretrain=True), name
    )


def test_BaseTClassifier_predict_return_ci(generate_classification_data):
    np.random.seed(RANDOM_SEED)

    df, x_names = generate_classification_data()
    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )

    X = df[x_names].values
    treatment = df["treatment_group_key"].values
    y = df[CONVERSION].values

    learner = BaseTClassifier(learner=LogisticRegression(), control_name=0)

    # Test 1: return_ci=True returns (te, lb, ub)
    learner.fit(
        X,
        treatment,
        y,
        store_bootstraps=True,
        n_bootstraps=50,
        bootstrap_size=500,
        random_state=RANDOM_SEED,
    )
    tau, lb, ub = learner.predict(X, return_ci=True)
    assert tau.shape == (X.shape[0], len(learner.t_groups))
    assert lb.shape == tau.shape
    assert ub.shape == tau.shape
    assert (lb <= ub).all()

    # Test 2: ValueError without store_bootstraps
    learner2 = BaseTClassifier(learner=LogisticRegression(), control_name=0)
    learner2.fit(X, treatment, y)
    with pytest.raises(ValueError):
        learner2.predict(X, return_ci=True)

    # Test 3: return_ci + return_components raises ValueError
    with pytest.raises(ValueError):
        learner.predict(X, return_ci=True, return_components=True)

    # Test 4: old API unchanged
    tau_plain = learner.predict(X)
    assert tau_plain.shape == (X.shape[0], len(learner.t_groups))


# =============================================================================
# sklearn compliance tests — issue #911
# =============================================================================

# Fixed-seed data (independent of conftest fixtures so tests are self-contained)
_RNG = np.random.RandomState(42)
_N = 400
_X = _RNG.randn(_N, 5)
_TREATMENT = _RNG.choice([0, 1], _N)
_Y_CONT = _X[:, 0] + _TREATMENT * 0.5 + _RNG.randn(_N) * 0.1
_Y_BIN = (_Y_CONT > 0).astype(int)
_DT_REG = DecisionTreeRegressor(random_state=0, max_depth=3)
_DT_CLF = DecisionTreeClassifier(random_state=0, max_depth=3)


def _sklearn_fit(learner, y=None):
    """Fit with the right outcome type; returns whatever fit() returns."""
    if y is None:
        y = _Y_BIN if "Classifier" in type(learner).__name__ else _Y_CONT
    return learner.fit(_X, _TREATMENT, y)


def _sklearn_predict(learner, **kw):
    """Call predict(), omitting verbose for R-learners which don't accept it."""
    from causalml.inference.meta.rlearner import BaseRLearner

    if isinstance(learner, BaseRLearner):
        return learner.predict(_X, **kw)
    return learner.predict(_X, verbose=False, **kw)


def _params_repr_equal(p1, p2):
    """Compare get_params() dicts via repr() — safe for numpy arrays and estimators."""
    if p1.keys() != p2.keys():
        return False
    return all(repr(p1[k]) == repr(p2[k]) for k in p1)


_REGRESSOR_CONFIGS = [
    (BaseSRegressor, {"learner": _DT_REG}),
    (BaseTRegressor, {"learner": _DT_REG}),
    (BaseXRegressor, {"learner": _DT_REG}),
    (BaseDRRegressor, {"learner": _DT_REG}),
    (BaseRRegressor, {"learner": _DT_REG}),
]

_CLASSIFIER_CONFIGS = [
    (BaseSClassifier, {"learner": _DT_CLF}),
    (BaseTClassifier, {"learner": _DT_CLF}),
    (BaseXClassifier, {"outcome_learner": _DT_CLF, "effect_learner": _DT_REG}),
]


@pytest.mark.parametrize("Cls,kwargs", _REGRESSOR_CONFIGS)
def test_clone_get_params_regressor(Cls, kwargs):
    """clone() / get_params() round-trip — all regressors."""
    m = Cls(**kwargs)
    params = m.get_params()
    assert isinstance(params, dict)
    m2 = clone(m)
    assert type(m2) is type(m)
    assert _params_repr_equal(
        m2.get_params(), params
    ), f"{Cls.__name__}: clone params mismatch"
    assert not hasattr(m2, "t_groups"), "clone() must return an unfitted estimator"


@pytest.mark.parametrize("Cls,kwargs", _CLASSIFIER_CONFIGS)
def test_clone_get_params_classifier(Cls, kwargs):
    """clone() / get_params() round-trip — all classifiers (botched-merge check)."""
    m = Cls(**kwargs)
    params = m.get_params()
    assert isinstance(params, dict)
    m2 = clone(m)
    assert type(m2) is type(m)
    assert _params_repr_equal(m2.get_params(), params)
    assert not hasattr(m2, "t_groups")


@pytest.mark.parametrize("Cls,kwargs", _REGRESSOR_CONFIGS)
def test_fit_returns_self_regressor(Cls, kwargs):
    """fit() returns self for all regressors (Pipeline / GridSearchCV requirement)."""
    m = Cls(**kwargs)
    assert _sklearn_fit(m) is m, f"{Cls.__name__}.fit() must return self"


@pytest.mark.parametrize("Cls,kwargs", _CLASSIFIER_CONFIGS)
def test_fit_returns_self_classifier(Cls, kwargs):
    """fit() returns self for all classifiers."""
    m = Cls(**kwargs)
    assert _sklearn_fit(m) is m, f"{Cls.__name__}.fit() must return self"


def test_xgb_rregressor_clone_no_kwargs():
    """XGBRRegressor() (no extra kwargs) clones cleanly; xgb_kwargs stored verbatim."""
    r = XGBRRegressor()
    params = r.get_params()
    assert "xgb_kwargs" in params and params["xgb_kwargs"] is None
    r2 = clone(r)
    assert r2.xgb_kwargs is None
    assert _params_repr_equal(r2.get_params(), r.get_params())


def test_xgb_rregressor_clone_with_kwargs():
    """XGBRRegressor(xgb_kwargs={...}) round-trips through clone / get_params."""
    r = XGBRRegressor(xgb_kwargs={"max_depth": 3, "learning_rate": 0.05})
    assert r.get_params()["xgb_kwargs"] == {"max_depth": 3, "learning_rate": 0.05}
    r2 = clone(r)
    assert r2.xgb_kwargs == {"max_depth": 3, "learning_rate": 0.05}
    assert _params_repr_equal(r2.get_params(), r.get_params())


def test_xgb_rregressor_fit_predict_return_ci():
    """XGBRRegressor.fit_predict(return_ci=True) exercises clone(self) in bootstrap.

    Before the BaseEstimator refactor, clone(self, safe=False) silently deepcopied
    a fitted model on every bootstrap iteration.
    """
    r = XGBRRegressor(effect_learner_n_estimators=20, random_state=0)
    te, lo, hi = r.fit_predict(
        _X, _TREATMENT, _Y_CONT, return_ci=True, n_bootstraps=5, bootstrap_size=200
    )
    assert te.shape == lo.shape == hi.shape == (_N, 1)
    assert np.all(lo <= hi), "CI lower bound must be <= upper bound"


@pytest.mark.parametrize(
    "Cls,kwargs",
    [
        (BaseSRegressor, {"learner": _DT_REG}),
        (BaseTRegressor, {"learner": _DT_REG}),
        (BaseXRegressor, {"learner": _DT_REG}),
    ],
)
def test_bit_identical_predict(Cls, kwargs):
    """Two independently fitted clones produce identical predictions (deterministic learners)."""
    m1 = Cls(**kwargs)
    _sklearn_fit(m1)
    te1 = _sklearn_predict(m1)
    m2 = clone(m1)
    _sklearn_fit(m2)
    te2 = _sklearn_predict(m2)
    np.testing.assert_array_equal(
        te1,
        te2,
        err_msg=f"{Cls.__name__}: predict not bit-identical across clone+refit",
    )


@pytest.mark.parametrize(
    "Cls,kwargs",
    [(BaseDRRegressor, {"learner": _DT_REG}), (BaseRRegressor, {"learner": _DT_REG})],
)
def test_finite_predict_stochastic(Cls, kwargs):
    """DR and R learners (internal shuffle-splits) produce finite predictions."""
    m1 = Cls(**kwargs)
    _sklearn_fit(m1)
    te1 = _sklearn_predict(m1)
    m2 = clone(m1)
    _sklearn_fit(m2)
    te2 = _sklearn_predict(m2)
    assert te1.shape == te2.shape
    assert np.isfinite(te1).all() and np.isfinite(te2).all()


def test_bit_identical_estimate_ate_s():
    """S-learner estimate_ate() CI triple is bit-identical across clone+refit."""
    m1 = BaseSRegressor(learner=_DT_REG)
    ate1, lb1, ub1 = m1.estimate_ate(_X, _TREATMENT, _Y_CONT, return_ci=True)
    m2 = clone(m1)
    ate2, lb2, ub2 = m2.estimate_ate(_X, _TREATMENT, _Y_CONT, return_ci=True)
    np.testing.assert_array_equal(ate1, ate2)
    np.testing.assert_array_equal(lb1, lb2)
    np.testing.assert_array_equal(ub1, ub2)


def test_t_regressor_return_ci_bit_identical():
    """TLearner stored bootstrap ensemble is reproducible with fixed random_state."""
    kw = dict(store_bootstraps=True, n_bootstraps=10, random_state=7)
    m1 = BaseTRegressor(learner=_DT_REG)
    m1.fit(_X, _TREATMENT, _Y_CONT, **kw)
    te1, lo1, hi1 = m1.predict(_X, return_ci=True, verbose=False)
    m2 = BaseTRegressor(learner=_DT_REG)
    m2.fit(_X, _TREATMENT, _Y_CONT, **kw)
    te2, lo2, hi2 = m2.predict(_X, return_ci=True, verbose=False)
    np.testing.assert_array_equal(te1, te2)
    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(hi1, hi2)


def test_t_classifier_clone_fit_predict():
    """BaseTClassifier clone round-trip and fit/predict smoke test."""
    m = BaseTClassifier(learner=_DT_CLF)
    m2 = clone(m)
    assert _params_repr_equal(m2.get_params(), m.get_params())
    _sklearn_fit(m2)
    assert _sklearn_predict(m2).shape == (_N, 1)


def test_x_classifier_verbatim_store_clone():
    """BaseXClassifier stores outcome/effect learner verbatim (not pre-resolved)."""
    m = BaseXClassifier(outcome_learner=_DT_CLF, effect_learner=_DT_REG)
    params = m.get_params()
    assert params["outcome_learner"] is _DT_CLF
    assert params["control_outcome_learner"] is None  # verbatim, not resolved
    m2 = clone(m)
    assert type(m2.outcome_learner) is type(_DT_CLF)
    assert m2.outcome_learner is not _DT_CLF  # cloned copy
    assert m2.control_outcome_learner is None
    _sklearn_fit(m2)
    assert _sklearn_predict(m2).shape == (_N, 1)


def test_t_classifier_fail_fast_return_ci_and_components():
    """BaseTClassifier.predict raises before any computation when both flags set."""
    m = BaseTClassifier(learner=_DT_CLF)
    _sklearn_fit(m)
    with pytest.raises(ValueError, match="cannot both be True"):
        m.predict(_X, return_ci=True, return_components=True, verbose=False)


def test_x_learner_pretrain_before_fit_raises():
    """BaseXRegressor.estimate_ate(pretrain=True) before fit() raises ValueError."""
    with pytest.raises(ValueError):
        BaseXRegressor(learner=_DT_REG).estimate_ate(
            _X, _TREATMENT, _Y_CONT, pretrain=True
        )


def test_dr_learner_pretrain_before_fit_raises():
    """BaseDRRegressor.estimate_ate(pretrain=True) before fit() raises ValueError.

    self.propensity = {} sentinel in __init__ (consistent with BaseXLearner) ensures
    a clean ValueError rather than an AttributeError on missing t_groups.
    """
    with pytest.raises(ValueError):
        BaseDRRegressor(learner=_DT_REG).estimate_ate(
            _X, _TREATMENT, _Y_CONT, pretrain=True
        )
