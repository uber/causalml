import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
