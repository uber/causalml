import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost import XGBRegressor
import warnings

from causalml.inference.iv import BaseDRIVLearner
from causalml.metrics import ape, get_cumgain

from .const import RANDOM_SEED, N_SAMPLE, ERROR_THRESHOLD, CONTROL_NAME, CONVERSION

def test_drivlearner():
    np.random.seed(RANDOM_SEED)
    n = 1000
    p = 8
    sigma = 1.0

    X = np.random.uniform(size=n*p).reshape((n, -1))
    b = np.sin(np.pi * X[:, 0] * X[:, 1]) + 2 * (X[:, 2] - 0.5) ** 2 + X[:, 3] + 0.5 * X[:, 4]
    assignment = (np.random.uniform(size=n)>0.5).astype(int)
    eta = 0.1
    e_raw = np.maximum(np.repeat(eta, n), np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1-eta, n)))
    e = e_raw.copy()
    e[assignment == 0] = 0
    tau = (X[:, 0] + X[:, 1]) / 2
    X_obs = X[:, [i for i in range(8) if i!=1]]

    w = np.random.binomial(1, e, size=n)
    treatment = w
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    learner = BaseDRIVLearner(learner=XGBRegressor(), treatment_effect_learner=LinearRegression())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, assignment=assignment, treatment=treatment, y=y, p=(np.ones(n)*1e-6, e_raw))
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(X=X, assignment=assignment, treatment=treatment, y=y, p=(np.ones(n)*1e-6, e_raw), return_ci=True, n_bootstraps=10)

    auuc_metrics = pd.DataFrame({'cate_p': cate_p.flatten(),
                                 'W': treatment,
                                 'y': y,
                                 'treatment_effect_col': tau})

    cumgain = get_cumgain(auuc_metrics,
                          outcome_col='y',
                          treatment_col='W',
                          treatment_effect_col='tau')

    # Check if the cumulative gain when using the model's prediction is
    # higher than it would be under random targeting
    assert cumgain['cate_p'].sum() > cumgain['Random'].sum()
