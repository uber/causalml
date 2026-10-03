import numpy as np

from causalml.propensity import (
    ElasticNetPropensityModel,
    GradientBoostedPropensityModel,
    LogisticRegressionPropensityModel,
)
from causalml.metrics import roc_auc_score


from .const import RANDOM_SEED


def test_logistic_regression_propensity_model(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    pm = LogisticRegressionPropensityModel(random_state=RANDOM_SEED)
    ps = pm.fit_predict(X, treatment)

    assert roc_auc_score(treatment, ps) > 0.5


def test_logistic_regression_propensity_model_model_kwargs(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    pm = LogisticRegressionPropensityModel(random_state=123)

    assert pm.model.random_state == 123


def test_logistic_regression_propensity_model_cs_grid():
    # The cross-validated C grid should span both strong (C < 1) and weak
    # (C > 1) regularization so that LogisticRegressionCV can actually tune
    # the penalty. A grid confined to C >= 1 can never pick strong
    # regularization.
    rng = np.random.RandomState(RANDOM_SEED)
    X = rng.normal(size=(200, 5))
    w = (X[:, 0] + rng.normal(size=200) > 0).astype(int)

    pm = ElasticNetPropensityModel(calibrate=False, random_state=RANDOM_SEED)
    pm.fit(X, w)

    grid = pm.model.Cs_
    assert grid.min() < 1 < grid.max()


def test_elasticnet_propensity_model(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    pm = ElasticNetPropensityModel(random_state=RANDOM_SEED)
    ps = pm.fit_predict(X, treatment)

    assert roc_auc_score(treatment, ps) > 0.5


def test_gradientboosted_propensity_model(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    pm = GradientBoostedPropensityModel(random_state=RANDOM_SEED)
    ps = pm.fit_predict(X, treatment)

    assert roc_auc_score(treatment, ps) > 0.5


def test_gradientboosted_propensity_model_earlystopping(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    pm = GradientBoostedPropensityModel(random_state=RANDOM_SEED, early_stop=True)
    ps = pm.fit_predict(X, treatment)

    assert roc_auc_score(treatment, ps) > 0.5


def test_propensity_models_imbalanced_1027():
    rng = np.random.RandomState(RANDOM_SEED)
    X = rng.normal(size=(400, 25))
    logit = 0.3 * X[:, 0] + rng.normal(size=400)
    treatment = (logit > np.quantile(logit, 0.90)).astype(int)

    pm_lr = LogisticRegressionPropensityModel(random_state=RANDOM_SEED)
    pm_lr.fit_predict(X, treatment)
    assert pm_lr.model.C_[0] > 1e-4

    pm_en = ElasticNetPropensityModel(random_state=RANDOM_SEED)
    pm_en.fit_predict(X, treatment)
    assert pm_en.model.C_[0] > 1e-4


def test_gradientboosted_propensity_model_earlystopping_is_reproducible():
    # The early-stopping validation split used to ignore random_state, so the
    # same seed could return different propensity scores on every fit.
    rng = np.random.RandomState(RANDOM_SEED)
    X = rng.normal(size=(1000, 10))
    treatment = (0.8 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=1000) > 0).astype(int)

    runs = []
    for _ in range(3):
        pm = GradientBoostedPropensityModel(random_state=RANDOM_SEED, early_stop=True)
        runs.append(pm.fit_predict(X, treatment))

    for other in runs[1:]:
        np.testing.assert_array_equal(runs[0], other)


def test_gradientboosted_propensity_model_earlystopping_keeps_both_arms():
    # Stratifying on treatment keeps treated units in the validation split even
    # when treatment is rare, so the early-stopping metric stays meaningful.
    from unittest import mock

    from causalml import propensity

    rng = np.random.RandomState(RANDOM_SEED)
    X = rng.normal(size=(200, 5))
    treatment = np.zeros(200, dtype=int)
    treatment[:20] = 1

    real_split = propensity.train_test_split
    with mock.patch.object(propensity, "train_test_split", wraps=real_split) as split:
        pm = GradientBoostedPropensityModel(random_state=RANDOM_SEED, early_stop=True)
        pm.fit(X, treatment)

    kwargs = split.call_args.kwargs
    assert kwargs["random_state"] == RANDOM_SEED
    np.testing.assert_array_equal(kwargs["stratify"], treatment)
