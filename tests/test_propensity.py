import numpy as np
import pytest

from causalml import propensity
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


def test_gradientboosted_propensity_model_earlystopping_reproducible(
    generate_regression_data,
):
    """Early stopping is reproducible: the validation split is seeded (#1045)."""
    y, X, treatment, tau, b, e = generate_regression_data()

    def fit_predict(random_state):
        pm = GradientBoostedPropensityModel(random_state=random_state, early_stop=True)
        return pm.fit_predict(X, treatment)

    np.testing.assert_array_equal(fit_predict(RANDOM_SEED), fit_predict(RANDOM_SEED))
    # A different seed must still move the split; otherwise the seed would be
    # ignored in a different way (e.g. a hard-coded constant).
    assert not np.array_equal(fit_predict(RANDOM_SEED), fit_predict(RANDOM_SEED + 1))


def test_gradientboosted_propensity_model_earlystopping_stratified(monkeypatch):
    """The early-stopping validation split keeps both treatment arms (#1045)."""
    rng = np.random.RandomState(RANDOM_SEED)
    X = rng.normal(size=(400, 10))
    treatment = (rng.uniform(size=400) < 0.1).astype(int)

    captured = {}
    train_test_split = propensity.train_test_split

    def spy(*args, **kwargs):
        split = train_test_split(*args, **kwargs)
        captured["y_val"] = split[3]
        return split

    monkeypatch.setattr(propensity, "train_test_split", spy)

    pm = GradientBoostedPropensityModel(random_state=RANDOM_SEED, early_stop=True)
    pm.fit(X, treatment)

    # Stratification preserves the treatment rate up to rounding.
    assert captured["y_val"].mean() == pytest.approx(treatment.mean(), abs=0.01)


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
