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
