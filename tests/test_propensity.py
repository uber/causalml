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


def test_compute_propensity_score_clips_a_user_supplied_model():
    """clip_bounds has to hold for any model, not only the built-in ones.

    ``PropensityModel.predict`` clips, but a plain scikit-learn classifier does
    not, and a score of exactly 0 or 1 divides by zero in the DR-learner and in
    TMLE, and fails ``check_p_conditions``.
    """
    from sklearn.linear_model import LogisticRegression

    from causalml.propensity import compute_propensity_score

    rng = np.random.RandomState(RANDOM_SEED)
    n = 400
    X = rng.normal(size=(n, 3))
    # perfectly separable, so an unregularised classifier saturates at 0 and 1
    treatment = (X[:, 0] > 0).astype(int)

    clip_bounds = (1e-3, 1 - 1e-3)
    p, _ = compute_propensity_score(
        X=X,
        treatment=treatment,
        p_model=LogisticRegression(C=1e9, max_iter=1000),
        clip_bounds=clip_bounds,
    )

    assert p.min() >= clip_bounds[0]
    assert p.max() <= clip_bounds[1]


def test_compute_propensity_score_honors_custom_clip_bounds():
    from causalml.propensity import compute_propensity_score

    rng = np.random.RandomState(RANDOM_SEED)
    n = 400
    X = rng.normal(size=(n, 3))
    treatment = (X[:, 0] > 0).astype(int)

    clip_bounds = (0.2, 0.8)
    p, _ = compute_propensity_score(X=X, treatment=treatment, clip_bounds=clip_bounds)

    assert p.min() >= clip_bounds[0]
    assert p.max() <= clip_bounds[1]
