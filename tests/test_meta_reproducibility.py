import numpy as np
import pytest
from packaging import version
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from causalml.inference.meta import (
    BaseSLearner,
    BaseSRegressor,
    BaseSClassifier,
    LRSRegressor,
    BaseXLearner,
    BaseXRegressor,
    BaseXClassifier,
    BaseTLearner,
    BaseTRegressor,
    BaseTClassifier,
    XGBTRegressor,
    MLPTRegressor,
    XGBTClassifier,
)


def make_dummy_data():
    np.random.seed(42)
    X = np.random.normal(size=(100, 4))
    w = np.random.binomial(1, 0.5, size=100)
    y = X[:, 0] + 0.5 * w + np.random.normal(scale=0.1, size=100)
    p = np.full(100, 0.5)
    return X, w, y, p


def test_slearner_bootstrap_reproducibility():
    X, w, y, _ = make_dummy_data()
    learner1 = BaseSRegressor(LinearRegression(), random_state=42)
    learner1.fit(X, treatment=w, y=y)
    np.random.rand(100)
    ci1 = learner1.estimate_ate(X, treatment=w, y=y, bootstrap_ci=True, n_bootstraps=5)
    cate1 = learner1.fit_predict(X, treatment=w, y=y, return_ci=True, n_bootstraps=5)

    learner2 = BaseSRegressor(LinearRegression(), random_state=42)
    learner2.fit(X, treatment=w, y=y)
    np.random.rand(100)
    ci2 = learner2.estimate_ate(X, treatment=w, y=y, bootstrap_ci=True, n_bootstraps=5)
    cate2 = learner2.fit_predict(X, treatment=w, y=y, return_ci=True, n_bootstraps=5)

    assert np.allclose(ci1[1], ci2[1])
    assert np.allclose(ci1[2], ci2[2])
    assert np.allclose(cate1[1], cate2[1])
    assert np.allclose(cate1[2], cate2[2])


def test_xlearner_bootstrap_reproducibility():
    X, w, y, p = make_dummy_data()
    learner1 = BaseXRegressor(LinearRegression(), random_state=42)
    learner1.fit(X, treatment=w, y=y, p=p)
    np.random.rand(100)
    ci1 = learner1.estimate_ate(
        X, treatment=w, y=y, p=p, bootstrap_ci=True, n_bootstraps=5
    )
    cate1 = learner1.fit_predict(
        X, treatment=w, y=y, p=p, return_ci=True, n_bootstraps=5
    )

    learner2 = BaseXRegressor(LinearRegression(), random_state=42)
    learner2.fit(X, treatment=w, y=y, p=p)
    np.random.rand(100)
    ci2 = learner2.estimate_ate(
        X, treatment=w, y=y, p=p, bootstrap_ci=True, n_bootstraps=5
    )
    cate2 = learner2.fit_predict(
        X, treatment=w, y=y, p=p, return_ci=True, n_bootstraps=5
    )

    assert np.allclose(ci1[1], ci2[1])
    assert np.allclose(ci1[2], ci2[2])
    assert np.allclose(cate1[1], cate2[1])
    assert np.allclose(cate1[2], cate2[2])


def test_tlearner_bootstrap_reproducibility():
    X, w, y, _ = make_dummy_data()
    learner1 = BaseTRegressor(LinearRegression(), random_state=42)
    learner1.fit(X, treatment=w, y=y)
    np.random.rand(100)
    ci1 = learner1.estimate_ate(X, treatment=w, y=y, bootstrap_ci=True, n_bootstraps=5)
    cate1 = learner1.fit_predict(X, treatment=w, y=y, return_ci=True, n_bootstraps=5)

    learner2 = BaseTRegressor(LinearRegression(), random_state=42)
    learner2.fit(X, treatment=w, y=y)
    np.random.rand(100)
    ci2 = learner2.estimate_ate(X, treatment=w, y=y, bootstrap_ci=True, n_bootstraps=5)
    cate2 = learner2.fit_predict(X, treatment=w, y=y, return_ci=True, n_bootstraps=5)

    assert np.allclose(ci1[1], ci2[1])
    assert np.allclose(ci1[2], ci2[2])
    assert np.allclose(cate1[1], cate2[1])
    assert np.allclose(cate1[2], cate2[2])


@pytest.mark.parametrize(
    "learner_class",
    [
        BaseSLearner,
        BaseSRegressor,
        BaseSClassifier,
        BaseXLearner,
        BaseXRegressor,
        BaseXClassifier,
        BaseTLearner,
        BaseTRegressor,
        BaseTClassifier,
    ],
)
def test_clone_roundtrip(learner_class):
    learner = learner_class(random_state=42)
    params = learner.get_params()
    assert params.get("random_state") == 42

    cloned_learner = clone(learner)
    assert cloned_learner.random_state == 42


@pytest.mark.parametrize(
    "learner_class",
    [
        LRSRegressor,
        XGBTClassifier,
    ],
)
def test_specialized_clone_roundtrip(learner_class):
    learner = learner_class(random_state=42)
    params = learner.get_params()
    assert params.get("random_state") == 42

    cloned_learner = clone(learner)
    assert cloned_learner.random_state == 42


def test_tlearner_predict_ci_constructor_seed():
    X, w, y, _ = make_dummy_data()

    learner1 = BaseTRegressor(LinearRegression(), random_state=42)
    learner1.fit(X, treatment=w, y=y, store_bootstraps=True, n_bootstraps=5)
    np.random.rand(100)
    te1, lb1, ub1 = learner1.predict(X, return_ci=True)

    learner2 = BaseTRegressor(LinearRegression(), random_state=42)
    learner2.fit(X, treatment=w, y=y, store_bootstraps=True, n_bootstraps=5)
    np.random.rand(100)
    te2, lb2, ub2 = learner2.predict(X, return_ci=True)

    assert np.allclose(lb1, lb2)
    assert np.allclose(ub1, ub2)


def test_positional_args_backward_compatibility():
    clf = XGBTClassifier(0.05, 0, {"max_depth": 4})
    assert clf.xgb_kwargs == {"max_depth": 4}
    assert clf.random_state is None

    reg = MLPTRegressor(0.05, 0, (50, 50))
    assert reg.random_state is None
    if version.parse(sklearn.__version__) >= version.parse("1.9.0"):
        assert reg.learner.loss == (50, 50)
    else:
        assert reg.learner.hidden_layer_sizes == (50, 50)

    reg_kw = MLPTRegressor(0.05, 0, hidden_layer_sizes=(50, 50))
    assert reg_kw.learner.hidden_layer_sizes == (50, 50)
    assert reg_kw.random_state is None

    clf_seeded = XGBTClassifier(0.05, 0, {"max_depth": 4}, random_state=42)
    assert clf_seeded.xgb_kwargs == {"max_depth": 4}
    assert clf_seeded.random_state == 42

    reg_seeded = MLPTRegressor(0.05, 0, random_state=42)
    assert reg_seeded.random_state == 42
