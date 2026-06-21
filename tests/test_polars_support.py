"""
Tests for Polars DataFrame/Series support across CausalML meta-learners.

Run with:
    pytest tests/test_polars_support.py -v --noconftest
"""

import pytest
import numpy as np
import pandas as pd

# Skip the entire module if polars is not installed
polars = pytest.importorskip("polars", reason="polars is not installed")
import polars as pl

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
from causalml.inference.meta.tlearner import BaseTClassifier
from causalml.inference.meta.slearner import BaseSClassifier
from causalml.inference.meta.xlearner import BaseXClassifier
from causalml.inference.meta.tlearner import BaseTRegressor
from causalml.inference.meta.slearner import BaseSRegressor
from causalml.inference.meta.xlearner import BaseXRegressor
from causalml.inference.meta.rlearner import BaseRRegressor
from causalml.inference.meta.drlearner import BaseDRRegressor
from causalml.inference.meta.utils import convert_pd_to_np, check_p_conditions

# Fixtures

N = 200
N_FEATURES = 5
RANDOM_STATE = 42


@pytest.fixture(scope="module")
def synthetic_data_numpy():
    """Return (X, treatment, y) as NumPy arrays — the baseline."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N, N_FEATURES))
    treatment = rng.choice([0, 1], size=N)
    y = X[:, 0] * treatment + rng.standard_normal(N) * 0.1
    return X, treatment, y


@pytest.fixture(scope="module")
def synthetic_data_pandas(synthetic_data_numpy):
    """Return (X, treatment, y) as pandas objects."""
    X_np, t_np, y_np = synthetic_data_numpy
    X = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(N_FEATURES)])
    treatment = pd.Series(t_np, name="treatment")
    y = pd.Series(y_np, name="outcome")
    return X, treatment, y


@pytest.fixture(scope="module")
def synthetic_data_polars(synthetic_data_numpy):
    """Return (X, treatment, y) as polars objects."""
    X_np, t_np, y_np = synthetic_data_numpy
    X = pl.DataFrame({f"f{i}": X_np[:, i] for i in range(N_FEATURES)})
    treatment = pl.Series("treatment", t_np)
    y = pl.Series("outcome", y_np)
    return X, treatment, y


@pytest.fixture(scope="module")
def synthetic_data_polars_lazy(synthetic_data_polars):
    """Return X as a polars LazyFrame, treatment and y as Series."""
    X, treatment, y = synthetic_data_polars
    return X.lazy(), treatment, y


# convert_pd_to_np unit tests


class TestConvertPdToNp:
    def test_numpy_passthrough(self, synthetic_data_numpy):
        X, t, y = synthetic_data_numpy
        X_out, t_out, y_out = convert_pd_to_np(X, t, y)
        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(t_out, t)

    def test_pandas_conversion(self, synthetic_data_pandas):
        X, t, y = synthetic_data_pandas
        X_out, t_out, y_out = convert_pd_to_np(X, t, y)
        assert isinstance(X_out, np.ndarray)
        assert isinstance(t_out, np.ndarray)
        assert isinstance(y_out, np.ndarray)

    def test_polars_dataframe_conversion(self, synthetic_data_polars):
        X, t, y = synthetic_data_polars
        X_out, t_out, y_out = convert_pd_to_np(X, t, y)
        assert isinstance(X_out, np.ndarray)
        assert X_out.shape == (N, N_FEATURES)
        assert isinstance(t_out, np.ndarray)
        assert t_out.shape == (N,)
        assert isinstance(y_out, np.ndarray)

    def test_polars_lazyframe_conversion(self, synthetic_data_polars_lazy):
        X_lazy, t, y = synthetic_data_polars_lazy
        X_out = convert_pd_to_np(X_lazy)
        assert isinstance(X_out, np.ndarray)
        assert X_out.shape == (N, N_FEATURES)

    def test_none_passthrough(self):
        result = convert_pd_to_np(None)
        assert result is None

    def test_single_arg(self, synthetic_data_polars):
        X, _, _ = synthetic_data_polars
        X_out = convert_pd_to_np(X)
        assert isinstance(X_out, np.ndarray)

    def test_single_column_polars_df_is_1d(self):
        """A single-column pl.DataFrame should be squeezed to 1-D."""
        s = pl.DataFrame({"a": [1, 2, 3]})
        out = convert_pd_to_np(s)
        assert out.ndim == 1

    def test_multi_column_polars_df_is_2d(self, synthetic_data_polars):
        X, _, _ = synthetic_data_polars
        out = convert_pd_to_np(X)
        assert out.ndim == 2


# check_p_conditions with polars Series


class TestCheckPConditions:
    def test_polars_series_accepted(self):
        t_groups = np.array([1])
        p = pl.Series("p", np.linspace(0.1, 0.9, N))
        # Should not raise
        check_p_conditions(p, t_groups)

    def test_polars_series_out_of_bounds_raises(self):
        t_groups = np.array([1])
        p = pl.Series("p", np.linspace(0.0, 1.0, N))  # includes 0 and 1
        with pytest.raises(AssertionError):
            check_p_conditions(p, t_groups)


# Helper: assert predictions are close between two input formats


def _assert_te_close(te_ref, te_other, atol=1e-5):
    """Treatment-effect arrays from two input formats should be identical."""
    np.testing.assert_allclose(
        te_ref,
        te_other,
        atol=atol,
        err_msg="Treatment effects differ between input formats",
    )


# T-Learner


class TestTLearnerPolars:
    @pytest.fixture(autouse=True)
    def _learner(self):
        self.learner = BaseTRegressor(learner=LinearRegression())

    def _fit_predict(self, X, treatment, y):
        self.learner.fit(X, treatment, y)
        return self.learner.predict(X)

    def test_polars_matches_numpy(self, synthetic_data_numpy, synthetic_data_polars):
        te_np = self._fit_predict(*synthetic_data_numpy)
        self.learner = BaseTRegressor(learner=LinearRegression())
        te_pl = self._fit_predict(*synthetic_data_polars)
        _assert_te_close(te_np, te_pl)

    def test_polars_matches_pandas(self, synthetic_data_pandas, synthetic_data_polars):
        te_pd = self._fit_predict(*synthetic_data_pandas)
        self.learner = BaseTRegressor(learner=LinearRegression())
        te_pl = self._fit_predict(*synthetic_data_polars)
        _assert_te_close(te_pd, te_pl)

    def test_lazyframe_input(self, synthetic_data_numpy, synthetic_data_polars_lazy):
        te_np = self._fit_predict(*synthetic_data_numpy)
        self.learner = BaseTRegressor(learner=LinearRegression())
        te_pl = self._fit_predict(*synthetic_data_polars_lazy)
        _assert_te_close(te_np, te_pl)

    def test_fit_predict_returns_numpy(self, synthetic_data_polars):
        te = self._fit_predict(*synthetic_data_polars)
        assert isinstance(te, np.ndarray)

    def test_estimate_ate_polars(self, synthetic_data_polars):
        X, treatment, y = synthetic_data_polars
        ate, lb, ub = self.learner.estimate_ate(X, treatment, y)
        assert isinstance(ate, np.ndarray)
        assert lb[0] < ate[0] < ub[0]


# S-Learner


class TestSLearnerPolars:
    @pytest.fixture(autouse=True)
    def _learner(self):
        self.learner = BaseSRegressor(learner=LinearRegression())

    def _fit_predict(self, X, treatment, y):
        self.learner.fit(X, treatment, y)
        return self.learner.predict(X)

    def test_polars_matches_numpy(self, synthetic_data_numpy, synthetic_data_polars):
        te_np = self._fit_predict(*synthetic_data_numpy)
        self.learner = BaseSRegressor(learner=LinearRegression())
        te_pl = self._fit_predict(*synthetic_data_polars)
        _assert_te_close(te_np, te_pl)

    def test_fit_predict_returns_numpy(self, synthetic_data_polars):
        te = self._fit_predict(*synthetic_data_polars)
        assert isinstance(te, np.ndarray)

    def test_estimate_ate_polars(self, synthetic_data_polars):
        X, treatment, y = synthetic_data_polars
        # BaseSLearner.estimate_ate returns only `ate` when return_ci=False (default)
        ate = self.learner.estimate_ate(X, treatment, y)
        assert isinstance(ate, np.ndarray)
        # With return_ci=True it returns (ate, lb, ub)
        ate, lb, ub = self.learner.estimate_ate(X, treatment, y, return_ci=True)
        assert lb[0] < ate[0] < ub[0]


# X-Learner


class TestXLearnerPolars:
    @pytest.fixture(autouse=True)
    def _learner(self):
        self.learner = BaseXRegressor(learner=LinearRegression())

    def _fit_predict(self, X, treatment, y):
        self.learner.fit(X, treatment, y)
        return self.learner.predict(X)

    def test_polars_matches_numpy(self, synthetic_data_numpy, synthetic_data_polars):
        te_np = self._fit_predict(*synthetic_data_numpy)
        self.learner = BaseXRegressor(learner=LinearRegression())
        te_pl = self._fit_predict(*synthetic_data_polars)
        _assert_te_close(te_np, te_pl)

    def test_fit_predict_returns_numpy(self, synthetic_data_polars):
        te = self._fit_predict(*synthetic_data_polars)
        assert isinstance(te, np.ndarray)


# R-Learner


class TestRLearnerPolars:
    @pytest.fixture(autouse=True)
    def _learner(self):
        # fixed random_state so KFold splits are identical across both runs
        self.learner = BaseRRegressor(
            learner=LinearRegression(), random_state=RANDOM_STATE
        )

    def _fit_predict(self, X, treatment, y):
        self.learner.fit(X, treatment, y)
        return self.learner.predict(X)

    def test_polars_matches_numpy(self, synthetic_data_numpy, synthetic_data_polars):
        # With a fixed random_state the KFold splits are deterministic,
        # so numpy and polars inputs must produce identical results.
        te_np = self._fit_predict(*synthetic_data_numpy)
        self.learner = BaseRRegressor(
            learner=LinearRegression(), random_state=RANDOM_STATE
        )
        te_pl = self._fit_predict(*synthetic_data_polars)
        _assert_te_close(te_np, te_pl)

    def test_fit_predict_returns_numpy(self, synthetic_data_polars):
        te = self._fit_predict(*synthetic_data_polars)
        assert isinstance(te, np.ndarray)


# DR-Learner


class TestDRLearnerPolars:
    @pytest.fixture(autouse=True)
    def _learner(self):
        self.learner = BaseDRRegressor(learner=LinearRegression())

    def _fit_predict(self, X, treatment, y, seed=None):
        self.learner.fit(X, treatment, y, seed=seed)
        return self.learner.predict(X)

    def test_polars_matches_numpy(self, synthetic_data_numpy, synthetic_data_polars):
        # DR-Learner uses KFold with a seed parameter passed to fit(); fix it
        # so both runs use the same splits.
        te_np = self._fit_predict(*synthetic_data_numpy, seed=RANDOM_STATE)
        self.learner = BaseDRRegressor(learner=LinearRegression())
        te_pl = self._fit_predict(*synthetic_data_polars, seed=RANDOM_STATE)
        _assert_te_close(te_np, te_pl)

    def test_fit_predict_returns_numpy(self, synthetic_data_polars):
        te = self._fit_predict(*synthetic_data_polars)
        assert isinstance(te, np.ndarray)


# Edge cases


class TestEdgeCases:
    def test_mixed_inputs_polars_x_numpy_treatment(
        self, synthetic_data_numpy, synthetic_data_polars
    ):
        """X as polars DataFrame, treatment and y as numpy — should work fine."""
        X_pl, _, _ = synthetic_data_polars
        _, t_np, y_np = synthetic_data_numpy
        learner = BaseTRegressor(learner=LinearRegression())
        learner.fit(X_pl, t_np, y_np)
        te = learner.predict(X_pl)
        assert isinstance(te, np.ndarray)

    def test_polars_predict_only(self, synthetic_data_numpy, synthetic_data_polars):
        """Fit on numpy, predict on polars — predict must accept polars X."""
        X_np, t_np, y_np = synthetic_data_numpy
        X_pl, _, _ = synthetic_data_polars
        learner = BaseTRegressor(learner=LinearRegression())
        learner.fit(X_np, t_np, y_np)
        te = learner.predict(X_pl)
        assert isinstance(te, np.ndarray)
        assert te.shape[0] == N


# T-Learner Classifier


class TestTClassifierPolars:
    @pytest.fixture(autouse=True)
    def _learner(self):
        self.learner = BaseTClassifier(learner=LogisticRegression())

    def _fit_predict(self, X, treatment, y):
        self.learner.fit(X, treatment, y)
        return self.learner.predict(X)

    def test_polars_matches_numpy(self, synthetic_data_numpy, synthetic_data_polars):
        te_np = self._fit_predict(*synthetic_data_numpy)
        self.learner = BaseTClassifier(learner=LogisticRegression())
        te_pl = self._fit_predict(*synthetic_data_polars)
        _assert_te_close(te_np, te_pl)

    def test_lazyframe_input(self, synthetic_data_numpy, synthetic_data_polars_lazy):
        te_np = self._fit_predict(*synthetic_data_numpy)
        self.learner = BaseTClassifier(learner=LogisticRegression())
        te_pl = self._fit_predict(*synthetic_data_polars_lazy)
        _assert_te_close(te_np, te_pl)

    def test_fit_predict_returns_numpy(self, synthetic_data_polars):
        te = self._fit_predict(*synthetic_data_polars)
        assert isinstance(te, np.ndarray)


# S-Learner Classifier


class TestSClassifierPolars:
    @pytest.fixture(autouse=True)
    def _learner(self):
        self.learner = BaseSClassifier(learner=LogisticRegression())

    def _fit_predict(self, X, treatment, y):
        self.learner.fit(X, treatment, y)
        return self.learner.predict(X)

    def test_polars_matches_numpy(self, synthetic_data_numpy, synthetic_data_polars):
        te_np = self._fit_predict(*synthetic_data_numpy)
        self.learner = BaseSClassifier(learner=LogisticRegression())
        te_pl = self._fit_predict(*synthetic_data_polars)
        _assert_te_close(te_np, te_pl)

    def test_lazyframe_input(self, synthetic_data_numpy, synthetic_data_polars_lazy):
        te_np = self._fit_predict(*synthetic_data_numpy)
        self.learner = BaseSClassifier(learner=LogisticRegression())
        te_pl = self._fit_predict(*synthetic_data_polars_lazy)
        _assert_te_close(te_np, te_pl)

    def test_fit_predict_returns_numpy(self, synthetic_data_polars):
        te = self._fit_predict(*synthetic_data_polars)
        assert isinstance(te, np.ndarray)


# X-Learner Classifier


class TestXClassifierPolars:
    @pytest.fixture(autouse=True)
    def _learner(self):
        self.learner = BaseXClassifier(
            outcome_learner=LogisticRegression(),
            effect_learner=LinearRegression(),
        )

    def _fit_predict(self, X, treatment, y):
        self.learner.fit(X, treatment, y)
        return self.learner.predict(X)

    def test_polars_matches_numpy(self, synthetic_data_numpy, synthetic_data_polars):
        te_np = self._fit_predict(*synthetic_data_numpy)
        self.learner = BaseXClassifier(
            outcome_learner=LogisticRegression(),
            effect_learner=LinearRegression(),
        )
        te_pl = self._fit_predict(*synthetic_data_polars)
        _assert_te_close(te_np, te_pl)

    def test_lazyframe_input(self, synthetic_data_numpy, synthetic_data_polars_lazy):
        te_np = self._fit_predict(*synthetic_data_numpy)
        self.learner = BaseXClassifier(
            outcome_learner=LogisticRegression(),
            effect_learner=LinearRegression(),
        )
        te_pl = self._fit_predict(*synthetic_data_polars_lazy)
        _assert_te_close(te_np, te_pl)

    def test_fit_predict_returns_numpy(self, synthetic_data_polars):
        te = self._fit_predict(*synthetic_data_polars)
        assert isinstance(te, np.ndarray)
