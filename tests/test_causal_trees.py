import multiprocessing as mp
from abc import abstractmethod

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from causalml.inference.tree import CausalTreeRegressor, CausalRandomForestRegressor
from causalml.metrics import ape
from causalml.metrics import qini_score
from .const import RANDOM_SEED, ERROR_THRESHOLD, N_SAMPLE


class CausalTreeBase:
    test_size: float = 0.2
    control_name: int = 0

    @abstractmethod
    def prepare_model(self, *args, **kwargs):
        return

    @abstractmethod
    def test_fit(self, *args, **kwargs):
        return

    @abstractmethod
    def test_predict(self, *args, **kwargs):
        return

    def prepare_data(self, generate_regression_data, n_treatments: int) -> pd.DataFrame:
        data = []
        sigmas = np.abs(np.random.normal(size=n_treatments))
        for i in range(n_treatments):
            _, X, w, tau, b, e = generate_regression_data(mode=2, sigma=sigmas[i])
            w = np.where(w == 1, i + 1, 0)
            y = b + (w - 0.5) * tau + sigmas[i] * np.random.normal(size=N_SAMPLE)
            data.append([y, X, w, tau, b, e])

        y = np.hstack([chunk[0] for chunk in data])
        X = np.vstack([chunk[1] for chunk in data])
        w = np.hstack([chunk[2] for chunk in data])
        tau = np.hstack([chunk[3] for chunk in data])

        df = pd.DataFrame(X)
        df.columns = [f"feature_{i}" for i in range(X.shape[1])]
        df["outcome"] = y
        df["treatment"] = w
        df["treatment_effect"] = tau
        df = df.sample(frac=1.0).reset_index(drop=True)

        df_balanced = (
            pd.concat(
                [
                    df[df["treatment"] != 0],
                    df[df["treatment"] == 0].sample(frac=1 / n_treatments),
                ]
            )
            .sample(frac=1.0)
            .reset_index(drop=True)
        )
        return df_balanced

    def prepare_multi_treatment_data(self, generate_regression_data, n_treatments: int):
        return self.prepare_data(generate_regression_data, n_treatments=n_treatments)

    def split_data(self, df: pd.DataFrame) -> tuple:
        self.df_train, self.df_test = train_test_split(
            df, test_size=self.test_size, random_state=RANDOM_SEED
        )
        feature_names = [x for x in self.df_train.columns if x.startswith("feature_")]
        X_train, X_test = (
            self.df_train[feature_names].values,
            self.df_test[feature_names].values,
        )
        y_train, y_test = (
            self.df_train["outcome"].values,
            self.df_test["outcome"].values,
        )
        treatment_train, treatment_test = (
            self.df_train["treatment"].values,
            self.df_test["treatment"].values,
        )
        return X_train, X_test, y_train, y_test, treatment_train, treatment_test


@pytest.mark.parametrize(
    "n_treatments",
    (
        1,
        2,
    ),
)
class TestCausalTreeCase(CausalTreeBase):

    def prepare_model(self) -> CausalTreeRegressor:
        ctree = CausalTreeRegressor(
            control_name=self.control_name, groups_cnt=True, random_state=RANDOM_SEED
        )
        return ctree

    def test_fit(self, generate_regression_data, n_treatments: int):
        ctree = self.prepare_model()
        data = self.prepare_multi_treatment_data(generate_regression_data, n_treatments)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            treatment_train,
            treatment_test,
        ) = self.split_data(data)
        ctree.fit(X=X_train, treatment=treatment_train, y=y_train)
        preds = ctree.predict(X=X_test)

        df_result = pd.DataFrame(
            {
                "outcome": y_test,
                "group": treatment_test,
                "treatment_effect": self.df_test["treatment_effect"],
            }
        )
        for i, group in enumerate(range(1, n_treatments + 1)):
            df_result[f"ite_pred_t{group}"] = preds[:, i] if n_treatments > 1 else preds
            df_group_result = df_result[df_result["group"].isin([0, group])].copy()
            df_group_result["is_treated"] = (df_group_result["group"] == group).astype(
                int
            )
            df_group_result = df_group_result[
                ["outcome", "is_treated", "treatment_effect", f"ite_pred_t{group}"]
            ]
            df_qini = qini_score(
                df_group_result,
                outcome_col="outcome",
                treatment_col="is_treated",
                treatment_effect_col="treatment_effect",
            )
            assert df_qini[f"ite_pred_t{group}"] > 0.0

    def test_predict(self, generate_regression_data, n_treatments: int):
        ctree = self.prepare_model()
        data = self.prepare_multi_treatment_data(generate_regression_data, n_treatments)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            treatment_train,
            treatment_test,
        ) = self.split_data(data)
        ctree.fit(X=X_train, treatment=treatment_train, y=y_train)
        y_pred = ctree.predict(X_test)
        y_pred = y_pred.reshape(-1, n_treatments) if n_treatments == 1 else y_pred
        y_pred_with_outcomes = ctree.predict(X_test, with_outcomes=True)
        assert y_pred.shape == (X_test.shape[0], n_treatments)
        assert y_pred_with_outcomes.shape == (
            X_test.shape[0],
            n_treatments + (n_treatments + 1),
        )

    def test_ate(self, generate_regression_data, n_treatments: int):
        ctree = self.prepare_model()
        data = self.prepare_multi_treatment_data(generate_regression_data, n_treatments)
        feature_names = [x for x in data.columns if x.startswith("feature_")]
        X, y, treatment = data[feature_names], data["outcome"], data["treatment"]
        tau = data["treatment_effect"]
        ate, ate_lower, ate_upper = ctree.estimate_ate(
            X=X.values, treatment=treatment.values, y=y.values
        )
        assert (ate >= ate_lower) and (ate <= ate_upper)
        assert ape(tau.mean(), ate) < ERROR_THRESHOLD


@pytest.mark.parametrize(
    "n_treatments",
    (
        1,
        2,
    ),
)
@pytest.mark.parametrize(
    "n_estimators",
    (
        5,
        10,
    ),
)
class TestCausalRandomForestCase(CausalTreeBase):
    def prepare_model(self, n_estimators: int) -> CausalRandomForestRegressor:
        crforest = CausalRandomForestRegressor(
            criterion="causal_mse",
            control_name=self.control_name,
            n_estimators=n_estimators,
            n_jobs=mp.cpu_count() - 1,
        )
        return crforest

    def test_fit(self, generate_regression_data, n_estimators: int, n_treatments: int):
        crforest = self.prepare_model(n_estimators=n_estimators)
        data = self.prepare_multi_treatment_data(generate_regression_data, n_treatments)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            treatment_train,
            treatment_test,
        ) = self.split_data(data)
        crforest.fit(X=X_train, treatment=treatment_train, y=y_train)
        preds = crforest.predict(X=X_test)

        df_result = pd.DataFrame(
            {
                "outcome": y_test,
                "group": treatment_test,
                "treatment_effect": self.df_test["treatment_effect"],
            }
        )
        for i, group in enumerate(range(1, n_treatments + 1)):
            df_result[f"ite_pred_t{group}"] = preds[:, i] if n_treatments > 1 else preds
            df_group_result = df_result[df_result["group"].isin([0, group])].copy()
            df_group_result["is_treated"] = (df_group_result["group"] == group).astype(
                int
            )
            df_group_result = df_group_result[
                ["outcome", "is_treated", "treatment_effect", f"ite_pred_t{group}"]
            ]
            df_qini = qini_score(
                df_group_result,
                outcome_col="outcome",
                treatment_col="is_treated",
                treatment_effect_col="treatment_effect",
            )
            assert df_qini[f"ite_pred_t{group}"] > 0.0

    def test_predict(
        self, generate_regression_data, n_estimators: int, n_treatments: int
    ):
        crforest = self.prepare_model(n_estimators=n_estimators)
        data = self.prepare_multi_treatment_data(generate_regression_data, n_treatments)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            treatment_train,
            treatment_test,
        ) = self.split_data(data)
        crforest.fit(X=X_train, treatment=treatment_train, y=y_train)
        y_pred = crforest.predict(X_test)
        y_pred = y_pred.reshape(-1, n_treatments) if n_treatments == 1 else y_pred
        y_pred_with_outcomes = crforest.predict(X_test, with_outcomes=True)
        assert y_pred.shape == (X_test.shape[0], n_treatments)
        assert y_pred_with_outcomes.shape == (
            X_test.shape[0],
            n_treatments + (n_treatments + 1),
        )

    def test_unbiased_sampling_error(
        self, generate_regression_data, n_estimators: int, n_treatments: int
    ):
        crforest = self.prepare_model(n_estimators=n_estimators)
        data = self.prepare_multi_treatment_data(generate_regression_data, n_treatments)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            treatment_train,
            treatment_test,
        ) = self.split_data(data)
        crforest.fit(X=X_train, treatment=treatment_train, y=y_train)
        if n_treatments == 1:
            crforest_test_var = crforest.calculate_error(X_train=X_train, X_test=X_test)
            assert (crforest_test_var > 0).all()
            assert crforest_test_var.shape[0] == y_test.shape[0]


def test_CausalRandomForestRegressor_no_inf_predictions():
    """Test that CausalRandomForestRegressor does not predict inf values
    when some tree splits have zero-count treatment/control groups (#589)."""
    np.random.seed(RANDOM_SEED)
    n = 100
    X = np.random.randn(n, 5)
    # Heavily imbalanced: very few treated samples so tree splits
    # can produce nodes with zero treatment count
    treatment = np.array([0] * 90 + [1] * 10)
    y = np.random.randn(n)

    model = CausalRandomForestRegressor(
        criterion="causal_mse",
        control_name=0,
        n_estimators=10,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
    )
    model.fit(X=X, treatment=treatment, y=y)
    preds = model.predict(X=X)

    assert np.all(np.isfinite(preds)), "Predictions contain inf or NaN values"


def test_CausalRandomForestRegressor_no_inf_predictions_ttest():
    """Test that CausalRandomForestRegressor with criterion='ttest' does not
    predict inf values when some tree splits have zero-count
    treatment/control groups (#589)."""
    np.random.seed(RANDOM_SEED)
    n = 100
    X = np.random.randn(n, 5)
    treatment = np.array([0] * 90 + [1] * 10)
    y = np.random.randn(n)

    model = CausalRandomForestRegressor(
        criterion="t_test",
        control_name=0,
        n_estimators=10,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
    )
    model.fit(X=X, treatment=treatment, y=y)
    preds = model.predict(X=X)

    assert np.all(np.isfinite(preds)), "Predictions contain inf or NaN values"


# ---------------------------------------------------------------------------
# Native missing-value (NaN) support (prerequisite for issue #955)
#
# The shared causal criterion implements scikit-learn's per-feature
# missing-value accumulation, threaded through the dense best splitter, so the
# causal tree and forest accept NaNs in X natively (inf is still rejected).
# ---------------------------------------------------------------------------


def _make_missing_effect_data(n=4000, seed=RANDOM_SEED):
    """Regression uplift data whose treatment effect is carried by missingness.

    The informative feature is NaN for half the rows and a varied value
    otherwise; the treatment effect is 2.0 exactly on the missing rows and 0.0
    on the observed rows. A missing-aware tree must route NaN accordingly.
    """
    rng = np.random.RandomState(seed)
    treatment = rng.randint(0, 2, n)
    missing = rng.rand(n) < 0.5
    informative = np.where(missing, np.nan, rng.normal(size=n))
    noise = rng.normal(size=n)
    X = np.column_stack([informative, noise]).astype(np.float64)
    tau = np.where(missing, 2.0, 0.0)
    y = 0.3 * noise + treatment * tau + rng.normal(scale=0.1, size=n)
    return X, treatment, y


def test_causal_tree_reports_missing_value_support():
    """The causal tree advertises native NaN support for the dense best splitter."""
    from scipy.sparse import csr_matrix

    est = CausalTreeRegressor(criterion="causal_mse", control_name=0)
    X = np.random.RandomState(RANDOM_SEED).randn(100, 4)
    assert est.__sklearn_tags__().input_tags.allow_nan is True
    assert est._support_missing_values(X) is True
    assert est._support_missing_values(csr_matrix(X)) is False


@pytest.mark.parametrize("criterion", ["causal_mse", "standard_mse", "t_test"])
def test_causal_tree_fit_predict_with_nan(criterion):
    """The causal tree fits and predicts finite effects with NaNs in X."""
    rng = np.random.RandomState(RANDOM_SEED)
    n = 2000
    X = rng.randn(n, 5)
    treatment = rng.randint(0, 2, n)
    y = X[:, 1] * 0.3 + treatment * (X[:, 0] > 0) * 0.5 + rng.normal(scale=0.1, size=n)
    Xn = X.copy()
    for c in (0, 2, 4):
        idx = rng.choice(n, size=int(0.15 * n), replace=False)
        Xn[idx, c] = np.nan

    model = CausalTreeRegressor(
        criterion=criterion,
        control_name=0,
        max_depth=4,
        min_samples_leaf=50,
        random_state=RANDOM_SEED,
    )
    model.fit(X=Xn, treatment=treatment, y=y)
    te = np.ravel(model.predict(Xn))
    assert te.shape == (n,)
    assert np.isfinite(te).all()


def test_causal_tree_learns_missing_routing():
    """The tree routes NaN to isolate a treatment effect carried by missingness."""
    X, treatment, y = _make_missing_effect_data(seed=RANDOM_SEED)
    model = CausalTreeRegressor(
        criterion="causal_mse",
        control_name=0,
        max_depth=3,
        min_samples_leaf=100,
        random_state=RANDOM_SEED,
    )
    model.fit(X=X, treatment=treatment, y=y)
    te_missing = float(np.ravel(model.predict(np.array([[np.nan, 0.0]])))[0])
    te_observed = float(np.ravel(model.predict(np.array([[0.0, 0.0]])))[0])
    assert te_missing > 1.0
    assert te_missing - te_observed > 0.8


def test_causal_forest_fit_predict_with_nan():
    """The causal forest fits and predicts finite effects with NaNs and reports the tag."""
    X, treatment, y = _make_missing_effect_data(n=3000, seed=RANDOM_SEED)
    model = CausalRandomForestRegressor(
        criterion="causal_mse",
        control_name=0,
        n_estimators=10,
        max_depth=4,
        min_samples_leaf=50,
        random_state=RANDOM_SEED,
    )
    assert model.__sklearn_tags__().input_tags.allow_nan is True
    model.fit(X=X, treatment=treatment, y=y)
    te = np.ravel(model.predict(X))
    assert np.isfinite(te).all()
    # The effect is concentrated on the missing rows, so their mean estimated
    # effect must clearly exceed the observed rows'.
    missing_rows = np.isnan(X[:, 0])
    assert np.mean(te[missing_rows]) - np.mean(te[~missing_rows]) > 0.5


def test_causal_tree_rejects_inf():
    """Non-NaN, non-finite values (inf) are still rejected by the causal tree."""
    rng = np.random.RandomState(RANDOM_SEED)
    X = rng.randn(200, 4)
    treatment = rng.randint(0, 2, 200)
    y = rng.randn(200)
    X[0, 0] = np.inf
    model = CausalTreeRegressor(criterion="causal_mse", control_name=0)
    with pytest.raises(ValueError, match="infinity"):
        model.fit(X=X, treatment=treatment, y=y)
