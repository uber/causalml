import multiprocessing as mp
from abc import abstractmethod

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from causalml.inference.tree import CausalTreeRegressor, CausalRandomForestRegressor
from causalml.metrics import ape
from causalml.metrics import qini_score
from .const import RANDOM_SEED, ERROR_THRESHOLD


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

    def prepare_data(self, generate_regression_data) -> tuple:
        y, X, treatment, tau, b, e = generate_regression_data(mode=2)
        df = pd.DataFrame(X)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df.columns = feature_names
        df["outcome"] = y
        df["treatment"] = treatment
        df["treatment_effect"] = tau
        self.df_train, self.df_test = train_test_split(
            df, test_size=self.test_size, random_state=RANDOM_SEED
        )
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


class TestCausalTreeRegressor(CausalTreeBase):
    def prepare_model(self) -> CausalTreeRegressor:
        ctree = CausalTreeRegressor(
            control_name=self.control_name, groups_cnt=True, random_state=RANDOM_SEED
        )
        return ctree

    def test_fit(self, generate_regression_data):
        ctree = self.prepare_model()
        (
            X_train,
            X_test,
            y_train,
            y_test,
            treatment_train,
            treatment_test,
        ) = self.prepare_data(generate_regression_data)
        ctree.fit(X=X_train, treatment=treatment_train, y=y_train)
        df_result = pd.DataFrame(
            {
                "ctree_ite_pred": ctree.predict(X_test),
                "outcome": y_test,
                "is_treated": treatment_test,
                "treatment_effect": self.df_test["treatment_effect"],
            }
        )
        df_qini = qini_score(
            df_result,
            outcome_col="outcome",
            treatment_col="is_treated",
            treatment_effect_col="treatment_effect",
        )
        assert df_qini["ctree_ite_pred"] > 0.0

    @pytest.mark.parametrize("return_ci", (False, True))
    @pytest.mark.parametrize("bootstrap_size", (500, 800))
    @pytest.mark.parametrize("n_bootstraps", (1000,))
    def test_fit_predict(
        self, generate_regression_data, return_ci, bootstrap_size, n_bootstraps
    ):
        y, X, treatment, tau, b, e = generate_regression_data(mode=1)
        ctree = self.prepare_model()
        output = ctree.fit_predict(
            X=X,
            treatment=treatment,
            y=y,
            return_ci=return_ci,
            n_bootstraps=n_bootstraps,
            bootstrap_size=bootstrap_size,
            n_jobs=mp.cpu_count() - 1,
            verbose=False,
        )
        if return_ci:
            te, te_lower, te_upper = output
            assert len(output) == 3
            assert (te_lower <= te).all() and (te_upper >= te).all()
        else:
            te = output
            assert te.shape[0] == y.shape[0]

    def test_predict(self, generate_regression_data):
        y, X, treatment, tau, b, e = generate_regression_data(mode=2)
        ctree = self.prepare_model()
        ctree.fit(X=X, treatment=treatment, y=y)
        y_pred = ctree.predict(X[:1, :])
        y_pred_with_outcomes = ctree.predict(X[:1, :], with_outcomes=True)
        assert y_pred.shape == (1,)
        assert y_pred_with_outcomes.shape == (1, 3)

    def test_ate(self, generate_regression_data):
        y, X, treatment, tau, b, e = generate_regression_data(mode=2)
        ctree = self.prepare_model()
        ate, ate_lower, ate_upper = ctree.estimate_ate(X=X, y=y, treatment=treatment)
        assert (ate >= ate_lower) and (ate <= ate_upper)
        assert ape(tau.mean(), ate) < ERROR_THRESHOLD


class TestCausalRandomForestRegressor(CausalTreeBase):
    def prepare_model(self, n_estimators: int) -> CausalRandomForestRegressor:
        crforest = CausalRandomForestRegressor(
            criterion="causal_mse",
            control_name=self.control_name,
            n_estimators=n_estimators,
            n_jobs=mp.cpu_count() - 1,
        )
        return crforest

    @pytest.mark.parametrize("n_estimators", (5, 10, 50))
    def test_fit(self, generate_regression_data, n_estimators):
        crforest = self.prepare_model(n_estimators=n_estimators)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            treatment_train,
            treatment_test,
        ) = self.prepare_data(generate_regression_data)
        crforest.fit(X=X_train, treatment=treatment_train, y=y_train)

        df_result = pd.DataFrame(
            {
                "crforest_ite_pred": crforest.predict(X_test),
                "is_treated": treatment_test,
                "treatment_effect": self.df_test["treatment_effect"],
            }
        )
        df_qini = qini_score(
            df_result,
            outcome_col="outcome",
            treatment_col="is_treated",
            treatment_effect_col="treatment_effect",
        )
        assert df_qini["crforest_ite_pred"] > 0.0

    @pytest.mark.parametrize("n_estimators", (5,))
    def test_predict(self, generate_regression_data, n_estimators):
        y, X, treatment, tau, b, e = generate_regression_data(mode=2)
        ctree = self.prepare_model(n_estimators=n_estimators)
        ctree.fit(X=X, y=y, treatment=treatment)
        y_pred = ctree.predict(X[:1, :])
        y_pred_with_outcomes = ctree.predict(X[:1, :], with_outcomes=True)
        assert y_pred.shape == (1,)
        assert y_pred_with_outcomes.shape == (1, 3)

    @pytest.mark.parametrize("n_estimators", (5,))
    def test_unbiased_sampling_error(self, generate_regression_data, n_estimators):
        crforest = self.prepare_model(n_estimators=n_estimators)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            treatment_train,
            treatment_test,
        ) = self.prepare_data(generate_regression_data)
        crforest.fit(X=X_train, treatment=treatment_train, y=y_train)
        crforest_test_var = crforest.calculate_error(X_train=X_train, X_test=X_test)
        assert (crforest_test_var > 0).all()
        assert crforest_test_var.shape[0] == y_test.shape[0]
