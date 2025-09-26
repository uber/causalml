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
        crforest.fit(X=X_train, treatment=treatment_train, y=y_train)
        if n_treatments == 1:
            crforest_test_var = crforest.calculate_error(X_train=X_train, X_test=X_test)
            assert (crforest_test_var > 0).all()
            assert crforest_test_var.shape[0] == y_test.shape[0]
