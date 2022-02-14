from abc import ABCMeta, abstractclassmethod
import logging
import numpy as np
import pandas as pd

from causalml.inference.meta.explainer import Explainer
from causalml.inference.meta.utils import check_p_conditions, convert_pd_to_np
from causalml.propensity import compute_propensity_score


logger = logging.getLogger("causalml")


class BaseLearner(metaclass=ABCMeta):
    @abstractclassmethod
    def fit(self, X, treatment, y, p=None):
        pass

    @abstractclassmethod
    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        pass

    def fit_predict(
        self,
        X,
        treatment,
        y,
        p=None,
        return_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
        return_components=False,
        verbose=True,
    ):
        self.fit(X, treatment, y, p)
        return self.predict(X, treatment, y, p, return_components, verbose)

    @abstractclassmethod
    def estimate_ate(
        self,
        X,
        treatment,
        y,
        p=None,
        bootstrap_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
    ):
        pass

    def bootstrap(self, X, treatment, y, p=None, size=10000):
        """Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population."""
        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]

        if p is not None:
            p_b = {group: _p[idxs] for group, _p in p.items()}
        else:
            p_b = None

        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b, p=p_b)
        return self.predict(X=X, p=p)

    @staticmethod
    def _format_p(p, t_groups):
        """Format propensity scores into a dictionary of {treatment group: propensity scores}.

        Args:
            p (np.ndarray, pd.Series, or dict): propensity scores
            t_groups (list): treatment group names.

        Returns:
            dict of {treatment group: propensity scores}
        """
        check_p_conditions(p, t_groups)

        if isinstance(p, (np.ndarray, pd.Series)):
            treatment_name = t_groups[0]
            p = {treatment_name: convert_pd_to_np(p)}
        elif isinstance(p, dict):
            p = {
                treatment_name: convert_pd_to_np(_p) for treatment_name, _p in p.items()
            }

        return p

    def _set_propensity_models(self, X, treatment, y):
        """Set self.propensity and self.propensity_models.

        It trains propensity models for all treatment groups, save them in self.propensity_models, and
        save propensity scores in self.propensity in dictionaries with treatment groups as keys.

        It will use self.model_p if available to train propensity models. Otherwise, it will use a default
        PropensityModel (i.e. ElasticNetPropensityModel).

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        """
        logger.info("Generating propensity score")
        p = dict()
        p_model = dict()
        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            w_filt = (treatment_filt == group).astype(int)
            w = (treatment == group).astype(int)
            propensity_model = self.model_p if hasattr(self, "model_p") else None
            p[group], p_model[group] = compute_propensity_score(
                X=X_filt,
                treatment=w_filt,
                p_model=propensity_model,
                X_pred=X,
                treatment_pred=w,
            )
        self.propensity_model = p_model
        self.propensity = p

    def get_importance(
        self,
        X=None,
        tau=None,
        model_tau_feature=None,
        features=None,
        method="auto",
        normalize=True,
        test_size=0.3,
        random_state=None,
    ):
        """
        Builds a model (using X to predict estimated/actual tau), and then calculates feature importances
        based on a specified method.

        Currently supported methods are:
            - auto (calculates importance based on estimator's default implementation of feature importance;
                    estimator must be tree-based)
                    Note: if none provided, it uses lightgbm's LGBMRegressor as estimator, and "gain" as
                    importance type
            - permutation (calculates importance based on mean decrease in accuracy when a feature column is permuted;
                           estimator can be any form)
        Hint: for permutation, downsample data for better performance especially if X.shape[1] is large

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (np.array): list/array of feature names. If None, an enumerated list will be used
            method (str): auto, permutation
            normalize (bool): normalize by sum of importances if method=auto (defaults to True)
            test_size (float/int): if float, represents the proportion of the dataset to include in the test split.
                                   If int, represents the absolute number of test samples (used for estimating
                                   permutation importance)
            random_state (int/RandomState instance/None): random state used in permutation importance estimation
        """
        explainer = Explainer(
            method=method,
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            classes=self._classes,
            normalize=normalize,
            test_size=test_size,
            random_state=random_state,
        )
        return explainer.get_importance()

    def get_shap_values(self, X=None, model_tau_feature=None, tau=None, features=None):
        """
        Builds a model (using X to predict estimated/actual tau), and then calculates shapley values.
        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used.
        """
        explainer = Explainer(
            method="shapley",
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            classes=self._classes,
        )
        return explainer.get_shap_values()

    def plot_importance(
        self,
        X=None,
        tau=None,
        model_tau_feature=None,
        features=None,
        method="auto",
        normalize=True,
        test_size=0.3,
        random_state=None,
    ):
        """
        Builds a model (using X to predict estimated/actual tau), and then plots feature importances
        based on a specified method.

        Currently supported methods are:
            - auto (calculates importance based on estimator's default implementation of feature importance;
                    estimator must be tree-based)
                    Note: if none provided, it uses lightgbm's LGBMRegressor as estimator, and "gain" as
                    importance type
            - permutation (calculates importance based on mean decrease in accuracy when a feature column is permuted;
                           estimator can be any form)
        Hint: for permutation, downsample data for better performance especially if X.shape[1] is large

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used
            method (str): auto, permutation
            normalize (bool): normalize by sum of importances if method=auto (defaults to True)
            test_size (float/int): if float, represents the proportion of the dataset to include in the test split.
                                   If int, represents the absolute number of test samples (used for estimating
                                   permutation importance)
            random_state (int/RandomState instance/None): random state used in permutation importance estimation
        """
        explainer = Explainer(
            method=method,
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            classes=self._classes,
            normalize=normalize,
            test_size=test_size,
            random_state=random_state,
        )
        explainer.plot_importance()

    def plot_shap_values(
        self,
        X=None,
        tau=None,
        model_tau_feature=None,
        features=None,
        shap_dict=None,
        **kwargs
    ):
        """
        Plots distribution of shapley values.

        If shapley values have been pre-computed, pass it through the shap_dict parameter.
        If shap_dict is not provided, this builds a new model (using X to predict estimated/actual tau),
        and then calculates shapley values.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix. Required if shap_dict is None.
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used.
            shap_dict (optional, dict): a dict of shapley value matrices. If None, shap_dict will be computed.
        """
        override_checks = False if shap_dict is None else True
        explainer = Explainer(
            method="shapley",
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            override_checks=override_checks,
            classes=self._classes,
        )
        explainer.plot_shap_values(shap_dict=shap_dict)

    def plot_shap_dependence(
        self,
        treatment_group,
        feature_idx,
        X,
        tau,
        model_tau_feature=None,
        features=None,
        shap_dict=None,
        interaction_idx="auto",
        **kwargs
    ):
        """
        Plots dependency of shapley values for a specified feature, colored by an interaction feature.

        If shapley values have been pre-computed, pass it through the shap_dict parameter.
        If shap_dict is not provided, this builds a new model (using X to predict estimated/actual tau),
        and then calculates shapley values.

        This plots the value of the feature on the x-axis and the SHAP value of the same feature
        on the y-axis. This shows how the model depends on the given feature, and is like a
        richer extension of the classical partial dependence plots. Vertical dispersion of the
        data points represents interaction effects.

        Args:
            treatment_group (str or int): name of treatment group to create dependency plot on
            feature_idx (str or int): feature index / name to create dependency plot on
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used.
            shap_dict (optional, dict): a dict of shapley value matrices. If None, shap_dict will be computed.
            interaction_idx (optional, str or int): feature index / name used in coloring scheme as interaction feature.
                If "auto" then shap.common.approximate_interactions is used to pick what seems to be the
                strongest interaction (note that to find to true strongest interaction you need to compute
                the SHAP interaction values).
        """
        override_checks = False if shap_dict is None else True
        explainer = Explainer(
            method="shapley",
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            override_checks=override_checks,
            classes=self._classes,
        )
        explainer.plot_shap_dependence(
            treatment_group=treatment_group,
            feature_idx=feature_idx,
            shap_dict=shap_dict,
            interaction_idx=interaction_idx,
            **kwargs
        )
