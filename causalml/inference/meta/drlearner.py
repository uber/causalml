import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from causalml.inference.meta.explainer import Explainer
from causalml.inference.meta.utils import (
    check_treatment_vector,
    check_p_conditions,
    convert_pd_to_np,
)
from causalml.metrics import regression_metrics, classification_metrics
from causalml.propensity import compute_propensity_score
from scipy.stats import norm
from sklearn.model_selection import cross_val_predict, KFold
from tqdm import tqdm

logger = logging.getLogger("causalml")


class BaseDRLearner(object):
    """A parent class for DR-learner regressor classes.

    An DR-learner estimates treatment effects with four machine learning models.

    Details of DR-learner are available at Kennedy (2020) (https://arxiv.org/abs/2004.14497).
    """

    def __init__(
        self,
        learner=None,
        control_outcome_learner=None,
        treatment_outcome_learner=None,
        treatment_effect_learner=None,
        ate_alpha=0.05,
        control_name=0,
    ):
        """Initialize a X-learner.

        Args:
            learner (optional): a model to estimate outcomes and treatment effects in both the control and treatment
                groups
            control_outcome_learner (optional): a model to estimate outcomes in the control group
            treatment_outcome_learner (optional): a model to estimate outcomes in the treatment group
            treatment_effect_learner (optional): a model to estimate treatment effects in the treatment group
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        assert (learner is not None) or (
            (control_outcome_learner is not None)
            and (treatment_outcome_learner is not None)
            and (treatment_effect_learner is not None)
        )

        if control_outcome_learner is None:
            self.model_mu_c = deepcopy(learner)
        else:
            self.model_mu_c = control_outcome_learner

        if treatment_outcome_learner is None:
            self.model_mu_t = deepcopy(learner)
        else:
            self.model_mu_t = treatment_outcome_learner

        if treatment_effect_learner is None:
            self.model_tau = deepcopy(learner)
        else:
            self.model_tau = treatment_effect_learner

        self.ate_alpha = ate_alpha
        self.control_name = control_name

        self.propensity = None

    def __repr__(self):
        return (
            "{}(control_outcome_learner={},\n"
            "\ttreatment_outcome_learner={},\n"
            "\ttreatment_effect_learner={})".format(
                self.__class__.__name__,
                self.model_mu_c.__repr__(),
                self.model_mu_t.__repr__(),
                self.model_tau.__repr__(),
            )
        )

    def fit(self, X, treatment, y, p=None, seed=None):
        """Fit the inference model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            seed (int): random seed for cross-fitting
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()
        self._classes = {group: i for i, group in enumerate(self.t_groups)}

        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        split_indices = [index for _, index in cv.split(y)]

        self.models_mu_c = [
            deepcopy(self.model_mu_c),
            deepcopy(self.model_mu_c),
            deepcopy(self.model_mu_c),
        ]
        self.models_mu_t = {
            group: [
                deepcopy(self.model_mu_t),
                deepcopy(self.model_mu_t),
                deepcopy(self.model_mu_t),
            ]
            for group in self.t_groups
        }
        self.models_tau = {
            group: [
                deepcopy(self.model_tau),
                deepcopy(self.model_tau),
                deepcopy(self.model_tau),
            ]
            for group in self.t_groups
        }
        if p is None:
            self.propensity = {group: np.zeros(y.shape[0]) for group in self.t_groups}

        for ifold in range(3):
            treatment_idx = split_indices[ifold]
            outcome_idx = split_indices[(ifold + 1) % 3]
            tau_idx = split_indices[(ifold + 2) % 3]

            treatment_treat, treatment_out, treatment_tau = (
                treatment[treatment_idx],
                treatment[outcome_idx],
                treatment[tau_idx],
            )
            y_out, y_tau = y[outcome_idx], y[tau_idx]
            X_treat, X_out, X_tau = X[treatment_idx], X[outcome_idx], X[tau_idx]

            if p is None:
                logger.info("Generating propensity score")
                cur_p = dict()
                cur_p_model = dict()

                for group in self.t_groups:
                    mask = (treatment_treat == group) | (
                        treatment_treat == self.control_name
                    )
                    treatment_filt = treatment_treat[mask]
                    X_filt = X_treat[mask]
                    w_filt = (treatment_filt == group).astype(int)
                    w = (treatment_tau == group).astype(int)
                    cur_p[group], _ = compute_propensity_score(
                        X=X_filt, treatment=w_filt, X_pred=X_tau, treatment_pred=w
                    )
                    self.propensity[group][tau_idx] = cur_p[group]
            else:
                cur_p = dict()
                if isinstance(p, (np.ndarray, pd.Series)):
                    cur_p = {self.t_groups[0]: convert_pd_to_np(p[tau_idx])}
                else:
                    cur_p = {g: prop[tau_idx] for g, prop in p.items()}
                check_p_conditions(cur_p, self.t_groups)

            logger.info("Generate outcome regressions")
            self.models_mu_c[ifold].fit(
                X_out[treatment_out == self.control_name],
                y_out[treatment_out == self.control_name],
            )
            for group in self.t_groups:
                self.models_mu_t[group][ifold].fit(
                    X_out[treatment_out == group], y_out[treatment_out == group]
                )

            logger.info("Fit pseudo outcomes from the DR formula")

            for group in self.t_groups:
                mask = (treatment_tau == group) | (treatment_tau == self.control_name)
                treatment_filt = treatment_tau[mask]
                X_filt = X_tau[mask]
                y_filt = y_tau[mask]
                w_filt = (treatment_filt == group).astype(int)
                p_filt = cur_p[group][mask]
                mu_t = self.models_mu_t[group][ifold].predict(X_filt)
                mu_c = self.models_mu_c[ifold].predict(X_filt)
                dr = (
                    (w_filt - p_filt)
                    / p_filt
                    / (1 - p_filt)
                    * (y_filt - mu_t * w_filt - mu_c * (1 - w_filt))
                    + mu_t
                    - mu_c
                )
                self.models_tau[group][ifold].fit(X_filt, dr)

    def predict(self, X, treatment=None, y=None, return_components=False, verbose=True):
        """Predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series, optional): a treatment vector
            y (np.array or pd.Series, optional): an outcome vector
            verbose (bool, optional): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        yhat_cs = {}
        yhat_ts = {}

        for i, group in enumerate(self.t_groups):
            models_tau = self.models_tau[group]
            _te = np.r_[[model.predict(X) for model in models_tau]].mean(axis=0)
            te[:, i] = np.ravel(_te)
            yhat_cs[group] = np.r_[[model.predict(X) for model in self.models_mu_c]].mean(axis=0)
            yhat_ts[group] = np.r_[[model.predict(X) for model in self.models_mu_t[group]]].mean(axis=0)

            if (y is not None) and (treatment is not None) and verbose:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                X_filt = X[mask]
                y_filt = y[mask]
                w = (treatment_filt == group).astype(int)

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = yhat_cs[group][mask][w == 0]
                yhat[w == 1] = yhat_ts[group][mask][w == 1]

                logger.info("Error metrics for group {}".format(group))
                regression_metrics(y_filt, yhat, w)

        if not return_components:
            return te
        else:
            return te, yhat_cs, yhat_ts

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
        seed=None,
    ):
        """Fit the treatment effect and outcome models of the R learner and predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            return_ci (bool): whether to return confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            return_components (bool, optional): whether to return outcome for treatment and control seperately
            verbose (str): whether to output progress logs
            seed (int): random seed for cross-fitting
        Returns:
            (numpy.ndarray): Predictions of treatment effects. Output dim: [n_samples, n_treatment]
                If return_ci, returns CATE [n_samples, n_treatment], LB [n_samples, n_treatment],
                UB [n_samples, n_treatment]
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        self.fit(X, treatment, y, p, seed)

        if p is None:
            p = self.propensity

        check_p_conditions(p, self.t_groups)
        if isinstance(p, (np.ndarray, pd.Series)):
            treatment_name = self.t_groups[0]
            p = {treatment_name: convert_pd_to_np(p)}
        elif isinstance(p, dict):
            p = {
                treatment_name: convert_pd_to_np(_p) for treatment_name, _p in p.items()
            }

        te = self.predict(
            X, treatment=treatment, y=y, return_components=return_components
        )

        if not return_ci:
            return te
        else:
            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_mu_c_global = deepcopy(self.models_mu_c)
            models_mu_t_global = deepcopy(self.models_mu_t)
            models_tau_global = deepcopy(self.models_tau)
            te_bootstraps = np.zeros(
                shape=(X.shape[0], self.t_groups.shape[0], n_bootstraps)
            )

            logger.info("Bootstrap Confidence Intervals")
            for i in tqdm(range(n_bootstraps)):
                te_b = self.bootstrap(X, treatment, y, p, size=bootstrap_size, seed=seed)
                te_bootstraps[:, :, i] = te_b

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha / 2) * 100, axis=2)
            te_upper = np.percentile(
                te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=2
            )

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.models_mu_c = deepcopy(models_mu_c_global)
            self.models_mu_t = deepcopy(models_mu_t_global)
            self.models_tau = deepcopy(models_tau_global)

            return (te, te_lower, te_upper)

    def estimate_ate(
        self,
        X,
        treatment,
        y,
        p=None,
        bootstrap_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
        seed=None,
    ):
        """Estimate the Average Treatment Effect (ATE).

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            bootstrap_ci (bool): whether run bootstrap for confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            seed (int): random seed for cross-fitting
        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        te, yhat_cs, yhat_ts = self.fit_predict(X, treatment, y, p, return_components=True, seed=seed)
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        if p is None:
            p = self.propensity
        else:
            check_p_conditions(p, self.t_groups)
        if isinstance(p, (np.ndarray, pd.Series)):
            treatment_name = self.t_groups[0]
            p = {treatment_name: convert_pd_to_np(p)}
        elif isinstance(p, dict):
            p = {
                treatment_name: convert_pd_to_np(_p) for treatment_name, _p in p.items()
            }

        ate = np.zeros(self.t_groups.shape[0])
        ate_lb = np.zeros(self.t_groups.shape[0])
        ate_ub = np.zeros(self.t_groups.shape[0])

        for i, group in enumerate(self.t_groups):
            _ate = te[:, i].mean()

            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            w = (treatment_filt == group).astype(int)
            prob_treatment = float(sum(w)) / w.shape[0]

            yhat_c = yhat_cs[group][mask]
            yhat_t = yhat_ts[group][mask]

            # SE formula is based on the lower bound formula (7) from Imbens, Guido W., and Jeffrey M. Wooldridge. 2009.
            # "Recent Developments in the Econometrics of Program Evaluation." Journal of Economic Literature
            se = np.sqrt((
                (y_filt[w == 0] - yhat_c[w == 0]).var()
                / (1 - prob_treatment) +
                (y_filt[w == 1] - yhat_t[w == 1]).var()
                / prob_treatment +
                (yhat_t - yhat_c).var()
            ) / y_filt.shape[0])

            _ate_lb = _ate - se * norm.ppf(1 - self.ate_alpha / 2)
            _ate_ub = _ate + se * norm.ppf(1 - self.ate_alpha / 2)

            ate[i] = _ate
            ate_lb[i] = _ate_lb
            ate_ub[i] = _ate_ub

        if not bootstrap_ci:
            return ate, ate_lb, ate_ub
        else:
            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_mu_c_global = deepcopy(self.models_mu_c)
            models_mu_t_global = deepcopy(self.models_mu_t)
            models_tau_global = deepcopy(self.models_tau)

            logger.info("Bootstrap Confidence Intervals for ATE")
            ate_bootstraps = np.zeros(shape=(self.t_groups.shape[0], n_bootstraps))

            for n in tqdm(range(n_bootstraps)):
                cate_b = self.bootstrap(X, treatment, y, p, size=bootstrap_size, seed=seed)
                ate_bootstraps[:, n] = cate_b.mean()

            ate_lower = np.percentile(
                ate_bootstraps, (self.ate_alpha / 2) * 100, axis=1
            )
            ate_upper = np.percentile(
                ate_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=1
            )

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.models_mu_c = deepcopy(models_mu_c_global)
            self.models_mu_t = deepcopy(models_mu_t_global)
            self.models_tau = deepcopy(models_tau_global)
            return ate, ate_lower, ate_upper

    def bootstrap(self, X, treatment, y, p, size=10000, seed=None):
        """Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population."""
        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]
        p_b = {group: _p[idxs] for group, _p in p.items()}
        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b, p=p_b, seed=seed)
        te_b = self.predict(X=X)
        return te_b

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

class BaseDRRegressor(BaseDRLearner):
    """
    A parent class for X-learner regressor classes.
    """

    def __init__(self,
                 learner=None,
                 control_outcome_learner=None,
                 treatment_outcome_learner=None,
                 treatment_effect_learner=None,
                 ate_alpha=.05,
                 control_name=0):
        """Initialize an DR-learner regressor.

        Args:
            learner (optional): a model to estimate outcomes and treatment effects in both the control and treatment
                groups
            control_outcome_learner (optional): a model to estimate outcomes in the control group
            treatment_outcome_learner (optional): a model to estimate outcomes in the treatment group
            control_effect_learner (optional): a model to estimate treatment effects in the control group
            treatment_effect_learner (optional): a model to estimate treatment effects in the treatment group
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        super().__init__(
            learner=learner,
            control_outcome_learner=control_outcome_learner,
            treatment_outcome_learner=treatment_outcome_learner,
            treatment_effect_learner=treatment_effect_learner,
            ate_alpha=ate_alpha,
            control_name=control_name)

class XGBTDRRegressor(BaseDRRegressor):
    def __init__(self, ate_alpha=.05, control_name=0, *args, **kwargs):
        """Initialize a T-learner with two XGBoost models."""
        super().__init__(learner=XGBRegressor(*args, **kwargs),
                         ate_alpha=ate_alpha,
                         control_name=control_name)
