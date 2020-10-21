from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from xgboost import XGBRegressor

from causalml.inference.meta.utils import (check_treatment_vector, check_p_conditions,
    get_xgboost_objective_metric, convert_pd_to_np)
from causalml.inference.meta.explainer import Explainer
from causalml.propensity import compute_propensity_score

logger = logging.getLogger('causalml')


class BaseRLearner(object):
    """A parent class for R-learner classes.

    An R-learner estimates treatment effects with two machine learning models and the propensity score.

    Details of R-learner are available at Nie and Wager (2019) (https://arxiv.org/abs/1712.04912).
    """

    def __init__(self,
                 learner=None,
                 outcome_learner=None,
                 effect_learner=None,
                 ate_alpha=.05,
                 control_name=0,
                 n_fold=5,
                 random_state=None):
        """Initialize an R-learner.

        Args:
            learner (optional): a model to estimate outcomes and treatment effects
            outcome_learner (optional): a model to estimate outcomes
            effect_learner (optional): a model to estimate treatment effects. It needs to take `sample_weight` as an
                input argument for `fit()`
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
            n_fold (int, optional): the number of cross validation folds for outcome_learner
            random_state (int or RandomState, optional): a seed (int) or random number generator (RandomState)
        """
        assert (learner is not None) or ((outcome_learner is not None) and (effect_learner is not None))

        if outcome_learner is None:
            self.model_mu = deepcopy(learner)
        else:
            self.model_mu = outcome_learner

        if effect_learner is None:
            self.model_tau = deepcopy(learner)
        else:
            self.model_tau = effect_learner

        self.ate_alpha = ate_alpha
        self.control_name = control_name

        self.random_state = random_state
        self.cv = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)

        self.propensity = None
        self.propensity_model = None

    def __repr__(self):
        return ('{}(model_mu={},\n'
                '\tmodel_tau={})'.format(self.__class__.__name__,
                                         self.model_mu.__repr__(),
                                         self.model_tau.__repr__()))

    def fit(self, X, treatment, y, p=None, verbose=True):
        """Fit the treatment effect and outcome models of the R learner.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            verbose (bool, optional): whether to output progress logs
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()

        if p is None:
            logger.info('Generating propensity score')
            p = dict()
            p_model = dict()
            for group in self.t_groups:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                X_filt = X[mask]
                w_filt = (treatment_filt == group).astype(int)
                w = (treatment == group).astype(int)
                p[group], p_model[group] = compute_propensity_score(X=X_filt, treatment=w_filt,
                                                                    X_pred=X, treatment_pred=w)
            self.propensity_model = p_model
            self.propensity = p
        else:
            check_p_conditions(p, self.t_groups)

        if isinstance(p, (np.ndarray, pd.Series)):
            treatment_name = self.t_groups[0]
            p = {treatment_name: convert_pd_to_np(p)}
        elif isinstance(p, dict):
            p = {treatment_name: convert_pd_to_np(_p) for treatment_name, _p in p.items()}

        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_tau = {group: deepcopy(self.model_tau) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        if verbose:
            logger.info('generating out-of-fold CV outcome estimates')
        yhat = cross_val_predict(self.model_mu, X, y, cv=self.cv, n_jobs=-1)

        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            y_filt = y[mask]
            yhat_filt = yhat[mask]
            p_filt = p[group][mask]
            w = (treatment_filt == group).astype(int)

            if verbose:
                logger.info('training the treatment effect model for {} with R-loss'.format(group))
            self.models_tau[group].fit(X_filt, (y_filt - yhat_filt) / (w - p_filt),
                                       sample_weight=(w - p_filt) ** 2)

            self.vars_c[group] = (y_filt[w == 0] - yhat_filt[w == 0]).var()
            self.vars_t[group] = (y_filt[w == 1] - yhat_filt[w == 1]).var()

    def predict(self, X):
        """Predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X = convert_pd_to_np(X)
        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            dhat = self.models_tau[group].predict(X)
            te[:, i] = dhat

        return te

    def fit_predict(self, X, treatment, y, p=None, return_ci=False,
                    n_bootstraps=1000, bootstrap_size=10000, verbose=True):
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
            verbose (bool): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects. Output dim: [n_samples, n_treatment].
                If return_ci, returns CATE [n_samples, n_treatment], LB [n_samples, n_treatment],
                UB [n_samples, n_treatment]
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        self.fit(X, treatment, y, p, verbose=verbose)
        te = self.predict(X)

        if p is None:
            p = self.propensity
        else:
            check_p_conditions(p, self.t_groups)

        if isinstance(p, (np.ndarray, pd.Series)):
            treatment_name = self.t_groups[0]
            p = {treatment_name: convert_pd_to_np(p)}
        elif isinstance(p, dict):
            p = {treatment_name: convert_pd_to_np(_p) for treatment_name, _p in p.items()}

        if not return_ci:
            return te
        else:
            t_groups_global = self.t_groups
            _classes_global = self._classes
            model_mu_global = deepcopy(self.model_mu)
            models_tau_global = deepcopy(self.models_tau)
            te_bootstraps = np.zeros(shape=(X.shape[0], self.t_groups.shape[0], n_bootstraps))

            logger.info('Bootstrap Confidence Intervals')
            for i in tqdm(range(n_bootstraps)):
                te_b = self.bootstrap(X, treatment, y, p, size=bootstrap_size)
                te_bootstraps[:, :, i] = te_b

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha / 2) * 100, axis=2)
            te_upper = np.percentile(te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=2)

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.model_mu = deepcopy(model_mu_global)
            self.models_tau = deepcopy(models_tau_global)

            return (te, te_lower, te_upper)

    def estimate_ate(self, X, treatment, y, p=None, bootstrap_ci=False, n_bootstraps=1000, bootstrap_size=10000):
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
        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        te = self.fit_predict(X, treatment, y, p)

        if p is None:
            p = self.propensity
        else:
            check_p_conditions(p, self.t_groups)

        if isinstance(p, (np.ndarray, pd.Series)):
            treatment_name = self.t_groups[0]
            p = {treatment_name: convert_pd_to_np(p)}
        elif isinstance(p, dict):
            p = {treatment_name: convert_pd_to_np(_p) for treatment_name, _p in p.items()}

        ate = np.zeros(self.t_groups.shape[0])
        ate_lb = np.zeros(self.t_groups.shape[0])
        ate_ub = np.zeros(self.t_groups.shape[0])

        for i, group in enumerate(self.t_groups):
            w = (treatment == group).astype(int)
            prob_treatment = float(sum(w)) / X.shape[0]
            _ate = te[:, i].mean()

            se = (np.sqrt((self.vars_t[group] / prob_treatment)
                          + (self.vars_c[group] / (1 - prob_treatment))
                          + te[:, i].var())
                  / X.shape[0])

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
            model_mu_global = deepcopy(self.model_mu)
            models_tau_global = deepcopy(self.models_tau)

            logger.info('Bootstrap Confidence Intervals for ATE')
            ate_bootstraps = np.zeros(shape=(self.t_groups.shape[0], n_bootstraps))

            for n in tqdm(range(n_bootstraps)):
                cate_b = self.bootstrap(X, treatment, y, p, size=bootstrap_size)
                ate_bootstraps[:, n] = cate_b.mean()

            ate_lower = np.percentile(ate_bootstraps, (self.ate_alpha / 2) * 100, axis=1)
            ate_upper = np.percentile(ate_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=1)

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.model_mu = deepcopy(model_mu_global)
            self.models_tau = deepcopy(models_tau_global)
            return ate, ate_lower, ate_upper

    def bootstrap(self, X, treatment, y, p, size=10000):
        """Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population."""

        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]
        p_b = {group: _p[idxs] for group, _p in p.items()}
        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b, p=p_b, verbose=False)
        te_b = self.predict(X=X)
        return te_b

    def get_importance(self, X=None, tau=None, model_tau_feature=None, features=None, method='auto', normalize=True,
                       test_size=0.3, random_state=None):
        """
        Builds a model (using X to predict estimated/actual tau), and then calculates feature importances
        based on a specified method.

        Currently supported methods include:
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
            normalize (bool): normalize by sum of importances if method=gini (defaults to True)
            test_size (float/int): if float, represents the proportion of the dataset to include in the test split.
                                   If int, represents the absolute number of test samples (used for estimating
                                   permutation importance)
            random_state (int/RandomState instance/None): random state used in permutation importance estimation
        """
        explainer = Explainer(method=method, control_name=self.control_name,
                              X=X, tau=tau, model_tau=model_tau_feature,
                              features=features, classes=self._classes, normalize=normalize,
                              test_size=test_size, random_state=random_state)
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
        explainer = Explainer(method='shapley', control_name=self.control_name,
                              X=X, tau=tau, model_tau=model_tau_feature,
                              features=features, classes=self._classes)
        return explainer.get_shap_values()

    def plot_importance(self, X=None, tau=None, model_tau_feature=None, features=None, method='auto', normalize=True,
                        test_size=0.3, random_state=None):
        """
        Builds a model (using X to predict estimated/actual tau), and then plots feature importances
        based on a specified method.

        Currently supported methods include:
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
            normalize (bool): normalize by sum of importances if method=gini (defaults to True)
            test_size (float/int): if float, represents the proportion of the dataset to include in the test split.
                                   If int, represents the absolute number of test samples (used for estimating
                                   permutation importance)
            random_state (int/RandomState instance/None): random state used in permutation importance estimation
        """
        explainer = Explainer(method=method, control_name=self.control_name,
                              X=X, tau=tau, model_tau=model_tau_feature,
                              features=features, classes=self._classes, normalize=normalize,
                              test_size=test_size, random_state=random_state)
        explainer.plot_importance()

    def plot_shap_values(self, X=None, tau=None, model_tau_feature=None, features=None, shap_dict=None, **kwargs):
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
        explainer = Explainer(method='shapley', control_name=self.control_name,
                              X=X, tau=tau, model_tau=model_tau_feature,
                              features=features, override_checks=override_checks, classes=self._classes)
        explainer.plot_shap_values(shap_dict=shap_dict)

    def plot_shap_dependence(self, treatment_group, feature_idx, X, tau, model_tau_feature=None, features=None,
                             shap_dict=None, interaction_idx='auto', **kwargs):
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
        explainer = Explainer(method='shapley', control_name=self.control_name,
                              X=X, tau=tau, model_tau=model_tau_feature,
                              features=features, override_checks=override_checks,
                              classes=self._classes)
        explainer.plot_shap_dependence(treatment_group=treatment_group,
                                       feature_idx=feature_idx,
                                       shap_dict=shap_dict,
                                       interaction_idx=interaction_idx,
                                       **kwargs)


class BaseRRegressor(BaseRLearner):
    """
    A parent class for R-learner regressor classes.
    """

    def __init__(self,
                 learner=None,
                 outcome_learner=None,
                 effect_learner=None,
                 ate_alpha=.05,
                 control_name=0,
                 n_fold=5,
                 random_state=None):
        """Initialize an R-learner regressor.

        Args:
            learner (optional): a model to estimate outcomes and treatment effects
            outcome_learner (optional): a model to estimate outcomes
            effect_learner (optional): a model to estimate treatment effects. It needs to take `sample_weight` as an
                input argument for `fit()`
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
            n_fold (int, optional): the number of cross validation folds for outcome_learner
            random_state (int or RandomState, optional): a seed (int) or random number generator (RandomState)
        """
        super().__init__(
            learner=learner,
            outcome_learner=outcome_learner,
            effect_learner=effect_learner,
            ate_alpha=ate_alpha,
            control_name=control_name,
            n_fold=n_fold,
            random_state=random_state)


class BaseRClassifier(BaseRLearner):
    """
    A parent class for R-learner classifier classes.
    """

    def __init__(self,
                 outcome_learner=None,
                 effect_learner=None,
                 ate_alpha=.05,
                 control_name=0,
                 n_fold=5,
                 random_state=None):
        """Initialize an R-learner classifier.

        Args:
            outcome_learner: a model to estimate outcomes. Should be a classifier.
            effect_learner: a model to estimate treatment effects. It needs to take `sample_weight` as an
                input argument for `fit()`. Should be a regressor.
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
            n_fold (int, optional): the number of cross validation folds for outcome_learner
            random_state (int or RandomState, optional): a seed (int) or random number generator (RandomState)
        """
        super().__init__(
            learner=None,
            outcome_learner=outcome_learner,
            effect_learner=effect_learner,
            ate_alpha=ate_alpha,
            control_name=control_name,
            n_fold=n_fold,
            random_state=random_state)

        if (outcome_learner is None) and (effect_learner is None):
            raise ValueError("Either the outcome learner or the effect learner must be specified.")

    def fit(self, X, treatment, y, p=None, verbose=True):
        """Fit the treatment effect and outcome models of the R learner.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            verbose (bool, optional): whether to output progress logs
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()

        if p is None:
            logger.info('Generating propensity score')
            p = dict()
            p_model = dict()
            for group in self.t_groups:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                X_filt = X[mask]
                w_filt = (treatment_filt == group).astype(int)
                w = (treatment == group).astype(int)
                p[group], p_model[group] = compute_propensity_score(X=X_filt, treatment=w_filt,
                                                                    X_pred=X, treatment_pred=w)
            self.propensity_model = p_model
            self.propensity = p
        else:
            check_p_conditions(p, self.t_groups)

        if isinstance(p, (np.ndarray, pd.Series)):
            treatment_name = self.t_groups[0]
            p = {treatment_name: convert_pd_to_np(p)}
        elif isinstance(p, dict):
            p = {treatment_name: convert_pd_to_np(_p) for treatment_name, _p in p.items()}

        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_tau = {group: deepcopy(self.model_tau) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        if verbose:
            logger.info('generating out-of-fold CV outcome estimates')
        yhat = cross_val_predict(self.model_mu, X, y, cv=self.cv, method='predict_proba', n_jobs=-1)[:, 1]

        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            y_filt = y[mask]
            yhat_filt = yhat[mask]
            p_filt = p[group][mask]
            w = (treatment_filt == group).astype(int)

            if verbose:
                logger.info('training the treatment effect model for {} with R-loss'.format(group))
            self.models_tau[group].fit(X_filt, (y_filt - yhat_filt) / (w - p_filt),
                                       sample_weight=(w - p_filt) ** 2)

            self.vars_c[group] = (y_filt[w == 0] - yhat_filt[w == 0]).var()
            self.vars_t[group] = (y_filt[w == 1] - yhat_filt[w == 1]).var()

    def predict(self, X):
        """Predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            dhat = self.models_tau[group].predict(X)
            te[:, i] = dhat

        return te


class XGBRRegressor(BaseRRegressor):
    def __init__(self,
                 early_stopping=True,
                 test_size=0.3,
                 early_stopping_rounds=30,
                 effect_learner_objective='rank:pairwise',
                 effect_learner_n_estimators=500,
                 random_state=42,
                 *args,
                 **kwargs):
        """Initialize an R-learner regressor with XGBoost model using pairwise ranking objective.

        Args:
            early_stopping: whether or not to use early stopping when fitting effect learner
            test_size (float, optional): the proportion of the dataset to use as validation set when early stopping is
                                         enabled
            early_stopping_rounds (int, optional): validation metric needs to improve at least once in every
                                                   early_stopping_rounds round(s) to continue training
            effect_learner_objective (str, optional): the learning objective for the effect learner
                                                      (default = 'rank:pairwise')
            effect_learner_n_estimators (int, optional): number of trees to fit for the effect learner (default = 500)
        """

        assert isinstance(random_state, int), 'random_state should be int.'

        objective, metric = get_xgboost_objective_metric(effect_learner_objective)
        self.effect_learner_objective = objective
        self.effect_learner_eval_metric = metric
        self.effect_learner_n_estimators = effect_learner_n_estimators
        self.early_stopping = early_stopping
        if self.early_stopping:
            self.test_size = test_size
            self.early_stopping_rounds = early_stopping_rounds

        super().__init__(
            outcome_learner=XGBRegressor(random_state=random_state, *args, **kwargs),
            effect_learner=XGBRegressor(objective=self.effect_learner_objective,
                                        n_estimators=self.effect_learner_n_estimators,
                                        random_state=random_state,
                                        *args,
                                        **kwargs)
        )

    def fit(self, X, treatment, y, p=None, verbose=True):
        """Fit the treatment effect and outcome models of the R learner.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            verbose (bool, optional): whether to output progress logs
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()

        if p is None:
            logger.info('Generating propensity score')
            p = dict()
            p_model = dict()
            for group in self.t_groups:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                X_filt = X[mask]
                w_filt = (treatment_filt == group).astype(int)
                w = (treatment == group).astype(int)
                p[group], p_model[group] = compute_propensity_score(X=X_filt, treatment=w_filt,
                                                                    X_pred=X, treatment_pred=w)
            self.propensity_model = p_model
            self.propensity = p
        else:
            check_p_conditions(p, self.t_groups)

        if isinstance(p, (np.ndarray, pd.Series)):
            treatment_name = self.t_groups[0]
            p = {treatment_name: convert_pd_to_np(p)}
        elif isinstance(p, dict):
            p = {treatment_name: convert_pd_to_np(_p) for treatment_name, _p in p.items()}

        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_tau = {group: deepcopy(self.model_tau) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        if verbose:
            logger.info('generating out-of-fold CV outcome estimates')
        yhat = cross_val_predict(self.model_mu, X, y, cv=self.cv, n_jobs=-1)

        for group in self.t_groups:
            treatment_mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[treatment_mask]
            w = (treatment_filt == group).astype(int)

            X_filt = X[treatment_mask]
            y_filt = y[treatment_mask]
            yhat_filt = yhat[treatment_mask]
            p_filt = p[group][treatment_mask]

            if verbose:
                logger.info('training the treatment effect model for {} with R-loss'.format(group))

            if self.early_stopping:
                X_train_filt, X_test_filt, y_train_filt, y_test_filt, yhat_train_filt, yhat_test_filt, \
                    w_train, w_test, p_train_filt, p_test_filt = train_test_split(
                        X_filt, y_filt, yhat_filt, w, p_filt,
                        test_size=self.test_size, random_state=self.random_state
                    )

                self.models_tau[group].fit(X=X_train_filt,
                                           y=(y_train_filt - yhat_train_filt) / (w_train - p_train_filt),
                                           sample_weight=(w_train - p_train_filt) ** 2,
                                           eval_set=[(X_test_filt,
                                                      (y_test_filt - yhat_test_filt) / (w_test - p_test_filt))],
                                           sample_weight_eval_set=[(w_test - p_test_filt) ** 2],
                                           eval_metric=self.effect_learner_eval_metric,
                                           early_stopping_rounds=self.early_stopping_rounds,
                                           verbose=verbose)

            else:
                self.models_tau[group].fit(X_filt, (y_filt - yhat_filt) / (w - p_filt),
                                           sample_weight=(w - p_filt) ** 2,
                                           eval_metric=self.effect_learner_eval_metric)

            self.vars_c[group] = (y_filt[w == 0] - yhat_filt[w == 0]).var()
            self.vars_t[group] = (y_filt[w == 1] - yhat_filt[w == 1]).var()
