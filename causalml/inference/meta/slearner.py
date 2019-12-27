import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from sklearn.dummy import DummyRegressor
import statsmodels.api as sm
from copy import deepcopy

from causalml.inference.meta.explainer import Explainer
from causalml.inference.meta.utils import check_treatment_vector, convert_pd_to_np
from causalml.metrics import regression_metrics, classification_metrics


logger = logging.getLogger('causalml')


class StatsmodelsOLS(object):
    """A sklearn style wrapper class for statsmodels' OLS."""

    def __init__(self, cov_type='HC1', alpha=.05):
        """Initialize a statsmodels' OLS wrapper class object.
        Args:
            cov_type (str, optional): covariance estimator type.
            alpha (float, optional): the confidence level alpha.
        """
        self.cov_type = cov_type
        self.alpha = alpha

    def fit(self, X, y):
        """Fit OLS.
        Args:
            X (np.matrix): a feature matrix
            y (np.array): a label vector
        """
        # Append ones. The first column is for the treatment indicator.
        X = sm.add_constant(X, prepend=False, has_constant='add')
        self.model = sm.OLS(y, X).fit(cov_type=self.cov_type)
        self.coefficients = self.model.params
        self.conf_ints = self.model.conf_int(alpha=self.alpha)

    def predict(self, X):
        # Append ones. The first column is for the treatment indicator.
        X = sm.add_constant(X, prepend=False, has_constant='add')
        return self.model.predict(X)


class BaseSLearner(object):
    """A parent class for S-learner classes.
    An S-learner estimates treatment effects with one machine learning model.
    Details of S-learner are available at Kunzel et al. (2018) (https://arxiv.org/abs/1706.03461).
    """

    def __init__(self, learner=None, ate_alpha=0.05, control_name=0):
        """Initialize an S-learner.
        Args:
            learner (optional): a model to estimate the treatment effect
            control_name (str or int, optional): name of control group
        """
        if learner is not None:
            self.model = learner
        else:
            self.model = DummyRegressor()
        self.ate_alpha = ate_alpha
        self.control_name = control_name

    def __repr__(self):
        return '{}(model={})'.format(self.__class__.__name__,
                                     self.model.__repr__())

    def fit(self, X, treatment, y):
        """Fit the inference model
        Args:
            X (np.matrix, np.array, or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()
        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models = {group: deepcopy(self.model) for group in self.t_groups}

        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            y_filt = y[mask]

            w = (treatment_filt == group).astype(int)
            X_new = np.hstack((w.reshape((-1, 1)), X_filt))
            self.models[group].fit(X_new, y_filt)

    def predict(self, X, treatment=None, y=None, return_components=False, verbose=True):
        """Predict treatment effects.
        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series, optional): a treatment vector
            y (np.array or pd.Series, optional): an outcome vector
            return_components (bool, optional): whether to return outcome for treatment and control seperately
            verbose (bool, optional): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        yhat_cs = {}
        yhat_ts = {}

        for group in self.t_groups:
            model = self.models[group]

            # set the treatment column to zero (the control group)
            X_new = np.hstack((np.zeros((X.shape[0], 1)), X))
            yhat_cs[group] = model.predict(X_new)

            # set the treatment column to one (the treatment group)
            X_new[:, 0] = 1
            yhat_ts[group] = model.predict(X_new)

            if (y is not None) and (treatment is not None) and verbose:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                w = (treatment_filt == group).astype(int)
                y_filt = y[mask]

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = yhat_cs[group][mask][w == 0]
                yhat[w == 1] = yhat_ts[group][mask][w == 1]

                logger.info('Error metrics for group {}'.format(group))
                regression_metrics(y_filt, yhat, w)

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            te[:, i] = yhat_ts[group] - yhat_cs[group]

        if not return_components:
            return te
        else:
            return te, yhat_cs, yhat_ts

        return te

    def fit_predict(self, X, treatment, y, return_ci=False, n_bootstraps=1000, bootstrap_size=10000,
                    return_components=False, verbose=True):
        """Fit the inference model of the S learner and predict treatment effects.
        Args:
            X (np.matrix, np.array, or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            return_ci (bool, optional): whether to return confidence intervals
            n_bootstraps (int, optional): number of bootstrap iterations
            bootstrap_size (int, optional): number of samples per bootstrap
            return_components (bool, optional): whether to return outcome for treatment and control seperately
            verbose (bool, optional): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects. Output dim: [n_samples, n_treatment].
                If return_ci, returns CATE [n_samples, n_treatment], LB [n_samples, n_treatment],
                UB [n_samples, n_treatment]
        """
        self.fit(X, treatment, y)
        te = self.predict(X, treatment, y, return_components=return_components)

        if not return_ci:
            return te
        else:
            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_global = deepcopy(self.models)
            te_bootstraps = np.zeros(shape=(X.shape[0], self.t_groups.shape[0], n_bootstraps))

            logger.info('Bootstrap Confidence Intervals')
            for i in tqdm(range(n_bootstraps)):
                te_b = self.bootstrap(X, treatment, y, size=bootstrap_size)
                te_bootstraps[:, :, i] = te_b

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha/2)*100, axis=2)
            te_upper = np.percentile(te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=2)

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.models = deepcopy(models_global)

            return (te, te_lower, te_upper)

    def estimate_ate(self, X, treatment, y, return_ci=False, bootstrap_ci=False,
                     n_bootstraps=1000, bootstrap_size=10000):
        """Estimate the Average Treatment Effect (ATE).

        Args:
            X (np.matrix, np.array, or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            return_ci (bool, optional): whether to return confidence intervals
            bootstrap_ci (bool): whether to return confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            verbose (str): whether to output progress logs

        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        te, yhat_cs, yhat_ts = self.fit_predict(X, treatment, y, return_components=True)

        ate = np.zeros(self.t_groups.shape[0])
        ate_lb = np.zeros(self.t_groups.shape[0])
        ate_ub = np.zeros(self.t_groups.shape[0])

        for i, group in enumerate(self.t_groups):
            _ate = te[:, i].mean()

            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            y_filt = y[mask]
            w = (treatment_filt == group).astype(int)
            prob_treatment = float(sum(w)) / w.shape[0]

            yhat_c = yhat_cs[group][mask]
            yhat_t = yhat_ts[group][mask]

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

        if not return_ci:
            return ate
        elif return_ci and not bootstrap_ci:
            return ate, ate_lb, ate_ub
        else:
            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_global = deepcopy(self.models)

            logger.info('Bootstrap Confidence Intervals for ATE')
            ate_bootstraps = np.zeros(shape=(self.t_groups.shape[0], n_bootstraps))

            for n in tqdm(range(n_bootstraps)):
                ate_b = self.bootstrap(X, treatment, y, size=bootstrap_size)
                ate_bootstraps[:, n] = ate_b.mean()

            ate_lower = np.percentile(ate_bootstraps, (self.ate_alpha / 2) * 100, axis=1)
            ate_upper = np.percentile(ate_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=1)

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.models = deepcopy(models_global)

            return ate, ate_lower, ate_upper

    def bootstrap(self, X, treatment, y, size=10000):
        """Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population.
        """
        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]
        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b)
        te_b = self.predict(X=X, treatment=treatment, verbose=False)
        return te_b

    def get_importance(self, X=None, tau=None, model_tau_feature=None, features=None, method='auto', normalize=True):
        """
        Builds a model (using X to predict estimated/actual tau), and then calculates feature importances
        based on a specified method.

        Currently supported methods are:
            - auto (calculates importance based on estimator's default implementation of feature importance;
                    estimator must be tree-based)
                    Note: if none provided, it uses lightgbm's LGBMRegressor as estimator, and "gain" as
                    importance type
            - permutation (calculates importance based on mean decrease in accuracy; estimator can be any form)
        Hint: for permutation, downsample data for better performance especially if X.shape[1] is large

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (np.array): list/array of feature names. If None, an enumerated list will be used.
            method (str): auto, permutation
            normalize (bool): normalize by sum of importances if method=auto (defaults to True)
        """
        explainer = Explainer(method=method, control_name=self.control_name,
                              X=X, tau=tau, model_tau=model_tau_feature,
                              features=features, classes=self._classes, normalize=normalize)
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

    def plot_importance(self, X=None, tau=None, model_tau_feature=None, features=None, method='auto', normalize=True):
        """
        Builds a model (using X to predict estimated/actual tau), and then plots feature importances
        based on a specified method.

        Currently supported methods are:
            - auto (calculates importance based on estimator's default implementation of feature importance;
                    estimator must be tree-based)
                    Note: if none provided, it uses lightgbm's LGBMRegressor as estimator, and "gain" as
                    importance type
            - permutation (calculates importance based on mean decrease in accuracy; estimator can be any form)
        Hint: for permutation, downsample data for better performance especially if X.shape[1] is large

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used.
            method (str): auto, permutation
            normalize (bool): normalize by sum of importances if method=auto (defaults to True)
        """
        explainer = Explainer(method=method, control_name=self.control_name,
                              X=X, tau=tau, model_tau=model_tau_feature,
                              features=features, classes=self._classes, normalize=normalize)
        explainer.plot_importance()

    def plot_shap_values(self, X=None, tau=None, model_tau_feature=None, features=None, shap_dict=None, **kwargs):
        """
        Plots distribution of shapley values.

        If shapley values have been pre-computed, pass it through the shap_dict parameter.
        If shap_dict is not provided, this builds a new model (using X to predict estimated/actual tau),
        and then calculates shapley values.

        Args:
            X (np.matrix, np.array, or pd.Dataframe): a feature matrix. Required if shap_dict is None.
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


class BaseSRegressor(BaseSLearner):
    """
    A parent class for S-learner regressor classes.
    """

    def __init__(self, learner=None, ate_alpha=0.05, control_name=0):
        """Initialize an S-learner regressor.
        Args:
            learner (optional): a model to estimate the treatment effect
            control_name (str or int, optional): name of control group
        """
        super().__init__(
            learner=learner,
            ate_alpha=ate_alpha,
            control_name=control_name)


class BaseSClassifier(BaseSLearner):
    """
    A parent class for S-learner classifier classes.
    """

    def __init__(self, learner=None, ate_alpha=0.05, control_name=0):
        """Initialize an S-learner classifier.
        Args:
            learner (optional): a model to estimate the treatment effect.
                Should have a predict_proba() method.
            control_name (str or int, optional): name of control group
        """
        super().__init__(
            learner=learner,
            ate_alpha=ate_alpha,
            control_name=control_name)

    def predict(self, X, treatment=None, y=None, verbose=True):
        """Predict treatment effects.
        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series, optional): a treatment vector
            y (np.array or pd.Series, optional): an outcome vector
        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        yhat_cs = {}
        yhat_ts = {}

        for group in self.t_groups:
            model = self.models[group]

            # set the treatment column to zero (the control group)
            X_new = np.hstack((np.zeros((X.shape[0], 1)), X))
            yhat_cs[group] = model.predict_proba(X_new)[:, 1]

            # set the treatment column to one (the treatment group)
            X_new[:, 0] = 1
            yhat_ts[group] = model.predict_proba(X_new)[:, 1]

            if y is not None and (treatment is not None) and verbose:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                w = (treatment_filt == group).astype(int)
                y_filt = y[mask]

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = yhat_cs[group][mask][w == 0]
                yhat[w == 1] = yhat_ts[group][mask][w == 1]

                logger.info('Error metrics for group {}'.format(group))
                classification_metrics(y_filt, yhat, w)

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            te[:, i] = yhat_ts[group] - yhat_cs[group]

        return te


class LRSRegressor(BaseSRegressor):
    def __init__(self, ate_alpha=.05, control_name=0):
        """Initialize an S-learner with a linear regression model.
        Args:
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        super().__init__(StatsmodelsOLS(alpha=ate_alpha), ate_alpha, control_name)

    def estimate_ate(self, X, treatment, y):
        """Estimate the Average Treatment Effect (ATE).
        Args:
            X (np.matrix, np.array, or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        self.fit(X, treatment, y)

        ate = np.zeros(self.t_groups.shape[0])
        ate_lb = np.zeros(self.t_groups.shape[0])
        ate_ub = np.zeros(self.t_groups.shape[0])

        for i, group in enumerate(self.t_groups):
            ate[i] = self.models[group].coefficients[0]
            ate_lb[i] = self.models[group].conf_ints[0, 0]
            ate_ub[i] = self.models[group].conf_ints[0, 1]

        return ate, ate_lb, ate_ub
