from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from future.builtins import super
from copy import deepcopy
import logging
import pandas as pd
import numpy as np
from scipy.stats import norm

from .utils import check_control_in_treatment, check_p_conditions
from causalml.metrics import regression_metrics, classification_metrics


logger = logging.getLogger('causalml')


class BaseXLearner(object):
    """A parent class for X-learner regressor classes.

    An X-learner estimates treatment effects with four machine learning models.

    Details of X-learner are available at Kunzel et al. (2018) (https://arxiv.org/abs/1706.03461).
    """

    def __init__(self,
                 learner=None,
                 control_outcome_learner=None,
                 treatment_outcome_learner=None,
                 control_effect_learner=None,
                 treatment_effect_learner=None,
                 ate_alpha=.05,
                 control_name=0):
        """Initialize a X-learner.

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
        assert (learner is not None) or ((control_outcome_learner is not None) and
                                         (treatment_outcome_learner is not None) and
                                         (control_effect_learner is not None) and
                                         (treatment_effect_learner is not None))

        if control_outcome_learner is None:
            self.model_mu_c = deepcopy(learner)
        else:
            self.model_mu_c = control_outcome_learner

        if treatment_outcome_learner is None:
            self.model_mu_t = deepcopy(learner)
        else:
            self.model_mu_t = treatment_outcome_learner

        if control_effect_learner is None:
            self.model_tau_c = deepcopy(learner)
        else:
            self.model_tau_c = control_effect_learner

        if treatment_effect_learner is None:
            self.model_tau_t = deepcopy(learner)
        else:
            self.model_tau_t = treatment_effect_learner

        self.ate_alpha = ate_alpha
        self.control_name = control_name

    def __repr__(self):
        return ('{}(control_outcome_learner={},\n'
                '\ttreatment_outcome_learner={},\n'
                '\tcontrol_effect_learner={},\n'
                '\ttreatment_effect_learner={})'.format(self.__class__.__name__,
                                                        self.model_mu_c.__repr__(),
                                                        self.model_mu_t.__repr__(),
                                                        self.model_tau_c.__repr__(),
                                                        self.model_tau_t.__repr__()))

    def fit(self, X, treatment, y):
        """Fit the inference model.

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
        """
        check_control_in_treatment(treatment, self.control_name)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()
        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_mu_c = {group: deepcopy(self.model_mu_c) for group in self.t_groups}
        self.models_mu_t = {group: deepcopy(self.model_mu_t) for group in self.t_groups}
        self.models_tau_c = {group: deepcopy(self.model_tau_c) for group in self.t_groups}
        self.models_tau_t = {group: deepcopy(self.model_tau_t) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            y_filt = y[mask]
            w = (treatment_filt == group).astype(int)

            # Train outcome models
            self.models_mu_c[group].fit(X_filt[w == 0], y_filt[w == 0])
            self.models_mu_t[group].fit(X_filt[w == 1], y_filt[w == 1])

            # Calculate variances and treatment effects
            var_c = (y_filt[w == 0] - self.models_mu_c[group].predict(X_filt[w == 0])).var()
            self.vars_c[group] = var_c
            var_t = (y_filt[w == 1] - self.models_mu_t[group].predict(X_filt[w == 1])).var()
            self.vars_t[group] = var_t

            # Train treatment models
            d_c = self.models_mu_t[group].predict(X_filt[w == 0]) - y_filt[w == 0]
            d_t = y_filt[w == 1] - self.models_mu_c[group].predict(X_filt[w == 1])
            self.models_tau_c[group].fit(X_filt[w == 0], d_c)
            self.models_tau_t[group].fit(X_filt[w == 1], d_t)

    def predict(self, X, p, treatment=None, y=None, return_components=False, verbose=True):
        """Predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            p (np.ndarray or dict): an array of propensity scores of float (0,1) in the single-treatment case
                                    or, a dictionary of treatment groups that map to propensity vectors of float (0,1)
            treatment (np.array, optional): a treatment vector
            y (np.array, optional): an optional outcome vector

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        check_p_conditions(p, self.t_groups)
        if isinstance(p, np.ndarray):
            treatment_name = self.t_groups[0]
            p = {treatment_name: p}

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        dhat_cs = {}
        dhat_ts = {}

        for i, group in enumerate(self.t_groups):
            model_tau_c = self.models_tau_c[group]
            model_tau_t = self.models_tau_t[group]
            dhat_cs[group] = model_tau_c.predict(X)
            dhat_ts[group] = model_tau_t.predict(X)

            _te = (p[group] * dhat_cs[group] + (1 - p[group]) * dhat_cs[group]).reshape(-1, 1)
            te[:, i] = np.ravel(_te)

            if (y is not None) and (treatment is not None) and verbose:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                X_filt = X[mask]
                y_filt = y[mask]
                w = (treatment_filt == group).astype(int)

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = self.models_mu_c[group].predict(X_filt[w == 0])
                yhat[w == 1] = self.models_mu_t[group].predict(X_filt[w == 1])

                logger.info('Error metrics for group {}'.format(group))
                regression_metrics(y_filt, yhat, w)

        if not return_components:
            return te
        else:
            return te, dhat_cs, dhat_ts

    def fit_predict(self, X, p, treatment, y, return_ci=False, n_bootstraps=1000, bootstrap_size=10000,
                    return_components=False, verbose=True):
        """Fit the treatment effect and outcome models of the R learner and predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            p (np.ndarray or dict): an array of propensity scores of float (0,1) in the single-treatment case
                                    or, a dictionary of treatment groups that map to propensity vectors of float (0,1)
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
            return_ci (bool): whether to return confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            verbose (str): whether to output progress logs

        Returns:
            (numpy.ndarray): Predictions of treatment effects. Output dim: [n_samples, n_treatment]
                If return_ci, returns CATE [n_samples, n_treatment], LB [n_samples, n_treatment],
                UB [n_samples, n_treatment]
        """
        self.fit(X, treatment, y)
        te = self.predict(X, p, treatment=treatment, y=y, return_components=return_components)

        if not return_ci:
            return te
        else:
            start = pd.datetime.today()
            self.t_groups_global = self.t_groups
            self._classes_global = self._classes
            self.models_mu_c_global = deepcopy(self.models_mu_c)
            self.models_mu_t_global = deepcopy(self.models_mu_t)
            self.models_tau_c_global = deepcopy(self.models_tau_c)
            self.models_tau_t_global = deepcopy(self.models_tau_t)
            te_bootstraps = np.zeros(shape=(X.shape[0], self.t_groups.shape[0], n_bootstraps))
            for i in range(n_bootstraps):
                te_b = self.bootstrap(X, p, treatment, y, size=bootstrap_size)
                te_bootstraps[:, :, i] = te_b
                if verbose and i % 10 == 0 and i > 0:
                    now = pd.datetime.today()
                    lapsed = (now-start).seconds
                    logger.info('{}/{} bootstraps completed. ({}s lapsed)'.format(i, n_bootstraps, lapsed))

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha / 2) * 100, axis=2)
            te_upper = np.percentile(te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=2)

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = self.t_groups_global
            self._classes = self._classes_global
            self.models_mu_c = self.models_mu_c_global
            self.models_mu_t = self.models_mu_t_global
            self.models_tau_c = self.models_tau_c_global
            self.models_tau_t = self.models_tau_t_global

            return (te, te_lower, te_upper)

    def estimate_ate(self, X, p, treatment, y):
        """Estimate the Average Treatment Effect (ATE).

        Args:
            X (np.matrix): a feature matrix
            p (np.ndarray or dict): an array of propensity scores of float (0,1) in the single-treatment case
                                    or, a dictionary of treatment groups that map to propensity vectors of float (0,1)
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector

        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        te, dhat_cs, dhat_ts = self.fit_predict(X, p, treatment, y, return_components=True)

        check_p_conditions(p, self.t_groups)
        if isinstance(p, np.ndarray):
            treatment_name = treatment_name = self.t_groups[0]
            p = {treatment_name: p}

        ate = np.zeros(self.t_groups.shape[0])
        ate_lb = np.zeros(self.t_groups.shape[0])
        ate_ub = np.zeros(self.t_groups.shape[0])

        for i, group in enumerate(self.t_groups):
            _ate = te[:, i].mean()

            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            w = (treatment_filt == group).astype(int)
            prob_treatment = float(sum(w)) / w.shape[0]

            dhat_c = dhat_cs[group][mask]
            dhat_t = dhat_ts[group][mask]
            p_filt = p[group][mask]

            # SE formula is based on the lower bound formula (7) from Imbens, Guido W., and Jeffrey M. Wooldridge. 2009.
            # "Recent Developments in the Econometrics of Program Evaluation." Journal of Economic Literature
            se = np.sqrt((
                self.vars_t[group] / prob_treatment + self.vars_c[group] / (1 - prob_treatment) +
                (p_filt * dhat_c + (1 - p_filt) * dhat_t).var()
            ) / w.shape[0])

            _ate_lb = _ate - se * norm.ppf(1 - self.ate_alpha / 2)
            _ate_ub = _ate + se * norm.ppf(1 - self.ate_alpha / 2)

            ate[i] = _ate
            ate_lb[i] = _ate_lb
            ate_ub[i] = _ate_ub

        return ate, ate_lb, ate_ub

    def bootstrap(self, X, p, treatment, y, size=10000):
        """Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population."""
        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]
        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b)
        te_b = self.predict(X=X, p=p)
        return te_b


class BaseXRegressor(BaseXLearner):
    """
    A parent class for X-learner regressor classes.
    """

    def __init__(self,
                 learner=None,
                 control_outcome_learner=None,
                 treatment_outcome_learner=None,
                 control_effect_learner=None,
                 treatment_effect_learner=None,
                 ate_alpha=.05,
                 control_name=0):
        """Initialize an X-learner regressor.

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
            control_effect_learner=control_effect_learner,
            treatment_effect_learner=treatment_effect_learner,
            ate_alpha=ate_alpha,
            control_name=control_name)


class BaseXClassifier(BaseXLearner):
    """
    A parent class for X-learner classifier classes.
    """

    def __init__(self,
                 learner=None,
                 control_outcome_learner=None,
                 treatment_outcome_learner=None,
                 control_effect_learner=None,
                 treatment_effect_learner=None,
                 ate_alpha=.05,
                 control_name=0):
        """Initialize an X-learner classifier.

        Args:
            learner (optional): a model to estimate outcomes or treatment effects in both the control and treatment
                groups. Even if specified, the user must still input either the outcome learner or the effect learner
                pair.
            control_outcome_learner (optional): a model to estimate outcomes in the control group.
                Should have a predict_proba() method.
            treatment_outcome_learner (optional): a model to estimate outcomes in the treatment group.
                Should have a predict_proba() method.
            control_effect_learner (optional): a model to estimate treatment effects in the control group
            treatment_effect_learner (optional): a model to estimate treatment effects in the treatment group
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        super().__init__(
            learner=learner,
            control_outcome_learner=control_outcome_learner,
            treatment_outcome_learner=treatment_outcome_learner,
            control_effect_learner=control_effect_learner,
            treatment_effect_learner=treatment_effect_learner,
            ate_alpha=ate_alpha,
            control_name=control_name)

        if ((control_outcome_learner is None) or (treatment_outcome_learner is None)) and (
                (control_effect_learner is None) or (treatment_effect_learner is None)):
            raise ValueError("Either the outcome learner or the effect learner pair must be specified.")

    def fit(self, X, treatment, y):
        """Fit the inference model.

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
        """
        check_control_in_treatment(treatment, self.control_name)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()
        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_mu_c = {group: deepcopy(self.model_mu_c) for group in self.t_groups}
        self.models_mu_t = {group: deepcopy(self.model_mu_t) for group in self.t_groups}
        self.models_tau_c = {group: deepcopy(self.model_tau_c) for group in self.t_groups}
        self.models_tau_t = {group: deepcopy(self.model_tau_t) for group in self.t_groups}
        self.vars_c = {}
        self.vars_t = {}

        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            y_filt = y[mask]
            w = (treatment_filt == group).astype(int)

            # Train outcome models
            self.models_mu_c[group].fit(X_filt[w == 0], y_filt[w == 0])
            self.models_mu_t[group].fit(X_filt[w == 1], y_filt[w == 1])

            # Calculate variances and treatment effects
            var_c = (y_filt[w == 0] - self.models_mu_c[group].predict_proba(X_filt[w == 0])[:, 1]).var()
            self.vars_c[group] = var_c
            var_t = (y_filt[w == 1] - self.models_mu_t[group].predict_proba(X_filt[w == 1])[:, 1]).var()
            self.vars_t[group] = var_t

            # Train treatment models
            d_c = self.models_mu_t[group].predict_proba(X_filt[w == 0])[:, 1] - y_filt[w == 0]
            d_t = y_filt[w == 1] - self.models_mu_c[group].predict_proba(X_filt[w == 1])[:, 1]
            self.models_tau_c[group].fit(X_filt[w == 0], d_c)
            self.models_tau_t[group].fit(X_filt[w == 1], d_t)

    def predict(self, X, p, treatment=None, y=None, return_components=False, verbose=True):
        """Predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            p (np.ndarray or dict): an array of propensity scores of float (0,1) in the single-treatment case
                                    or, a dictionary of treatment groups that map to propensity vectors of float (0,1)
            treatment (np.array, optional): a treatment vector
            y (np.array, optional): an optional outcome vector

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        check_p_conditions(p, self.t_groups)
        if isinstance(p, np.ndarray):
            treatment_name = self.t_groups[0]
            p = {treatment_name: p}

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        dhat_cs = {}
        dhat_ts = {}

        for i, group in enumerate(self.t_groups):
            model_tau_c = self.models_tau_c[group]
            model_tau_t = self.models_tau_t[group]
            dhat_cs[group] = model_tau_c.predict(X)
            dhat_ts[group] = model_tau_t.predict(X)

            _te = (p[group] * dhat_cs[group] + (1 - p[group]) * dhat_cs[group]).reshape(-1, 1)
            te[:, i] = np.ravel(_te)

            if (y is not None) and (treatment is not None) and verbose:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                X_filt = X[mask]
                y_filt = y[mask]
                w = (treatment_filt == group).astype(int)

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = self.models_mu_c[group].predict_proba(X_filt[w == 0])[:, 1]
                yhat[w == 1] = self.models_mu_t[group].predict_proba(X_filt[w == 1])[:, 1]

                logger.info('Error metrics for group {}'.format(group))
                classification_metrics(y_filt, yhat, w)

        if not return_components:
            return te
        else:
            return te, dhat_cs, dhat_ts
