from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from future.builtins import super
from copy import deepcopy
import logging
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.utils.testing import ignore_warnings
from xgboost import XGBRegressor

from causalml.metrics import regression_metrics, classification_metrics


logger = logging.getLogger('causalml')


class BaseTLearner(object):
    """A parent class for T-learner regressor classes.

    A T-learner estimates treatment effects with two machine learning models.

    Details of T-learner are available at Kunzel et al. (2018) (https://arxiv.org/abs/1706.03461).
    """

    def __init__(self, learner=None, control_learner=None, treatment_learner=None, ate_alpha=.05, control_name=0):
        """Initialize a T-learner.

        Args:
            learner (model): a model to estimate control and treatment outcomes.
            control_learner (model, optional): a model to estimate control outcomes
            treatment_learner (model, optional): a model to estimate treatment outcomes
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        assert (learner is not None) or ((control_learner is not None) and (treatment_learner is not None))

        if control_learner is None:
            self.model_c = deepcopy(learner)
        else:
            self.model_c = control_learner

        if treatment_learner is None:
            self.model_t = deepcopy(learner)
        else:
            self.model_t = treatment_learner

        self.ate_alpha = ate_alpha
        self.control_name = control_name

    def __repr__(self):
        return '{}(model_c={}, model_t={})'.format(self.__class__.__name__,
                                                   self.model_c.__repr__(),
                                                   self.model_t.__repr__())

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, treatment, y):
        """Fit the inference model

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
        """
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()
        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_c = {group: deepcopy(self.model_c) for group in self.t_groups}
        self.models_t = {group: deepcopy(self.model_t) for group in self.t_groups}

        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            y_filt = y[mask]
            w = (treatment_filt == group).astype(int)

            self.models_c[group].fit(X_filt[w == 0], y_filt[w == 0])
            self.models_t[group].fit(X_filt[w == 1], y_filt[w == 1])

    def predict(self, X, treatment=None, y=None, return_components=False, verbose=True):
        """Predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array, optional): a treatment vector
            y (np.array, optional): an optional outcome vector

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        yhat_cs = {}
        yhat_ts = {}

        for group in self.t_groups:
            model_c = self.models_c[group]
            model_t = self.models_t[group]
            yhat_cs[group] = model_c.predict(X)
            yhat_ts[group] = model_t.predict(X)

            if (y is not None) and (treatment is not None) and verbose:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                y_filt = y[mask]
                w = (treatment_filt == group).astype(int)

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

    def fit_predict(self, X, treatment, y, return_ci=False, n_bootstraps=1000, bootstrap_size=10000,
                    return_components=False, verbose=True):
        """Fit the inference model of the T learner and predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
            return_ci (bool): whether to return confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            verbose (str): whether to output progress logs

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
            start = pd.datetime.today()
            self.t_groups_global = self.t_groups
            self._classes_global = self._classes
            self.models_c_global = deepcopy(self.models_c)
            self.models_t_global = deepcopy(self.models_t)
            te_bootstraps = np.zeros(shape=(X.shape[0], self.t_groups.shape[0], n_bootstraps))
            for i in range(n_bootstraps):
                te_b = self.bootstrap(X, treatment, y, size=bootstrap_size)
                te_bootstraps[:, :, i] = te_b
                if verbose and i % 10 == 0 and i > 0:
                    now = pd.datetime.today()
                    lapsed = (now-start).seconds
                    logger.info('{}/{} bootstraps completed. ({}s lapsed)'.format(i, n_bootstraps, lapsed))

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha/2)*100, axis=2)
            te_upper = np.percentile(te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=2)

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = self.t_groups_global
            self._classes = self._classes_global
            self.models_c = self.models_c_global
            self.models_t = self.models_t_global

            return (te, te_lower, te_upper)

    def estimate_ate(self, X, treatment, y):
        """Estimate the Average Treatment Effect (ATE).

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector

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

        return ate, ate_lb, ate_ub

    def bootstrap(self, X, treatment, y, size=10000):
        """Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population."""
        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]
        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b)
        te_b = self.predict(X=X, treatment=treatment, verbose=False)
        return te_b


class BaseTRegressor(BaseTLearner):
    """
    A parent class for T-learner regressor classes.
    """

    def __init__(self,
                 learner=None,
                 control_learner=None,
                 treatment_learner=None,
                 ate_alpha=.05,
                 control_name=0):
        """Initialize a T-learner regressor.

        Args:
            learner (model): a model to estimate control and treatment outcomes.
            control_learner (model, optional): a model to estimate control outcomes
            treatment_learner (model, optional): a model to estimate treatment outcomes
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        super().__init__(
            learner=learner,
            control_learner=control_learner,
            treatment_learner=treatment_learner,
            ate_alpha=ate_alpha,
            control_name=control_name)


class BaseTClassifier(BaseTLearner):
    """
    A parent class for T-learner classifier classes.
    """

    def __init__(self,
                 learner=None,
                 control_learner=None,
                 treatment_learner=None,
                 ate_alpha=.05,
                 control_name=0):
        """Initialize a T-learner classifier.

        Args:
            learner (model): a model to estimate control and treatment outcomes.
            control_learner (model, optional): a model to estimate control outcomes
            treatment_learner (model, optional): a model to estimate treatment outcomes
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        super().__init__(
            learner=learner,
            control_learner=control_learner,
            treatment_learner=treatment_learner,
            ate_alpha=ate_alpha,
            control_name=control_name)

    def predict(self, X, treatment=None, y=None, return_components=False, verbose=True):
        """Predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array, optional): a treatment vector
            y (np.array, optional): an optional outcome vector

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        yhat_cs = {}
        yhat_ts = {}

        for group in self.t_groups:
            model_c = self.models_c[group]
            model_t = self.models_t[group]
            yhat_cs[group] = model_c.predict_proba(X)[:, 1]
            yhat_ts[group] = model_t.predict_proba(X)[:, 1]

            if (y is not None) and (treatment is not None) and verbose:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                y_filt = y[mask]
                w = (treatment_filt == group).astype(int)

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = yhat_cs[group][mask][w == 0]
                yhat[w == 1] = yhat_ts[group][mask][w == 1]

                logger.info('Error metrics for group {}'.format(group))
                classification_metrics(y_filt, yhat, w)

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        for i, group in enumerate(self.t_groups):
            te[:, i] = yhat_ts[group] - yhat_cs[group]

        if not return_components:
            return te
        else:
            return te, yhat_cs, yhat_ts


class XGBTRegressor(BaseTRegressor):
    def __init__(self, ate_alpha=.05, control_name=0, *args, **kwargs):
        """Initialize a T-learner with two XGBoost models."""
        super().__init__(learner=XGBRegressor(*args, **kwargs),
                         ate_alpha=ate_alpha,
                         control_name=control_name)


class MLPTRegressor(BaseTRegressor):
    def __init__(self, ate_alpha=.05, control_name=0, *args, **kwargs):
        """Initialize a T-learner with two MLP models."""
        super().__init__(learner=MLPRegressor(*args, **kwargs),
                         ate_alpha=ate_alpha,
                         control_name=control_name)
