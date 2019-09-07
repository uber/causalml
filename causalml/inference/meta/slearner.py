from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from future.builtins import super
import logging
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
import statsmodels.api as sm
from copy import deepcopy

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
        if learner:
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
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
        """
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

    def predict(self, X, treatment=None, y=None, verbose=True):
        """Predict treatment effects.
        Args:
            X (np.matrix): a feature matrix
            treatment (np.array, optional): a treatment vector
            y (np.array, optional): an outcome vector
        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
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

        return te

    def fit_predict(self, X, treatment, y, return_ci=False, n_bootstraps=1000, bootstrap_size=10000, verbose=True):
        """Fit the inference model of the S learner and predict treatment effects.
        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
            return_ci (bool, optional): whether to return confidence intervals
            n_bootstraps (int, optional): number of bootstrap iterations
            bootstrap_size (int, optional): number of samples per bootstrap
            verbose (str, optional): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects. Output dim: [n_samples, n_treatment].
                If return_ci, returns CATE [n_samples, n_treatment], LB [n_samples, n_treatment],
                UB [n_samples, n_treatment]
        """
        self.fit(X, treatment, y)
        te = self.predict(X, treatment, y)

        if not return_ci:
            return te
        else:
            start = pd.datetime.today()
            self.t_groups_global = self.t_groups
            self._classes_global = self._classes
            self.models_global = deepcopy(self.models)
            te_bootstraps = np.zeros(shape=(X.shape[0], self.t_groups.shape[0], n_bootstraps))
            for i in range(n_bootstraps):
                te_b = self.bootstrap(X, treatment, y, size=bootstrap_size)
                te_bootstraps[:, :, i] = te_b
                if verbose and i % 10 == 0 and i > 0:
                    now = pd.datetime.today()
                    lapsed = (now-start).seconds
                    logger.info('{}/{} bootstraps completed. ({}s lapsed)'.format(i+1, n_bootstraps, lapsed))

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha/2)*100, axis=2)
            te_upper = np.percentile(te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=2)

            # set member variables back to global (currently last bootstrapped outcome)
            self.t_groups = self.t_groups_global
            self._classes = self._classes_global
            self.models = self.models_global

            return (te, te_lower, te_upper)

    def estimate_ate(self, X, treatment, y, return_ci=False, n_bootstraps=1000, bootstrap_size=10000, verbose=True):
        if return_ci:
            te, te_lb, te_ub = self.fit_predict(X, treatment, y, return_ci=True, n_bootstraps=n_bootstraps,
                                                bootstrap_size=bootstrap_size, verbose=verbose)

            ate = te.mean(axis=0)
            ate_lb = te_lb.mean(axis=0)
            ate_ub = te_ub.mean(axis=0)
            return ate, ate_lb, ate_ub

        else:
            te = self.fit_predict(X, treatment, y, return_ci=False, n_bootstraps=n_bootstraps,
                                  bootstrap_size=bootstrap_size, verbose=verbose)
            ate = te.mean(axis=0)
            return ate

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
            X (np.matrix): a feature matrix
            treatment (np.array, optional): a treatment vector
            y (np.array, optional): an outcome vector
        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
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
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
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
