from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from copy import deepcopy
import logging
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.neural_network import MLPRegressor
from sklearn.utils.testing import ignore_warnings
from xgboost import XGBRegressor


logger = logging.getLogger('causalml')



class BaseTLearner(object):
    """A parent class for T-learner classes.

    An T-learner estimates treatment effects with two machine learning models.

    Details of T-learner are available at Kunzel et al. (2018) (https://arxiv.org/abs/1706.03461).

    """

    def __init__(self, learner=None, control_learner=None, treatment_learner=None, ate_alpha=.05,
                 control_name=0):
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
        is_treatment = treatment!=self.control_name
        w = is_treatment.astype(int)

        t_groups = np.unique(treatment[is_treatment])
        self._classes = {}
        self._classes[t_groups[0]] = 0 # this should be updated for multi-treatment case
        X = np.hstack((w.reshape((-1, 1)), X))

        logger.info('Training a control group model')
        self.model_c.fit(X[~is_treatment], y[~is_treatment])

        logger.info('Training a treatment group model')
        self.model_t.fit(X[is_treatment], y[is_treatment])

    def predict(self, X, treatment=None, y=None):
        """Predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array, optional): an optional outcome vector

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        is_treatment = treatment!=self.control_name
        w = is_treatment.astype(int)

        X = np.hstack((w.reshape((-1, 1)), X))

        yhat_c = self.model_c.predict(X)
        yhat_t = self.model_t.predict(X)

        if (y is not None) and (w is not None):
            is_treatment = w == 1
            logger.info('RMSE (Control): {:.6f}'.format(np.sqrt(mse(y[~is_treatment], yhat_c[~is_treatment]))))
            logger.info(' MAE (Control): {:.6f}'.format(mae(y[~is_treatment], yhat_c[~is_treatment])))
            logger.info('RMSE (Treatment): {:.6f}'.format(np.sqrt(mse(y[is_treatment], yhat_t[is_treatment]))))
            logger.info(' MAE (Treatment): {:.6f}'.format(mae(y[is_treatment], yhat_t[is_treatment])))

        return (yhat_t - yhat_c).reshape(-1,1)

    def fit_predict(self, X, treatment, y, return_ci=False, n_bootstraps=1000, bootstrap_size=10000, verbose=False):
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
            (numpy.ndarray): Predictions of treatment effects. Output dim: [n_samples, n_treatment]
                If return_ci, returns CATE [n_samples, n_treatment], LB [n_samples, n_treatment], UB [n_samples, n_treatment]
        """
        self.fit(X, treatment, y)
        te = self.predict(X, treatment, y)

        if not return_ci:
            return te
        else:
            start = pd.datetime.today()
            te_bootstraps = np.zeros(shape=(X.shape[0], n_bootstraps))
            for i in range(n_bootstraps):
                te_b = self.bootstrap(X, treatment, y, size=bootstrap_size)
                te_bootstraps[:,i] = np.ravel(te_b)
                if verbose:
                    now = pd.datetime.today()
                    lapsed = (now-start).seconds / 60
                    logger.info('{}/{} bootstraps completed. ({:.01f} min lapsed)'.format(i+1, n_bootstraps, lapsed))

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha/2)*100, axis=1)
            te_upper = np.percentile(te_bootstraps, (1 - self.ate_alpha/2)*100, axis=1)

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
        is_treatment = treatment != self.control_name
        w = is_treatment.astype(int)

        self.fit(X, treatment, y)

        X = np.hstack((w.reshape((-1, 1)), X))
        yhat_c = self.model_c.predict(X)
        yhat_t = self.model_t.predict(X)

        te = (yhat_t - yhat_c).mean()
        prob_treatment = float(sum(w))/X.shape[0]

        se = np.sqrt((
                (y[~is_treatment] - yhat_c[~is_treatment]).var()/(1-prob_treatment) +
                (y[is_treatment] - yhat_t[is_treatment]).var()/prob_treatment +
                (yhat_t - yhat_c).var()
            ) / y.shape[0])

        te_lb = te - se * norm.ppf(1 - self.ate_alpha / 2)
        te_ub = te + se * norm.ppf(1 - self.ate_alpha / 2)

        return te, te_lb, te_ub

    def bootstrap(self, X, treatment, y, size=10000):
        """
        Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population.
        """
        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]
        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b)
        te_b = self.predict(X=X, treatment=treatment, y=y)
        return te_b

class XGBTLearner(BaseTLearner):
    def __init__(self, ate_alpha=.05, control_name=0, *args, **kwargs):
        """Initialize a T-learner with two XGBoost models."""
        super(XGBTLearner, self).__init__(learner=XGBRegressor(*args, **kwargs),
                                          ate_alpha=ate_alpha,
                                          control_name=control_name)


class MLPTLearner(BaseTLearner):
    def __init__(self, ate_alpha=.05, control_name=0, *args, **kwargs):
        """Initialize a T-learner with two MLP models."""
        super(MLPTLearner, self).__init__(learner=MLPRegressor(*args, **kwargs),
                                          ate_alpha=ate_alpha,
                                          control_name=control_name)
