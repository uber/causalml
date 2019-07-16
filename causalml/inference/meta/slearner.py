from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
import statsmodels.api as sm


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
        X = sm.add_constant(X, prepend=False, has_constant='add')   # Append ones. The first column is for the treatment indicator.
        self.model = sm.OLS(y, X).fit(cov_type=self.cov_type)
        self.coefficients = self.model.params
        self.conf_ints = self.model.conf_int(alpha=self.alpha)

    def predict(self, X):
        X = sm.add_constant(X, prepend=False, has_constant='add')   # Append ones. The first column is for the treatment indicator.
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
        return '{}(model={})'.format(self.__class__.__name__, self.model.__repr__())

    def fit(self, X, treatment, y):
        """Fit the inference model

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
        """
        is_treatment = treatment != self.control_name
        w = is_treatment.astype(int)

        t_groups = np.unique(treatment[is_treatment])
        self._classes = {}
        self._classes[t_groups[0]] = 0 # this should be updated for multi-treatment case
        X = np.hstack((w.reshape((-1, 1)), X))
        self.model.fit(X, y)

    def predict(self, X, treatment, y=None):
        """Predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array, optional): an outcome vector

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        is_treatment = treatment != self.control_name
        w = is_treatment.astype(int)

        X = np.hstack((w.reshape((-1, 1)), X))

        X[:, 0] = 0    # set the treatment column to zero (the control group)
        yhat_c = self.model.predict(X)

        X[:, 0] = 1    # set the treatment column to one (the treatment group)
        yhat_t = self.model.predict(X)

        if y is not None:
            logger.info('RMSE (Control): {:.6f}'.format(np.sqrt(mse(y[~is_treatment], yhat_c[~is_treatment]))))
            logger.info(' MAE (Control): {:.6f}'.format(mae(y[~is_treatment], yhat_c[~is_treatment])))
            logger.info('RMSE (Treatment): {:.6f}'.format(np.sqrt(mse(y[is_treatment], yhat_t[is_treatment]))))
            logger.info(' MAE (Treatment): {:.6f}'.format(mae(y[is_treatment], yhat_t[is_treatment])))

        return (yhat_t - yhat_c).reshape(-1,1)

    def fit_predict(self, X, treatment, y, return_ci=False, n_bootstraps=1000, bootstrap_size=10000, verbose=False):
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
        raise NotImplementedError

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


class LRSLearner(BaseSLearner):
    def __init__(self, ate_alpha=.05, control_name=0):
        """Initialize an S-learner with a linear regression model.

        Args:
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        super(LRSLearner, self).__init__(StatsmodelsOLS(alpha=ate_alpha),
                                         ate_alpha,
                                         control_name)

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

        self.fit(X, w, y)
        te = self.model.coefficients[0]
        te_lb = self.model.conf_ints[0, 0]
        te_ub = self.model.conf_ints[0, 1]
        return te, te_lb, te_ub
