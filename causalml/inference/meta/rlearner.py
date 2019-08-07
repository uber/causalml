from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from copy import deepcopy
import logging
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import cross_val_predict, KFold


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
        """Initialize a R-learner.

        Args:
            learner (optional): a model to estimate outcomes and treatment effects
            outcome_learner (optional): a model to estimate outcomes
            effect_learner (optional): a model to estimate treatment effects. It needs to take `sample_weight` as an
                input argument for `fit()`
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
            n_fold (int, optional): the number of cross validation folds for outcome_learner
            random_state (int or RandomState, optional): a seed (int) or random number generater (RandomState)
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

        self.cv = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)

        self.t_var = 0.0
        self.c_var = 0.0

    def __repr__(self):
        return ('{}(model_mu={},\n'
                '\tmodel_tau={})'.format(self.__class__.__name__,
                                         self.model_mu.__repr__(),
                                         self.model_tau.__repr__()))

    def fit_predict(self, X, p, treatment, y, return_ci=False,
                    n_bootstraps=1000, bootstrap_size=10000, verbose=False):
        """Fit the treatment effect and outcome models of the R learner and predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            p (np.array): a propensity vector between 0 and 1 treatment (np.array): a treatment vector
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
        self.fit(X, p, treatment, y)
        te = self.predict(X)

        if not return_ci:
            return te
        else:
            start = pd.datetime.today()
            te_bootstraps = np.zeros(shape=(X.shape[0], n_bootstraps))
            for i in range(n_bootstraps):
                te_b = self.bootstrap(X, p, treatment, y, size=bootstrap_size)
                te_bootstraps[:, i] = np.ravel(te_b)
                if verbose:
                    now = pd.datetime.today()
                    lapsed = (now-start).seconds / 60
                    logger.info('{}/{} bootstraps completed. ({:.01f} min ' 'lapsed)'.format(i+1, n_bootstraps, lapsed))

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha / 2) * 100, axis=1)
            te_upper = np.percentile(te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=1)

            return (te, te_lower, te_upper)

    def estimate_ate(self, X, p, treatment, y):
        """Estimate the Average Treatment Effect (ATE).

        Args:
            X (np.matrix): a feature matrix
            p (np.array): a propensity vector between 0 and 1
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector

        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        dhat = self.fit_predict(X, p, treatment, y)

        te = dhat.mean()
        prob_treatment = float(sum(treatment != self.control_name)) / X.shape[0]

        se = np.sqrt(self.t_var / prob_treatment + self.c_var / (1 - prob_treatment) + dhat.var()) / X.shape[0]

        te_lb = te - se * norm.ppf(1 - self.ate_alpha / 2)
        te_ub = te + se * norm.ppf(1 - self.ate_alpha / 2)

        return te, te_lb, te_ub

    def fit(self, X, p, treatment, y):
        """Fit the treatment effect and outcome models of the R learner.

        Args:
            X (np.matrix): a feature matrix
            p (np.array): a propensity vector between 0 and 1
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
        """
        is_treatment = treatment != self.control_name
        w = is_treatment.astype(int)

        t_groups = np.unique(treatment[is_treatment])
        self._classes = {}
        # this should be updated for multi-treatment case
        self._classes[t_groups[0]] = 0

        logger.info('generating out-of-fold CV outcome estimates with {}'.format(self.model_mu))
        yhat = cross_val_predict(self.model_mu, X, y, cv=self.cv)

        logger.info('training the treatment effect model, {} with R-loss'.format(self.model_tau))
        self.model_tau.fit(X, (y - yhat) / (w - p), sample_weight=(w - p) ** 2)

        self.t_var = (y[w == 1] - yhat[w == 1]).var()
        self.c_var = (y[w == 0] - yhat[w == 0]).var()

    def predict(self, X):
        """Predict treatment effects.

        Args:
            X (np.matrix): a feature matrix

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """

        dhat = self.model_tau.predict(X)

        return dhat.reshape(-1, 1)

    def bootstrap(self, X, p, treatment, y, size=10000):
        """Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population."""

        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]
        p_b = p[idxs]
        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, p=p_b, treatment=treatment_b, y=y_b)
        te_b = self.predict(X=X)
        return te_b
