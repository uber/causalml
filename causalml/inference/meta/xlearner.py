from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from copy import deepcopy
import logging
import pandas as pd
import numpy as np
from scipy.stats import norm


logger = logging.getLogger('causalml')


class BaseXLearner(object):
    """A parent class for X-learner classes.

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
            learner (optional): a model to estimate outcomes and treatment effects in both the control and treatment groups
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

        self.t_var = 0.0
        self.c_var = 0.0

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
        is_treatment = treatment != self.control_name
        w = is_treatment.astype(int)

        t_groups = np.unique(treatment[is_treatment])
        self._classes = {}
        self._classes[t_groups[0]] = 0 # this should be updated for multi-treatment case

        logger.info('Training the control group outcome model')
        self.model_mu_c.fit(X[~is_treatment], y[~is_treatment])
        self.c_var = (y[~is_treatment] - self.model_mu_c.predict(X[~is_treatment])).var()

        logger.info('Training the treatment group outcome model')
        self.model_mu_t.fit(X[is_treatment], y[is_treatment])
        self.t_var = (y[is_treatment] - self.model_mu_t.predict(X[is_treatment])).var()

        # Estimate treatment effects of the control and treatment groups
        d_c = self.model_mu_t.predict(X[~is_treatment]) - y[~is_treatment]
        d_t = y[is_treatment] - self.model_mu_c.predict(X[is_treatment])

        logger.info('Training the control group treatment model')
        self.model_tau_c.fit(X[~is_treatment], d_c)

        logger.info('Training the treatment group treatment model')
        self.model_tau_t.fit(X[is_treatment], d_t)

    def predict(self, X, p):
        """Predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            p (np.array): a propensity vector of float between 0 and 1

        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """

        dhat_c = self.model_tau_c.predict(X)
        dhat_t = self.model_tau_t.predict(X)

        return (p * dhat_c + (1 - p) * dhat_t).reshape(-1,1)

    def fit_predict(self, X, p, treatment, y, return_ci=False, n_bootstraps=1000, bootstrap_size=10000, verbose=False):
        """Fit the treatment effect and outcome models of the R learner and predict treatment effects.

        Args:
            X (np.matrix): a feature matrix
            p (np.array): a propensity vector between 0 and 1
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
        te = self.predict(X, p)

        if not return_ci:
            return te
        else:
            start = pd.datetime.today()
            te_bootstraps = np.zeros(shape=(X.shape[0], n_bootstraps))
            for i in range(n_bootstraps):
                te_b = self.bootstrap(X, p, treatment, y, size=bootstrap_size)
                te_bootstraps[:,i] = np.ravel(te_b)
                if verbose:
                    now = pd.datetime.today()
                    lapsed = (now-start).seconds / 60
                    logger.info('{}/{} bootstraps completed. ({:.01f} min lapsed)'.format(i+1, n_bootstraps, lapsed))

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha/2)*100, axis=1)
            te_upper = np.percentile(te_bootstraps, (1 - self.ate_alpha/2)*100, axis=1)

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
        is_treatment = treatment != self.control_name
        w = is_treatment.astype(int)

        self.fit(X, treatment, y)

        prob_treatment = float(sum(w))/X.shape[0]

        dhat_c = self.model_tau_c.predict(X)
        dhat_t = self.model_tau_t.predict(X)


        te = (p * dhat_c + (1 - p) * dhat_t).mean()

        # SE formula is based on the lower bound formula (7) from Imbens, Guido W., and Jeffrey M. Wooldridge. 2009.
        # "Recent Developments in the Econometrics of Program Evaluation." Journal of Economic Literature

        se = np.sqrt((
                self.t_var/prob_treatment + self.c_var/(1-prob_treatment)+
                (p * dhat_c + (1-p) * dhat_t).var()
            ) / X.shape[0])

        te_lb = te - se * norm.ppf(1 - self.ate_alpha / 2)
        te_ub = te + se * norm.ppf(1 - self.ate_alpha / 2)

        return te, te_lb, te_ub

    def bootstrap(self, X, p, treatment, y, size=10000):
        """
        Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population.
        """
        idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]
        p_b = p[idxs]
        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b)
        te_b = self.predict(X=X, p=p)
        return te_b
