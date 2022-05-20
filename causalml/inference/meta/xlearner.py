from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm

from causalml.inference.meta.base import BaseLearner
from causalml.inference.meta.utils import (
    check_treatment_vector,
    check_p_conditions,
    convert_pd_to_np,
)
from causalml.inference.meta.explainer import Explainer
from causalml.metrics import regression_metrics, classification_metrics
from causalml.propensity import compute_propensity_score

logger = logging.getLogger("causalml")


class BaseXLearner(BaseLearner):
    """A parent class for X-learner regressor classes.

    An X-learner estimates treatment effects with four machine learning models.

    Details of X-learner are available at Kunzel et al. (2018) (https://arxiv.org/abs/1706.03461).
    """

    def __init__(
        self,
        learner=None,
        control_outcome_learner=None,
        treatment_outcome_learner=None,
        control_effect_learner=None,
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
            control_effect_learner (optional): a model to estimate treatment effects in the control group
            treatment_effect_learner (optional): a model to estimate treatment effects in the treatment group
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        assert (learner is not None) or (
            (control_outcome_learner is not None)
            and (treatment_outcome_learner is not None)
            and (control_effect_learner is not None)
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

        self.propensity = None
        self.propensity_model = None

    def __repr__(self):
        return (
            "{}(control_outcome_learner={},\n"
            "\ttreatment_outcome_learner={},\n"
            "\tcontrol_effect_learner={},\n"
            "\ttreatment_effect_learner={})".format(
                self.__class__.__name__,
                self.model_mu_c.__repr__(),
                self.model_mu_t.__repr__(),
                self.model_tau_c.__repr__(),
                self.model_tau_t.__repr__(),
            )
        )

    def fit(self, X, treatment, y, p=None):
        """Fit the inference model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=X, treatment=treatment, y=y)
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_mu_c = {group: deepcopy(self.model_mu_c) for group in self.t_groups}
        self.models_mu_t = {group: deepcopy(self.model_mu_t) for group in self.t_groups}
        self.models_tau_c = {
            group: deepcopy(self.model_tau_c) for group in self.t_groups
        }
        self.models_tau_t = {
            group: deepcopy(self.model_tau_t) for group in self.t_groups
        }
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
            var_c = (
                y_filt[w == 0] - self.models_mu_c[group].predict(X_filt[w == 0])
            ).var()
            self.vars_c[group] = var_c
            var_t = (
                y_filt[w == 1] - self.models_mu_t[group].predict(X_filt[w == 1])
            ).var()
            self.vars_t[group] = var_t

            # Train treatment models
            d_c = self.models_mu_t[group].predict(X_filt[w == 0]) - y_filt[w == 0]
            d_t = y_filt[w == 1] - self.models_mu_c[group].predict(X_filt[w == 1])
            self.models_tau_c[group].fit(X_filt[w == 0], d_c)
            self.models_tau_t[group].fit(X_filt[w == 1], d_t)

    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        """Predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series, optional): a treatment vector
            y (np.array or pd.Series, optional): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            return_components (bool, optional): whether to return outcome for treatment and control seperately
            verbose (bool, optional): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        if p is None:
            logger.info("Generating propensity score")
            p = dict()
            for group in self.t_groups:
                p_model = self.propensity_model[group]
                p[group] = p_model.predict(X)
        else:
            p = self._format_p(p, self.t_groups)

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        dhat_cs = {}
        dhat_ts = {}

        for i, group in enumerate(self.t_groups):
            model_tau_c = self.models_tau_c[group]
            model_tau_t = self.models_tau_t[group]
            dhat_cs[group] = model_tau_c.predict(X)
            dhat_ts[group] = model_tau_t.predict(X)

            _te = (p[group] * dhat_cs[group] + (1 - p[group]) * dhat_ts[group]).reshape(
                -1, 1
            )
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

                logger.info("Error metrics for group {}".format(group))
                regression_metrics(y_filt, yhat, w)

        if not return_components:
            return te
        else:
            return te, dhat_cs, dhat_ts

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
        Returns:
            (numpy.ndarray): Predictions of treatment effects. Output dim: [n_samples, n_treatment]
                If return_ci, returns CATE [n_samples, n_treatment], LB [n_samples, n_treatment],
                UB [n_samples, n_treatment]
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        self.fit(X, treatment, y, p)

        if p is None:
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        te = self.predict(
            X, treatment=treatment, y=y, p=p, return_components=return_components
        )

        if not return_ci:
            return te
        else:
            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_mu_c_global = deepcopy(self.models_mu_c)
            models_mu_t_global = deepcopy(self.models_mu_t)
            models_tau_c_global = deepcopy(self.models_tau_c)
            models_tau_t_global = deepcopy(self.models_tau_t)
            te_bootstraps = np.zeros(
                shape=(X.shape[0], self.t_groups.shape[0], n_bootstraps)
            )

            logger.info("Bootstrap Confidence Intervals")
            for i in tqdm(range(n_bootstraps)):
                te_b = self.bootstrap(X, treatment, y, p, size=bootstrap_size)
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
            self.models_tau_c = deepcopy(models_tau_c_global)
            self.models_tau_t = deepcopy(models_tau_t_global)

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
        pretrain=False,
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
            pretrain (bool): whether a model has been fit, default False.
        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        if pretrain and p is None:
            # when p is null, use pretrain propensity score
            if not self.propensity:
                raise ValueError("no propensity score, please call fit() first")
            te, dhat_cs, dhat_ts = self.predict(
                X, treatment, y, p=self.propensity, return_components=True
            )
        else:
            te, dhat_cs, dhat_ts = self.fit_predict(
                X, treatment, y, p, return_components=True
            )
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        if p is None:
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

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
            se = np.sqrt(
                (
                    self.vars_t[group] / prob_treatment
                    + self.vars_c[group] / (1 - prob_treatment)
                    + (p_filt * dhat_c + (1 - p_filt) * dhat_t).var()
                )
                / w.shape[0]
            )

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
            models_tau_c_global = deepcopy(self.models_tau_c)
            models_tau_t_global = deepcopy(self.models_tau_t)

            logger.info("Bootstrap Confidence Intervals for ATE")
            ate_bootstraps = np.zeros(shape=(self.t_groups.shape[0], n_bootstraps))

            for n in tqdm(range(n_bootstraps)):
                cate_b = self.bootstrap(X, treatment, y, p, size=bootstrap_size)
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
            self.models_tau_c = deepcopy(models_tau_c_global)
            self.models_tau_t = deepcopy(models_tau_t_global)
            return ate, ate_lower, ate_upper


class BaseXRegressor(BaseXLearner):
    """
    A parent class for X-learner regressor classes.
    """

    def __init__(
        self,
        learner=None,
        control_outcome_learner=None,
        treatment_outcome_learner=None,
        control_effect_learner=None,
        treatment_effect_learner=None,
        ate_alpha=0.05,
        control_name=0,
    ):
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
            control_name=control_name,
        )


class BaseXClassifier(BaseXLearner):
    """
    A parent class for X-learner classifier classes.
    """

    def __init__(
        self,
        outcome_learner=None,
        effect_learner=None,
        control_outcome_learner=None,
        treatment_outcome_learner=None,
        control_effect_learner=None,
        treatment_effect_learner=None,
        ate_alpha=0.05,
        control_name=0,
    ):
        """Initialize an X-learner classifier.

        Args:
            outcome_learner (optional): a model to estimate outcomes in both the control and treatment groups.
                Should be a classifier.
            effect_learner (optional): a model to estimate treatment effects in both the control and treatment groups.
                Should be a regressor.
            control_outcome_learner (optional): a model to estimate outcomes in the control group.
                Should be a classifier.
            treatment_outcome_learner (optional): a model to estimate outcomes in the treatment group.
                Should be a classifier.
            control_effect_learner (optional): a model to estimate treatment effects in the control group.
                Should be a regressor.
            treatment_effect_learner (optional): a model to estimate treatment effects in the treatment group
                Should be a regressor.
            ate_alpha (float, optional): the confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        if outcome_learner is not None:
            control_outcome_learner = outcome_learner
            treatment_outcome_learner = outcome_learner
        if effect_learner is not None:
            control_effect_learner = effect_learner
            treatment_effect_learner = effect_learner

        super().__init__(
            learner=None,
            control_outcome_learner=control_outcome_learner,
            treatment_outcome_learner=treatment_outcome_learner,
            control_effect_learner=control_effect_learner,
            treatment_effect_learner=treatment_effect_learner,
            ate_alpha=ate_alpha,
            control_name=control_name,
        )

        if (
            (control_outcome_learner is None) or (treatment_outcome_learner is None)
        ) and ((control_effect_learner is None) or (treatment_effect_learner is None)):
            raise ValueError(
                "Either the outcome learner or the effect learner pair must be specified."
            )

    def fit(self, X, treatment, y, p=None):
        """Fit the inference model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        self.t_groups = np.unique(treatment[treatment != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=X, treatment=treatment, y=y)
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}
        self.models_mu_c = {group: deepcopy(self.model_mu_c) for group in self.t_groups}
        self.models_mu_t = {group: deepcopy(self.model_mu_t) for group in self.t_groups}
        self.models_tau_c = {
            group: deepcopy(self.model_tau_c) for group in self.t_groups
        }
        self.models_tau_t = {
            group: deepcopy(self.model_tau_t) for group in self.t_groups
        }
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
            var_c = (
                y_filt[w == 0]
                - self.models_mu_c[group].predict_proba(X_filt[w == 0])[:, 1]
            ).var()
            self.vars_c[group] = var_c
            var_t = (
                y_filt[w == 1]
                - self.models_mu_t[group].predict_proba(X_filt[w == 1])[:, 1]
            ).var()
            self.vars_t[group] = var_t

            # Train treatment models
            d_c = (
                self.models_mu_t[group].predict_proba(X_filt[w == 0])[:, 1]
                - y_filt[w == 0]
            )
            d_t = (
                y_filt[w == 1]
                - self.models_mu_c[group].predict_proba(X_filt[w == 1])[:, 1]
            )
            self.models_tau_c[group].fit(X_filt[w == 0], d_c)
            self.models_tau_t[group].fit(X_filt[w == 1], d_t)

    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        """Predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series, optional): a treatment vector
            y (np.array or pd.Series, optional): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            return_components (bool, optional): whether to return outcome for treatment and control seperately
            return_p_score (bool, optional): whether to return propensity score
            verbose (bool, optional): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        if p is None:
            logger.info("Generating propensity score")
            p = dict()
            for group in self.t_groups:
                p_model = self.propensity_model[group]
                p[group] = p_model.predict(X)
        else:
            p = self._format_p(p, self.t_groups)

        te = np.zeros((X.shape[0], self.t_groups.shape[0]))
        dhat_cs = {}
        dhat_ts = {}

        for i, group in enumerate(self.t_groups):
            model_tau_c = self.models_tau_c[group]
            model_tau_t = self.models_tau_t[group]
            dhat_cs[group] = model_tau_c.predict(X)
            dhat_ts[group] = model_tau_t.predict(X)

            _te = (p[group] * dhat_cs[group] + (1 - p[group]) * dhat_ts[group]).reshape(
                -1, 1
            )
            te[:, i] = np.ravel(_te)

            if (y is not None) and (treatment is not None) and verbose:
                mask = (treatment == group) | (treatment == self.control_name)
                treatment_filt = treatment[mask]
                X_filt = X[mask]
                y_filt = y[mask]
                w = (treatment_filt == group).astype(int)

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = self.models_mu_c[group].predict_proba(X_filt[w == 0])[
                    :, 1
                ]
                yhat[w == 1] = self.models_mu_t[group].predict_proba(X_filt[w == 1])[
                    :, 1
                ]

                logger.info("Error metrics for group {}".format(group))
                classification_metrics(y_filt, yhat, w)

        if not return_components:
            return te
        else:
            return te, dhat_cs, dhat_ts
