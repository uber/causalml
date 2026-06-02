from copy import deepcopy
import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

from causalml.inference.meta.base import BaseLearner
from causalml.inference.meta.utils import (
    check_treatment_vector,
    filter_mask,
    to_numpy,
)
from causalml.metrics import regression_metrics, classification_metrics

logger = logging.getLogger("causalml")


class BaseXLearner(BaseLearner):
    """A parent class for X-learner regressor classes."""

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
        assert (learner is not None) or (
            (control_outcome_learner is not None)
            and (treatment_outcome_learner is not None)
            and (control_effect_learner is not None)
            and (treatment_effect_learner is not None)
        )

        self.model_mu_c = (
            deepcopy(learner)
            if control_outcome_learner is None
            else control_outcome_learner
        )
        self.model_mu_t = (
            deepcopy(learner)
            if treatment_outcome_learner is None
            else treatment_outcome_learner
        )
        self.model_tau_c = (
            deepcopy(learner)
            if control_effect_learner is None
            else control_effect_learner
        )
        self.model_tau_t = (
            deepcopy(learner)
            if treatment_effect_learner is None
            else treatment_effect_learner
        )

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
        check_treatment_vector(treatment, self.control_name)
        treatment_np = to_numpy(treatment)
        self.t_groups = np.unique(treatment_np[treatment_np != self.control_name])
        self.t_groups.sort()

        if p is None:
            # base.py does raw numpy indexing internally — convert at this boundary
            self._set_propensity_models(
                X=to_numpy(X), treatment=treatment_np, y=to_numpy(y)
            )
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
            mask = (treatment_np == group) | (treatment_np == self.control_name)
            treatment_filt = filter_mask(treatment, mask)
            X_filt = filter_mask(X, mask)
            y_filt = filter_mask(y, mask)
            w = (to_numpy(treatment_filt) == group).astype(int)

            self.models_mu_c[group].fit(
                filter_mask(X_filt, w == 0), filter_mask(y_filt, w == 0)
            )
            self.models_mu_t[group].fit(
                filter_mask(X_filt, w == 1), filter_mask(y_filt, w == 1)
            )

            y_filt_np = to_numpy(y_filt)
            X_filt_c = filter_mask(X_filt, w == 0)
            X_filt_t = filter_mask(X_filt, w == 1)

            var_c = (
                y_filt_np[w == 0] - self.models_mu_c[group].predict(X_filt_c)
            ).var()
            self.vars_c[group] = var_c
            var_t = (
                y_filt_np[w == 1] - self.models_mu_t[group].predict(X_filt_t)
            ).var()
            self.vars_t[group] = var_t

            d_c = self.models_mu_t[group].predict(X_filt_c) - y_filt_np[w == 0]
            d_t = y_filt_np[w == 1] - self.models_mu_c[group].predict(X_filt_t)
            self.models_tau_c[group].fit(X_filt_c, d_c)
            self.models_tau_t[group].fit(X_filt_t, d_t)

    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        if p is None:
            logger.info("Generating propensity score")
            p = {
                group: self.propensity_model[group].predict(X)
                for group in self.t_groups
            }
        else:
            p = self._format_p(p, self.t_groups)

        X_np = to_numpy(X)
        te = np.zeros((X_np.shape[0], self.t_groups.shape[0]))
        dhat_cs = {}
        dhat_ts = {}

        for i, group in enumerate(self.t_groups):
            dhat_cs[group] = self.models_tau_c[group].predict(X)
            dhat_ts[group] = self.models_tau_t[group].predict(X)

            _te = (p[group] * dhat_cs[group] + (1 - p[group]) * dhat_ts[group]).reshape(
                -1, 1
            )
            te[:, i] = np.ravel(_te)

            if (y is not None) and (treatment is not None) and verbose:
                treatment_np = to_numpy(treatment)
                mask = (treatment_np == group) | (treatment_np == self.control_name)
                treatment_filt_np = treatment_np[mask]
                X_filt = filter_mask(X, mask)
                y_filt = to_numpy(filter_mask(y, mask))
                w = (treatment_filt_np == group).astype(int)

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = self.models_mu_c[group].predict(
                    filter_mask(X_filt, w == 0)
                )
                yhat[w == 1] = self.models_mu_t[group].predict(
                    filter_mask(X_filt, w == 1)
                )

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
            X_np = to_numpy(X)
            treatment_np = to_numpy(treatment)
            y_np = to_numpy(y)

            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_mu_c_global = deepcopy(self.models_mu_c)
            models_mu_t_global = deepcopy(self.models_mu_t)
            models_tau_c_global = deepcopy(self.models_tau_c)
            models_tau_t_global = deepcopy(self.models_tau_t)
            te_bootstraps = np.zeros(
                shape=(X_np.shape[0], self.t_groups.shape[0], n_bootstraps)
            )

            logger.info("Bootstrap Confidence Intervals")
            for i in tqdm(range(n_bootstraps)):
                te_b = self.bootstrap(X_np, treatment_np, y_np, p, size=bootstrap_size)
                te_bootstraps[:, :, i] = te_b

            te_lower = np.percentile(te_bootstraps, (self.ate_alpha / 2) * 100, axis=2)
            te_upper = np.percentile(
                te_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=2
            )

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
        if pretrain:
            if p is None:
                if not self.propensity:
                    raise ValueError("no propensity score, please call fit() first")
                te, dhat_cs, dhat_ts = self.predict(
                    X, treatment, y, p=self.propensity, return_components=True
                )
            else:
                p = self._format_p(p, self.t_groups)
                te, dhat_cs, dhat_ts = self.predict(
                    X, treatment, y, p=p, return_components=True
                )
        else:
            te, dhat_cs, dhat_ts = self.fit_predict(
                X, treatment, y, p, return_components=True
            )

        treatment_np = to_numpy(treatment)

        if p is None:
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        ate = np.zeros(self.t_groups.shape[0])
        ate_lb = np.zeros(self.t_groups.shape[0])
        ate_ub = np.zeros(self.t_groups.shape[0])

        for i, group in enumerate(self.t_groups):
            _ate = te[:, i].mean()

            mask = (treatment_np == group) | (treatment_np == self.control_name)
            treatment_filt = treatment_np[mask]
            w = (treatment_filt == group).astype(int)
            prob_treatment = float(sum(w)) / w.shape[0]

            dhat_c = dhat_cs[group][mask]
            dhat_t = dhat_ts[group][mask]
            p_filt = p[group][mask]

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
            X_np = to_numpy(X)
            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_mu_c_global = deepcopy(self.models_mu_c)
            models_mu_t_global = deepcopy(self.models_mu_t)
            models_tau_c_global = deepcopy(self.models_tau_c)
            models_tau_t_global = deepcopy(self.models_tau_t)

            logger.info("Bootstrap Confidence Intervals for ATE")
            ate_bootstraps = np.zeros(shape=(self.t_groups.shape[0], n_bootstraps))

            for n in tqdm(range(n_bootstraps)):
                cate_b = self.bootstrap(
                    X_np, treatment_np, to_numpy(y), p, size=bootstrap_size
                )
                ate_bootstraps[:, n] = cate_b.mean(axis=0)

            ate_lower = np.percentile(
                ate_bootstraps, (self.ate_alpha / 2) * 100, axis=1
            )
            ate_upper = np.percentile(
                ate_bootstraps, (1 - self.ate_alpha / 2) * 100, axis=1
            )

            self.t_groups = t_groups_global
            self._classes = _classes_global
            self.models_mu_c = deepcopy(models_mu_c_global)
            self.models_mu_t = deepcopy(models_mu_t_global)
            self.models_tau_c = deepcopy(models_tau_c_global)
            self.models_tau_t = deepcopy(models_tau_t_global)
            return ate, ate_lower, ate_upper


class BaseXRegressor(BaseXLearner):
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
        check_treatment_vector(treatment, self.control_name)
        treatment_np = to_numpy(treatment)
        self.t_groups = np.unique(treatment_np[treatment_np != self.control_name])
        self.t_groups.sort()

        if p is None:
            # base.py does raw numpy indexing internally — convert at this boundary
            self._set_propensity_models(
                X=to_numpy(X), treatment=treatment_np, y=to_numpy(y)
            )
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
            mask = (treatment_np == group) | (treatment_np == self.control_name)
            treatment_filt = filter_mask(treatment, mask)
            X_filt = filter_mask(X, mask)
            y_filt = filter_mask(y, mask)
            w = (to_numpy(treatment_filt) == group).astype(int)

            self.models_mu_c[group].fit(
                filter_mask(X_filt, w == 0), filter_mask(y_filt, w == 0)
            )
            self.models_mu_t[group].fit(
                filter_mask(X_filt, w == 1), filter_mask(y_filt, w == 1)
            )

            y_filt_np = to_numpy(y_filt)
            X_filt_c = filter_mask(X_filt, w == 0)
            X_filt_t = filter_mask(X_filt, w == 1)

            var_c = (
                y_filt_np[w == 0]
                - self.models_mu_c[group].predict_proba(X_filt_c)[:, 1]
            ).var()
            self.vars_c[group] = var_c
            var_t = (
                y_filt_np[w == 1]
                - self.models_mu_t[group].predict_proba(X_filt_t)[:, 1]
            ).var()
            self.vars_t[group] = var_t

            d_c = (
                self.models_mu_t[group].predict_proba(X_filt_c)[:, 1]
                - y_filt_np[w == 0]
            )
            d_t = (
                y_filt_np[w == 1]
                - self.models_mu_c[group].predict_proba(X_filt_t)[:, 1]
            )
            self.models_tau_c[group].fit(X_filt_c, d_c)
            self.models_tau_t[group].fit(X_filt_t, d_t)

    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        if p is None:
            logger.info("Generating propensity score")
            p = {
                group: self.propensity_model[group].predict(X)
                for group in self.t_groups
            }
        else:
            p = self._format_p(p, self.t_groups)

        X_np = to_numpy(X)
        te = np.zeros((X_np.shape[0], self.t_groups.shape[0]))
        dhat_cs = {}
        dhat_ts = {}

        for i, group in enumerate(self.t_groups):
            dhat_cs[group] = self.models_tau_c[group].predict(X)
            dhat_ts[group] = self.models_tau_t[group].predict(X)

            _te = (p[group] * dhat_cs[group] + (1 - p[group]) * dhat_ts[group]).reshape(
                -1, 1
            )
            te[:, i] = np.ravel(_te)

            if (y is not None) and (treatment is not None) and verbose:
                treatment_np = to_numpy(treatment)
                mask = (treatment_np == group) | (treatment_np == self.control_name)
                treatment_filt_np = treatment_np[mask]
                X_filt = filter_mask(X, mask)
                y_filt = to_numpy(filter_mask(y, mask))
                w = (treatment_filt_np == group).astype(int)

                yhat = np.zeros_like(y_filt, dtype=float)
                yhat[w == 0] = self.models_mu_c[group].predict_proba(
                    filter_mask(X_filt, w == 0)
                )[:, 1]
                yhat[w == 1] = self.models_mu_t[group].predict_proba(
                    filter_mask(X_filt, w == 1)
                )[:, 1]

                logger.info("Error metrics for group {}".format(group))
                classification_metrics(y_filt, yhat, w)

        if not return_components:
            return te
        else:
            return te, dhat_cs, dhat_ts
