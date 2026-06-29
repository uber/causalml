from copy import deepcopy
import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

from causalml.inference.meta.base import BaseLearner
from causalml.inference.meta.utils import (
    check_treatment_vector,
    collect_if_lazy,
    filter_mask,
    n_rows,
    to_numpy,
    convert_pd_to_np,
)
from causalml.metrics import regression_metrics, classification_metrics

logger = logging.getLogger("causalml")


class BaseXLearner(BaseLearner):
    """A parent class for X-learner regressor classes.

    An X-learner estimates treatment effects with four machine learning models.

    Details of X-learner are available at `Kunzel et al. (2018) <https://arxiv.org/abs/1706.03461>`_.
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
        """
        Note: arguments are stored verbatim (scikit-learn convention) so that
        ``get_params`` / ``clone`` work correctly. Model construction is deferred to ``fit()``.
        Per the scikit-learn convention, ``__init__`` does not validate or raise —
        validation happens in ``fit()``.
        """
        # Store verbatim — no deepcopy, no logic (scikit-learn convention).
        self.learner = learner
        self.control_outcome_learner = control_outcome_learner
        self.treatment_outcome_learner = treatment_outcome_learner
        self.control_effect_learner = control_effect_learner
        self.treatment_effect_learner = treatment_effect_learner
        self.ate_alpha = ate_alpha
        self.control_name = control_name
        # Sentinel so estimate_ate(pretrain=True) raises a clean ValueError
        # ("no propensity score, please call fit() first") instead of AttributeError
        # when called before fit().
        self.propensity = {}

    def fit(self, X, treatment, y, p=None):
        """Fit the inference model.

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix.
                A pl.LazyFrame is collected once at the start of this method; the
                feature matrix is otherwise kept in its native format throughout.
            treatment (np.array, pd.Series, or pl.Series): a treatment vector
            y (np.array, pd.Series, or pl.Series): an outcome vector
            p (np.ndarray, pd.Series, pl.Series, or dict, optional): an array of propensity scores of float (0,1) in
                the single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
        """
        X = collect_if_lazy(X)
        if (self.learner is None) and (
            (self.control_outcome_learner is None)
            or (self.treatment_outcome_learner is None)
            or (self.control_effect_learner is None)
            or (self.treatment_effect_learner is None)
        ):
            raise ValueError(
                "Either `learner` or all four of `control_outcome_learner`, "
                "`treatment_outcome_learner`, `control_effect_learner`, and "
                "`treatment_effect_learner` must be specified."
            )
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        treatment_np = to_numpy(treatment)
        self.t_groups = np.unique(treatment_np[treatment_np != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=X, treatment=treatment_np, y=to_numpy(y))
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}

        # Resolve base models from stored constructor args (no templates needed).
        _control_outcome_learner = (
            self.control_outcome_learner
            if self.control_outcome_learner is not None
            else deepcopy(self.learner)
        )
        _treatment_outcome_learner = (
            self.treatment_outcome_learner
            if self.treatment_outcome_learner is not None
            else deepcopy(self.learner)
        )
        _control_effect_learner = (
            self.control_effect_learner
            if self.control_effect_learner is not None
            else deepcopy(self.learner)
        )
        _treatment_effect_learner = (
            self.treatment_effect_learner
            if self.treatment_effect_learner is not None
            else deepcopy(self.learner)
        )

        self.models_mu_t = {
            group: deepcopy(_treatment_outcome_learner) for group in self.t_groups
        }
        self.models_tau_c = {
            group: deepcopy(_control_effect_learner) for group in self.t_groups
        }
        self.models_tau_t = {
            group: deepcopy(_treatment_effect_learner) for group in self.t_groups
        }
        self.vars_t = {}

        # model_mu_c is trained on control only (identical across groups) — fit once.
        control_mask = treatment_np == self.control_name
        X_control = filter_mask(X, control_mask)
        y_control = to_numpy(filter_mask(y, control_mask))
        self.model_mu_c = deepcopy(self.model_mu_c)
        self.model_mu_c.fit(X_control, y_control)
        self.models_mu_c = {group: self.model_mu_c for group in self.t_groups}
        # var_c is a single scalar since control model is shared across groups
        self.var_c = (y_control - self.model_mu_c.predict(X_control)).var()
        # Keep vars_c dict for backward compat with estimate_ate
        # model_mu_c is trained on control data, identical for every treatment group.
        control_mask = treatment == self.control_name
        self.model_mu_c = deepcopy(_control_outcome_learner)
        self.model_mu_c.fit(X[control_mask], y[control_mask])
        self.models_mu_c = {group: self.model_mu_c for group in self.t_groups}

        y_control_pred = self.model_mu_c.predict(X[control_mask])
        self.var_c = (y[control_mask] - y_control_pred).var()
        self.vars_c = {group: self.var_c for group in self.t_groups}

        for group in self.t_groups:
            mask = (treatment_np == group) | (treatment_np == self.control_name)
            treatment_filt = filter_mask(treatment, mask)
            X_filt = filter_mask(X, mask)
            y_filt = filter_mask(y, mask)
            w = (to_numpy(treatment_filt) == group).astype(int)

            X_filt_c = filter_mask(X_filt, w == 0)
            X_filt_t = filter_mask(X_filt, w == 1)
            y_filt_np = to_numpy(y_filt)

            # Train treatment outcome model
            self.models_mu_t[group].fit(X_filt_t, filter_mask(y_filt, w == 1))

            var_t = (
                y_filt_np[w == 1] - self.models_mu_t[group].predict(X_filt_t)
            ).var()
            self.vars_t[group] = var_t

            # Train treatment effect models
            d_c = self.models_mu_t[group].predict(X_filt_c) - y_filt_np[w == 0]
            d_t = y_filt_np[w == 1] - self.model_mu_c.predict(X_filt_t)
            self.models_tau_c[group].fit(X_filt_c, d_c)
            self.models_tau_t[group].fit(X_filt_t, d_t)
        return self

    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        """Predict treatment effects.

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix.
                A pl.LazyFrame is collected once at the start of this method.
            treatment (np.array, pd.Series, or pl.Series, optional): a treatment vector
            y (np.array, pd.Series, or pl.Series, optional): an outcome vector
            p (np.ndarray, pd.Series, pl.Series, or dict, optional): an array of propensity scores of float (0,1) in
                the single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series, optional): a treatment vector
            y (np.array or pd.Series, optional): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): propensity scores
            return_components (bool, optional): whether to return outcome for treatment and control seperately
            verbose (bool, optional): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X = collect_if_lazy(X)

        if p is None:
            logger.info("Generating propensity score")
            p = {
                group: self.propensity_model[group].predict(X)
                for group in self.t_groups
            }
        else:
            p = self._format_p(p, self.t_groups)

        te = np.zeros((n_rows(X), self.t_groups.shape[0]))
        dhat_cs = {}
        dhat_ts = {}

        yhat_c_verbose = None
        if (y is not None) and (treatment is not None) and verbose:
            control_mask = treatment == self.control_name
            yhat_c_verbose = self.model_mu_c.predict(X[control_mask])

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
        """
        Fit the X-learner and predict treatment effects.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): propensity scores
            return_ci (bool): whether to return confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            return_components (bool, optional): whether to return outcome for treatment and control seperately
            verbose (str): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X = collect_if_lazy(X)
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
            treatment_np = to_numpy(treatment)
            y_np = to_numpy(y)

            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_mu_c_global = deepcopy(self.models_mu_c)
            models_mu_t_global = deepcopy(self.models_mu_t)
            models_tau_c_global = deepcopy(self.models_tau_c)
            models_tau_t_global = deepcopy(self.models_tau_t)
            te_bootstraps = np.zeros(
                shape=(n_rows(X), self.t_groups.shape[0], n_bootstraps)
            )

            logger.info("Bootstrap Confidence Intervals")
            for i in tqdm(range(n_bootstraps)):
                te_b = self.bootstrap(X, treatment_np, y_np, p, size=bootstrap_size)
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
        """Estimate the Average Treatment Effect (ATE).

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix
            treatment (np.array, pd.Series, or pl.Series): a treatment vector
            y (np.array, pd.Series, or pl.Series): an outcome vector
            p (np.ndarray, pd.Series, pl.Series, or dict, optional): an array of propensity scores of float (0,1) in
                the single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p (np.ndarray or pd.Series or dict, optional): propensity scores
            bootstrap_ci (bool): whether run bootstrap for confidence intervals
            n_bootstraps (int): number of bootstrap iterations
            bootstrap_size (int): number of samples per bootstrap
            pretrain (bool): whether a model has been fit, default False.
        Returns:
            The mean and confidence interval (LB, UB) of the ATE estimate.
        """
        X = collect_if_lazy(X)

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
            y_np = to_numpy(y)
            t_groups_global = self.t_groups
            _classes_global = self._classes
            models_mu_c_global = deepcopy(self.models_mu_c)
            models_mu_t_global = deepcopy(self.models_mu_t)
            models_tau_c_global = deepcopy(self.models_tau_c)
            models_tau_t_global = deepcopy(self.models_tau_t)

            logger.info("Bootstrap Confidence Intervals for ATE")
            ate_bootstraps = np.zeros(shape=(self.t_groups.shape[0], n_bootstraps))

            for n in tqdm(range(n_bootstraps)):
                cate_b = self.bootstrap(X, treatment_np, y_np, p, size=bootstrap_size)
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
    """A parent class for X-learner classifier classes."""

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
            outcome_learner (optional): a classifier for outcomes in both groups.
            effect_learner (optional): a regressor for treatment effects in both groups.
            control_outcome_learner (optional): a classifier for control outcomes.
            treatment_outcome_learner (optional): a classifier for treatment outcomes.
            control_effect_learner (optional): a regressor for control effects.
            treatment_effect_learner (optional): a regressor for treatment effects.
            ate_alpha (float, optional): confidence level alpha of the ATE estimate
            control_name (str or int, optional): name of control group
        """
        # Store all args verbatim (scikit-learn convention) — no resolution here.
        self.outcome_learner = outcome_learner
        self.effect_learner = effect_learner
        self.control_outcome_learner = control_outcome_learner
        self.treatment_outcome_learner = treatment_outcome_learner
        self.control_effect_learner = control_effect_learner
        self.treatment_effect_learner = treatment_effect_learner
        self.ate_alpha = ate_alpha
        self.control_name = control_name
        # Sentinel so estimate_ate(pretrain=True) raises cleanly before fit().
        self.propensity = {}

    def fit(self, X, treatment, y, p=None):
        """Fit the inference model.

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix.
                A pl.LazyFrame is collected once at the start of this method; the
                feature matrix is otherwise kept in its native format throughout.
            treatment (np.array, pd.Series, or pl.Series): a treatment vector
            y (np.array, pd.Series, or pl.Series): an outcome vector
            p (np.ndarray, pd.Series, pl.Series, or dict, optional): an array of propensity scores of float (0,1) in
                the single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
        """
        X = collect_if_lazy(X)
        if (self.outcome_learner is None) and (
            (self.control_outcome_learner is None)
            or (self.treatment_outcome_learner is None)
            or (self.control_effect_learner is None)
            or (self.treatment_effect_learner is None)
        ):
            raise ValueError(
                "Either `outcome_learner` and `effect_learner`, or all four specialized learners must be specified."
            )

        _control_outcome_learner = (
            deepcopy(self.outcome_learner)
            if self.control_outcome_learner is None
            else deepcopy(self.control_outcome_learner)
        )

        _treatment_outcome_learner = (
            deepcopy(self.outcome_learner)
            if self.treatment_outcome_learner is None
            else deepcopy(self.treatment_outcome_learner)
        )

        _control_effect_learner = (
            deepcopy(self.effect_learner)
            if self.control_effect_learner is None
            else deepcopy(self.control_effect_learner)
        )

        _treatment_effect_learner = (
            deepcopy(self.effect_learner)
            if self.treatment_effect_learner is None
            else deepcopy(self.treatment_effect_learner)
        )
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        check_treatment_vector(treatment, self.control_name)
        treatment_np = to_numpy(treatment)
        self.t_groups = np.unique(treatment_np[treatment_np != self.control_name])
        self.t_groups.sort()

        if p is None:
            self._set_propensity_models(X=X, treatment=treatment_np, y=to_numpy(y))
            p = self.propensity
        else:
            p = self._format_p(p, self.t_groups)

        self._classes = {group: i for i, group in enumerate(self.t_groups)}

        self.models_mu_t = {
            group: deepcopy(_treatment_outcome_learner) for group in self.t_groups
        }
        self.models_tau_c = {
            group: deepcopy(_control_effect_learner) for group in self.t_groups
        }
        self.models_tau_t = {
            group: deepcopy(_treatment_effect_learner) for group in self.t_groups
        }
        self.vars_t = {}

        # model_mu_c is trained on control only (identical across groups) — fit once.
        control_mask = treatment == self.control_name
        self.model_mu_c = deepcopy(_control_outcome_learner)
        self.model_mu_c.fit(X[control_mask], y[control_mask])
        self.models_mu_c = {group: self.model_mu_c for group in self.t_groups}

        y_control_pred = self.model_mu_c.predict_proba(X[control_mask])[:, 1]
        self.var_c = (y[control_mask] - y_control_pred).var()
        self.vars_c = {group: self.var_c for group in self.t_groups}

        for group in self.t_groups:
            mask = (treatment_np == group) | (treatment_np == self.control_name)
            treatment_filt = filter_mask(treatment, mask)
            X_filt = filter_mask(X, mask)
            y_filt = filter_mask(y, mask)
            w = (to_numpy(treatment_filt) == group).astype(int)

            X_filt_c = filter_mask(X_filt, w == 0)
            X_filt_t = filter_mask(X_filt, w == 1)
            y_filt_np = to_numpy(y_filt)

            # Train treatment outcome model
            self.models_mu_t[group].fit(X_filt_t, filter_mask(y_filt, w == 1))

            var_t = (
                y_filt_np[w == 1]
                - self.models_mu_t[group].predict_proba(X_filt_t)[:, 1]
            ).var()
            self.vars_t[group] = var_t

            # Train treatment models
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

        return self

    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        """Predict treatment effects.

        Args:
            X (np.matrix, np.array, pd.DataFrame, pl.DataFrame, or pl.LazyFrame): a feature matrix.
                A pl.LazyFrame is collected once at the start of this method.
            treatment (np.array, pd.Series, or pl.Series, optional): a treatment vector
            y (np.array, pd.Series, or pl.Series, optional): an outcome vector
            p (np.ndarray, pd.Series, pl.Series, or dict, optional): an array of propensity scores of float (0,1) in
                the single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1); if None will run ElasticNetPropensityModel() to generate the propensity scores.
            return_components (bool, optional): whether to return outcome for treatment and control seperately
            verbose (bool, optional): whether to output progress logs
        Returns:
            (numpy.ndarray): Predictions of treatment effects.
        """
        X = collect_if_lazy(X)
        """Predict treatment effects (classifier variant — uses predict_proba)."""
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        if p is None:
            logger.info("Generating propensity score")
            p = {
                group: self.propensity_model[group].predict(X)
                for group in self.t_groups
            }
        else:
            p = self._format_p(p, self.t_groups)

        te = np.zeros((n_rows(X), self.t_groups.shape[0]))
        dhat_cs = {}
        dhat_ts = {}

        yhat_c_verbose = None
        if (y is not None) and (treatment is not None) and verbose:
            control_mask = treatment == self.control_name
            yhat_c_verbose = self.model_mu_c.predict_proba(X[control_mask])[:, 1]

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
