from abc import ABCMeta, abstractmethod
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from tqdm import tqdm

from causalml.inference.meta.explainer import Explainer
from causalml.inference.meta.utils import check_p_conditions, convert_pd_to_np
from causalml.propensity import compute_propensity_score

logger = logging.getLogger("causalml")


def _fit_bootstrap_clone(learner_template, X, treatment, y, p, seed, bootstrap_size):
    """Module-level bootstrap helper for joblib pickling compatibility.

    Args:
        learner_template: an *unfitted* learner to clone as a template.
            Because BaseLearner now inherits BaseEstimator, ``clone(learner_template)``
            produces a clean unfitted copy via ``get_params``/``set_params``.
        X: feature matrix
        treatment: treatment vector
        y: outcome vector
        p: propensity scores or None
        seed (int): random seed for this bootstrap iteration
        bootstrap_size (int): number of samples to draw
    Returns:
        A fitted clone of learner_template trained on a bootstrap sample.
    """
    rng = np.random.RandomState(seed)
    idxs = rng.choice(np.arange(X.shape[0]), size=bootstrap_size)
    X_b = X[idxs]
    treatment_b = treatment[idxs]
    y_b = y[idxs]
    p_b = {group: _p[idxs] for group, _p in p.items()} if p is not None else None
    learner_b = clone(learner_template)
    learner_b.fit(X=X_b, treatment=treatment_b, y=y_b, p=p_b)
    return learner_b


class BaseLearner(BaseEstimator, metaclass=ABCMeta):
    """Base class for all causalml meta-learners.

    Inheriting ``sklearn.base.BaseEstimator`` gives every subclass:
    * ``get_params`` / ``set_params`` for free (requires verbatim ``__init__``
      argument storage — see scikit-learn conventions).
    * ``sklearn.base.clone`` support without ``safe=False``.
    * ``Pipeline`` / ``GridSearchCV`` compatibility.

    Subclass contract
    -----------------
    * ``__init__`` **must** store every argument verbatim as ``self.<param> = param``.
      No logic, no ``deepcopy``, no derived attributes.
    * All model construction and validation moves to ``fit()``.
    """

    @classmethod
    @abstractmethod
    def fit(self, X, treatment, y, p=None):
        pass

    @classmethod
    @abstractmethod
    def predict(
        self, X, treatment=None, y=None, p=None, return_components=False, verbose=True
    ):
        pass

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
        return self.predict(X, treatment, y, p, return_components, verbose)

    @classmethod
    @abstractmethod
    def estimate_ate(
        self,
        X,
        treatment,
        y,
        p=None,
        bootstrap_ci=False,
        n_bootstraps=1000,
        bootstrap_size=10000,
    ):
        pass

    def bootstrap(self, X, treatment, y, p=None, size=10000, rng=None):
        """Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population."""
        if rng is not None:
            idxs = rng.choice(np.arange(0, X.shape[0]), size=size)
        else:
            idxs = np.random.choice(np.arange(0, X.shape[0]), size=size)
        X_b = X[idxs]

        if p is not None:
            p_b = {group: _p[idxs] for group, _p in p.items()}
        else:
            p_b = None

        treatment_b = treatment[idxs]
        y_b = y[idxs]
        self.fit(X=X_b, treatment=treatment_b, y=y_b, p=p_b)
        return self.predict(X=X, p=p)

    def fit_bootstrap_ensemble(
        self,
        X,
        treatment,
        y,
        p=None,
        n_bootstraps=200,
        bootstrap_size=10000,
        random_state=None,
        n_jobs=1,
    ):
        """Train and store a bootstrap ensemble for post-fit CI estimation.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            p: propensity scores, passed through to fit() if provided
            n_bootstraps (int, optional): number of bootstrap iterations. Default: 200.
            bootstrap_size (int, optional): number of samples per bootstrap. Default: 10000.
            random_state (int, optional): random seed for reproducibility.
            n_jobs (int, optional): number of parallel jobs. -1 uses all cores. Default: 1.
        """
        unfitted_template = clone(self)

        rng = np.random.RandomState(random_state)
        seeds = rng.randint(0, np.iinfo(np.int32).max, size=n_bootstraps)
        logger.info("Storing bootstrap ensemble ({} iterations)".format(n_bootstraps))

        self.bootstrap_models_ = Parallel(n_jobs=n_jobs)(
            delayed(_fit_bootstrap_clone)(
                unfitted_template, X, treatment, y, p, s, bootstrap_size
            )
            for s in tqdm(seeds)
        )

    @staticmethod
    def _format_p(p, t_groups):
        """Format propensity scores into a dictionary of {treatment group: propensity scores}.

        Args:
            p (np.ndarray, pd.Series, or dict): propensity scores
            t_groups (list): treatment group names.

        Returns:
            dict of {treatment group: propensity scores}
        """
        check_p_conditions(p, t_groups)

        if isinstance(p, (np.ndarray, pd.Series)):
            treatment_name = t_groups[0]
            p = {treatment_name: convert_pd_to_np(p)}
        elif isinstance(p, dict):
            p = {
                treatment_name: convert_pd_to_np(_p) for treatment_name, _p in p.items()
            }

        return p

    def _set_propensity_models(self, X, treatment, y):
        """Set self.propensity and self.propensity_models."""
        logger.info("Generating propensity score")
        p = dict()
        p_model = dict()
        for group in self.t_groups:
            mask = (treatment == group) | (treatment == self.control_name)
            treatment_filt = treatment[mask]
            X_filt = X[mask]
            w_filt = (treatment_filt == group).astype(int)
            w = (treatment == group).astype(int)
            propensity_model = self.model_p if hasattr(self, "model_p") else None
            p[group], p_model[group] = compute_propensity_score(
                X=X_filt,
                treatment=w_filt,
                p_model=propensity_model,
                X_pred=X,
                treatment_pred=w,
            )
        self.propensity_model = p_model
        self.propensity = p

    def get_importance(
        self,
        X=None,
        tau=None,
        model_tau_feature=None,
        features=None,
        method="auto",
        normalize=True,
        test_size=0.3,
        random_state=None,
    ):
        explainer = Explainer(
            method=method,
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            classes=self._classes,
            normalize=normalize,
            test_size=test_size,
            random_state=random_state,
        )
        return explainer.get_importance()

    def get_shap_values(self, X=None, model_tau_feature=None, tau=None, features=None):
        explainer = Explainer(
            method="shapley",
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            classes=self._classes,
        )
        return explainer.get_shap_values()

    def plot_importance(
        self,
        X=None,
        tau=None,
        model_tau_feature=None,
        features=None,
        method="auto",
        normalize=True,
        test_size=0.3,
        random_state=None,
    ):
        explainer = Explainer(
            method=method,
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            classes=self._classes,
            normalize=normalize,
            test_size=test_size,
            random_state=random_state,
        )
        explainer.plot_importance()

    def plot_shap_values(
        self,
        X=None,
        tau=None,
        model_tau_feature=None,
        features=None,
        shap_dict=None,
        **kwargs,
    ):
        override_checks = shap_dict is not None
        explainer = Explainer(
            method="shapley",
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            override_checks=override_checks,
            classes=self._classes,
        )
        explainer.plot_shap_values(shap_dict=shap_dict, **kwargs)

    def plot_shap_dependence(
        self,
        treatment_group,
        feature_idx,
        X,
        tau,
        model_tau_feature=None,
        features=None,
        shap_dict=None,
        interaction_idx="auto",
        **kwargs,
    ):
        override_checks = False if shap_dict is None else True
        explainer = Explainer(
            method="shapley",
            control_name=self.control_name,
            X=X,
            tau=tau,
            model_tau=model_tau_feature,
            features=features,
            override_checks=override_checks,
            classes=self._classes,
        )
        explainer.plot_shap_dependence(
            treatment_group=treatment_group,
            feature_idx=feature_idx,
            shap_dict=shap_dict,
            interaction_idx=interaction_idx,
            **kwargs,
        )
