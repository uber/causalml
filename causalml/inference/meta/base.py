from abc import ABCMeta, abstractmethod
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from tqdm import tqdm

from causalml.inference.meta.explainer import Explainer
from causalml.inference.meta.serialization import SerializableLearner
from causalml.inference.meta.utils import (
    check_p_conditions,
    filter_mask,
    filter_index,
    n_rows,
    to_numpy,
)
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
    idxs = rng.choice(np.arange(n_rows(X)), size=bootstrap_size)

    X_b = filter_index(X, idxs)
    treatment_b = treatment[idxs]
    y_b = y[idxs]
    p_b = {group: _p[idxs] for group, _p in p.items()} if p is not None else None
    learner_b = clone(learner_template)  # safe=True works now via get_params/set_params
    learner_b.fit(X=X_b, treatment=treatment_b, y=y_b, p=p_b)
    return learner_b


class BaseLearner(SerializableLearner, BaseEstimator, metaclass=ABCMeta):
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
    * ``fit()`` deepcopies the verbatim-stored arg before fitting, so ``self.learner``
      (and related params) remain unfitted across repeated ``fit()`` calls — this is
      the warm-start invariant that replaces the old ``_model_*_template`` mechanism.
    * ``__repr__`` is inherited from ``BaseEstimator`` and reflects constructor params.
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
        """Runs a single bootstrap. Fits on bootstrapped sample, then predicts on whole population.

        Args:
            X (np.matrix, np.array, pd.DataFrame, or pl.DataFrame): a feature matrix.
                Resampled natively via :func:`filter_index`, so X stays in its
                original format (numpy/pandas/polars) throughout.
            treatment (np.array): a treatment vector (numpy)
            y (np.array): an outcome vector (numpy)
            p (dict, optional): a dict of {treatment group: propensity scores (numpy)}
            size (int, optional): number of samples to draw with replacement
            rng (np.random.Generator, optional): random number generator for
                deterministic resampling
        Returns:
            (numpy.ndarray): Predictions of treatment effects on the full X
                from a model trained on the resampled subset.
        """
        if rng is not None:
            idxs = rng.choice(np.arange(0, n_rows(X)), size=size)
        else:
            idxs = np.random.choice(np.arange(0, n_rows(X)), size=size)
        X_b = filter_index(X, idxs)

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

        Fits n_bootstraps cloned copies of the entire learner on bootstrap samples
        and stores them in self.bootstrap_models_. Used by predict(return_ci=True)
        to compute percentile-based confidence intervals on new data without refitting.

        Because ``BaseLearner`` now inherits ``BaseEstimator``, ``clone(self)``
        produces a clean unfitted copy via ``get_params``/``set_params``. The
        warm-start invariant — that ``self.learner`` stays unfitted across calls —
        is maintained by each ``fit()`` deepcopying the verbatim-stored constructor
        arg before fitting it.

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
        # clone(self) is now a proper sklearn clone — unfitted and cheap.
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
            p (np.ndarray, pd.Series, pl.Series, or dict): propensity scores
            t_groups (list): treatment group names.

        Returns:
            dict of {treatment group: propensity scores (numpy.ndarray)}
        """
        check_p_conditions(p, t_groups)

        if isinstance(p, dict):
            p = {treatment_name: to_numpy(_p) for treatment_name, _p in p.items()}
        else:
            treatment_name = t_groups[0]
            p = {treatment_name: to_numpy(p)}

        return p

    def _set_propensity_models(self, X, treatment, y):
        """Set self.propensity and self.propensity_models.

        It trains propensity models for all treatment groups, save them in self.propensity_models, and
        save propensity scores in self.propensity in dictionaries with treatment groups as keys.

        It will use self.model_p if available to train propensity models. Otherwise, it will use a default
        PropensityModel (i.e. ElasticNetPropensityModel).

        Args:
            X (np.matrix, np.array, pd.DataFrame, or pl.DataFrame): a feature matrix.
                Kept in its native format; scikit-learn >= 1.6 accepts pandas
                and Polars DataFrames natively, so no conversion is performed.
            treatment (np.array, pd.Series, or pl.Series): a treatment vector
            y (np.array, pd.Series, or pl.Series): an outcome vector
        """
        logger.info("Generating propensity score")
        treatment_np = to_numpy(treatment)
        p = dict()
        p_model = dict()
        for group in self.t_groups:
            mask = (treatment_np == group) | (treatment_np == self.control_name)
            treatment_filt_np = treatment_np[mask]
            X_filt = filter_mask(X, mask)
            w_filt = (treatment_filt_np == group).astype(int)
            w = (treatment_np == group).astype(int)
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
        """
        Builds a model (using X to predict estimated/actual tau), and then calculates feature importances
        based on a specified method.

        Currently supported methods are:
            - auto (calculates importance based on estimator's default implementation of feature importance;
                    estimator must be tree-based)
                    Note: if none provided, it uses lightgbm's LGBMRegressor as estimator, and "gain" as
                    importance type
            - permutation (calculates importance based on mean decrease in accuracy when a feature column is permuted;
                           estimator can be any form)
        Hint: for permutation, downsample data for better performance especially if X.shape[1] is large

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (np.array): list/array of feature names. If None, an enumerated list will be used
            method (str): auto, permutation
            normalize (bool): normalize by sum of importances if method=auto (defaults to True)
            test_size (float/int): if float, represents the proportion of the dataset to include in the test split.
                                   If int, represents the absolute number of test samples (used for estimating
                                   permutation importance)
            random_state (int/RandomState instance/None): random state used in permutation importance estimation
        """
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
        """
        Builds a model (using X to predict estimated/actual tau), and then calculates shapley values.
        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used.
        """
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
        """
        Builds a model (using X to predict estimated/actual tau), and then plots feature importances
        based on a specified method.

        Currently supported methods are:
            - auto (calculates importance based on estimator's default implementation of feature importance;
                    estimator must be tree-based)
                    Note: if none provided, it uses lightgbm's LGBMRegressor as estimator, and "gain" as
                    importance type
            - permutation (calculates importance based on mean decrease in accuracy when a feature column is permuted;
                           estimator can be any form)
        Hint: for permutation, downsample data for better performance especially if X.shape[1] is large

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used
            method (str): auto, permutation
            normalize (bool): normalize by sum of importances if method=auto (defaults to True)
            test_size (float/int): if float, represents the proportion of the dataset to include in the test split.
                                   If int, represents the absolute number of test samples (used for estimating
                                   permutation importance)
            random_state (int/RandomState instance/None): random state used in permutation importance estimation
        """
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
        """
        Plots distribution of shapley values.

        If shapley values have been pre-computed, pass it through the shap_dict parameter.
        If shap_dict is not provided, this builds a new model (using X to predict estimated/actual tau),
        and then calculates shapley values.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix. Required if shap_dict is None.
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used.
            shap_dict (optional, dict): a dict of shapley value matrices. If None, shap_dict will be computed.
        """
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
        """
        Plots dependency of shapley values for a specified feature, colored by an interaction feature.

        If shapley values have been pre-computed, pass it through the shap_dict parameter.
        If shap_dict is not provided, this builds a new model (using X to predict estimated/actual tau),
        and then calculates shapley values.

        This plots the value of the feature on the x-axis and the SHAP value of the same feature
        on the y-axis. This shows how the model depends on the given feature, and is like a
        richer extension of the classical partial dependence plots. Vertical dispersion of the
        data points represents interaction effects.

        Args:
            treatment_group (str or int): name of treatment group to create dependency plot on
            feature_idx (str or int): feature index / name to create dependency plot on
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            model_tau_feature (sklearn/lightgbm/xgboost model object): an unfitted model object
            features (optional, np.array): list/array of feature names. If None, an enumerated list will be used.
            shap_dict (optional, dict): a dict of shapley value matrices. If None, shap_dict will be computed.
            interaction_idx (optional, str or int): feature index / name used in coloring scheme as interaction feature.
                If "auto" then shap.common.approximate_interactions is used to pick what seems to be the
                strongest interaction (note that to find to true strongest interaction you need to compute
                the SHAP interaction values).
        """
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
