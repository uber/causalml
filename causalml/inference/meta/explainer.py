import pandas as pd
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from copy import deepcopy

from causalml.inference.meta.utils import convert_pd_to_np

VALID_METHODS = ("auto", "permutation", "shapley")


class Explainer:
    def __init__(
        self,
        method,
        control_name,
        X,
        tau,
        classes,
        model_tau=None,
        features=None,
        normalize=True,
        test_size=0.3,
        random_state=None,
        override_checks=False,
        r_learners=None,
    ):
        """
        The Explainer class handles all feature explanation/interpretation functions, including plotting
        feature importances, shapley value distributions, and shapley value dependency plots.

        Currently supported methods are:
            - auto (calculates importance based on estimator's default implementation of feature importance;
                    estimator must be tree-based)
                    Note: if none provided, it uses lightgbm's LGBMRegressor as estimator, and "gain" as
                    importance type
            - permutation (calculates importance based on mean decrease in accuracy when a feature column is permuted;
                           estimator can be any form)
            - shapley (calculates shapley values; estimator must be tree-based)
        Hint: for permutation, downsample data for better performance especially if X.shape[1] is large

        Args:
            method (str): auto, permutation, shapley
            control_name (str/int/float): name of control group
            X (np.matrix): a feature matrix
            tau (np.array): a treatment effect vector (estimated/actual)
            classes (dict): a mapping of treatment names to indices (used for indexing tau array)
            model_tau (sklearn/lightgbm/xgboost model object): a model object
            features (np.array): list/array of feature names. If None, an enumerated list will be used.
            normalize (bool): normalize by sum of importances if method=auto (defaults to True)
            test_size (float/int): if float, represents the proportion of the dataset to include in the test split.
                                   If int, represents the absolute number of test samples (used for estimating
                                   permutation importance)
            random_state (int/RandomState instance/None): random state used in permutation importance estimation
            override_checks (bool): overrides self.check_conditions (e.g. if importance/shapley values are pre-computed)
            r_learners (dict): a mapping of treatment group to fitted R Learners
        """
        self.method = method
        self.control_name = control_name
        self.X = convert_pd_to_np(X)
        self.tau = convert_pd_to_np(tau)
        if self.tau is not None and self.tau.ndim == 1:
            self.tau = self.tau.reshape(-1, 1)
        self.classes = classes
        self.model_tau = (
            LGBMRegressor(importance_type="gain") if model_tau is None else model_tau
        )
        self.features = features
        self.normalize = normalize
        self.test_size = test_size
        self.random_state = random_state
        self.override_checks = override_checks
        self.r_learners = r_learners

        if not self.override_checks:
            self.check_conditions()
            self.create_feature_names()
            self.build_new_tau_models()

    def check_conditions(self):
        """
        Checks for multiple conditions:
            - method is valid
            - X, tau, and classes are specified
            - model_tau has feature_importances_ attribute after fitting
        """
        assert self.method in VALID_METHODS, "Current supported methods: {}".format(
            ", ".join(VALID_METHODS)
        )

        assert all(
            obj is not None for obj in (self.X, self.tau, self.classes)
        ), "X, tau, and classes must be provided."

        model_test = deepcopy(self.model_tau)
        model_test.fit(
            [[0], [1]], [0, 1]
        )  # Fit w/ dummy data to check for feature_importances_ below
        assert hasattr(
            model_test, "feature_importances_"
        ), "model_tau must have the feature_importances_ method (after fitting)"

    def create_feature_names(self):
        """
        Creates feature names (simple enumerated list) if not provided in __init__.
        """
        if self.features is None:
            num_features = self.X.shape[1]
            self.features = ["Feature_{:03d}".format(i) for i in range(num_features)]

    def build_new_tau_models(self):
        """
        Builds tau models (using X to predict estimated/actual tau) for each treatment group.
        """
        if self.method in ("permutation"):
            self.X_train, self.X_test, self.tau_train, self.tau_test = train_test_split(
                self.X,
                self.tau,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        else:
            self.X_train, self.tau_train = self.X, self.tau

        if self.r_learners is not None:
            self.models_tau = deepcopy(self.r_learners)
        else:
            self.models_tau = {
                group: deepcopy(self.model_tau) for group in self.classes
            }
            for group, idx in self.classes.items():
                self.models_tau[group].fit(self.X_train, self.tau_train[:, idx])

    def get_importance(self):
        """
        Calculates feature importances for each treatment group, based on specified method in __init__.
        """
        importance_catalog = {
            "auto": self.default_importance,
            "permutation": self.perm_importance,
        }
        importance_dict = importance_catalog[self.method]()

        importance_dict = {
            group: pd.Series(array, index=self.features).sort_values(ascending=False)
            for group, array in importance_dict.items()
        }
        return importance_dict

    def default_importance(self):
        """
        Calculates feature importances for each treatment group, based on the model_tau's default implementation.
        """
        importance_dict = {}
        if self.r_learners is not None:
            self.models_tau = deepcopy(self.r_learners)
        for group, idx in self.classes.items():
            importance_dict[group] = self.models_tau[group].feature_importances_
            if self.normalize:
                importance_dict[group] = (
                    importance_dict[group] / importance_dict[group].sum()
                )

        return importance_dict

    def perm_importance(self):
        """
        Calculates feature importances for each treatment group, based on the permutation method.
        """
        importance_dict = {}
        if self.r_learners is not None:
            self.models_tau = deepcopy(self.r_learners)
            self.X_test, self.tau_test = self.X, self.tau
        for group, idx in self.classes.items():
            perm_estimator = self.models_tau[group]
            importance_dict[group] = permutation_importance(
                estimator=perm_estimator,
                X=self.X_test,
                y=self.tau_test[:, idx],
                random_state=self.random_state,
            ).importances_mean

        return importance_dict

    def get_shap_values(self):
        """
        Calculates shapley values for each treatment group.
        """
        shap_dict = {}
        for group, mod in self.models_tau.items():
            explainer = shap.TreeExplainer(mod)
            if self.r_learners is not None:
                explainer.model.original_model.params["objective"] = (
                    None  # hacky way of running shap without error
                )
            shap_values = explainer.shap_values(self.X)
            shap_dict[group] = shap_values

        return shap_dict

    def plot_importance(self, importance_dict=None, title_prefix="", figsize=(12, 8)):
        """
        Calculates and plots feature importances for each treatment group, based on specified method in __init__.
        Skips the calculation part if importance_dict is given.
        Args:
            importance_dict (optional, dict): a dict of feature importance matrics. If None, importance_dict will be
                computed.
            title_prefix (optional, str): a prefix to the title of the plot.
            figsize (optional, tuple): the size of the figure.
        """
        if importance_dict is None:
            importance_dict = self.get_importance()
        for group, series in importance_dict.items():
            plt.figure()
            series.sort_values().plot(kind="barh", figsize=figsize)
            title = group
            if title_prefix != "":
                title = "{} - {}".format(title_prefix, title)
            plt.title(title)

    def plot_shap_values(self, shap_dict=None, **kwargs):
        """
        Calculates and plots the distribution of shapley values of each feature, for each treatment group.
        Skips the calculation part if shap_dict is given.

        Args:
            shap_dict (optional, dict): a dict of shapley value matrics. If None, shap_dict will be computed.
        """
        if shap_dict is None:
            shap_dict = self.get_shap_values()

        for group, values in shap_dict.items():
            plt.title(group)
            shap.summary_plot(
                values, features=self.X, feature_names=self.features, **kwargs
            )

    def plot_shap_dependence(
        self,
        treatment_group,
        feature_idx,
        shap_dict=None,
        interaction_idx="auto",
        **kwargs,
    ):
        """
         Plots dependency of shapley values for a specified feature, colored by an interaction feature.
         Skips the calculation part if shap_dict is given.

         This plots the value of the feature on the x-axis and the SHAP value of the same feature
         on the y-axis. This shows how the model depends on the given feature, and is like a
         richer extension of the classical partial dependence plots. Vertical dispersion of the
         data points represents interaction effects.

        Args:
             treatment_group (str or int): name of treatment group to create dependency plot on
             feature_idx (str or int): feature index/name to create dependency plot on
             shap_dict (optional, dict): a dict of shapley value matrices. If None, shap_dict will be computed.
             interaction_idx (optional, str or int): feature index/name used in coloring scheme as interaction feature.
                 If "auto" then shap.common.approximate_interactions is used to pick what seems to be the
                 strongest interaction (note that to find to true strongest interaction you need to compute
                 the SHAP interaction values).
        """
        if shap_dict is None:
            shap_dict = self.get_shap_values()

        shap_values = shap_dict[treatment_group]

        shap.dependence_plot(
            feature_idx,
            shap_values,
            self.X,
            interaction_index=interaction_idx,
            feature_names=self.features,
            **kwargs,
        )
