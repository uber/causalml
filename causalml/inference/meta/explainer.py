import pandas as pd
import numpy as np
from eli5.sklearn import PermutationImportance
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from copy import deepcopy

VALID_METHODS = ('gini', 'permutation', 'shapley')


class Explainer(object):
    def __init__(self, method, control_name, X, tau, classes, model_tau=None,
                 features=None, normalize=True, override_checks=False, r_learners=None):
        self.method = method
        self.control_name = control_name
        self.X = X
        self.tau = tau
        self.classes = classes
        self.model_tau = LGBMRegressor() if model_tau is None else model_tau
        self.features = features
        self.normalize = normalize
        self.override_checks = override_checks
        self.r_learners = r_learners

        self.importance_catalog = {'gini': self.gini_importance, 'permutation': self.perm_importance}

        if not self.override_checks:
            self.check_conditions()
            self.create_feature_names()
            if self.method in ('gini', 'shapley'):
                self.build_new_tau_models()

    def check_conditions(self):
        assert self.method in VALID_METHODS, 'Current supported methods: {}'.format(', '.join(VALID_METHODS))

        assert all([obj is not None for obj in (self.X, self.tau, self.classes)]), \
            "X, tau, and classes must be provided."

        model_test = deepcopy(self.model_tau)
        model_test.fit([[0], [1]], [0, 1])  # Fit w/ dummy data to check for feature_importances_ below
        assert hasattr(model_test, "feature_importances_"), \
            "model_tau must have the feature_importances_ method (after fitting)"

    def create_feature_names(self):
        if self.features is None:
            num_features = self.X.shape[1]
            self.features = ['Feature_{:03d}'.format(i) for i in range(num_features)]

    def get_importance(self):
        assert self.method in self.importance_catalog
        importance_dict = self.importance_catalog[self.method]()

        if self.features is None:
            num_features = self.X.shape[1]
            self.features = ['Feature_{:03d}'.format(i) for i in range(num_features)]

        importance_dict = {group: pd.Series(array, index=self.features).sort_values(ascending=False)
                           for group, array in importance_dict.items()}
        return importance_dict

    def build_new_tau_models(self):
        if self.r_learners is not None:
            self.models_tau = deepcopy(self.r_learners)
        else:
            self.models_tau = {group: deepcopy(self.model_tau) for group in self.classes}
            for group, idx in self.classes.items():
                self.models_tau[group].fit(self.X, self.tau[:, idx])

    def gini_importance(self):
        importance_dict = {}
        for group, idx in self.classes.items():
            importance_dict[group] = self.models_tau[group].feature_importances_
            if self.normalize:
                importance_dict[group] = importance_dict[group] / importance_dict[group].sum()

        return importance_dict

    def perm_importance(self):
        importance_dict = {}
        for group, idx in self.classes.items():
            if self.r_learners is None:
                perm_estimator = self.model_tau
                cv = 3
            else:
                perm_estimator = self.r_learners[group]
                cv = 'prefit'
            perm_fitter = PermutationImportance(perm_estimator, cv=cv)
            perm_fitter.fit(self.X, self.tau[:, idx])
            importance_dict[group] = perm_fitter.feature_importances_

        return importance_dict

    def get_shap_values(self):
        shap_dict = {}
        for group, mod in self.models_tau.items():
            explainer = shap.TreeExplainer(mod)
            if self.r_learners is not None:
                explainer.model.original_model.params['objective'] = None  # hacky way of running shap without error
            shap_values = explainer.shap_values(self.X)
            shap_dict[group] = shap_values

        return shap_dict

    def plot_importance(self, importance_dict=None, title_prefix=''):
        if importance_dict is None:
            importance_dict = self.get_importance()
        for group, series in importance_dict.items():
            plt.figure()
            series.sort_values().plot(kind='barh', figsize=(12, 8))
            title = group
            if title_prefix != '':
                title = '{} - {}'.format(title_prefix, title)
            plt.title(title)

    def plot_shap_values(self, shap_dict=None):
        if shap_dict is None:
            shap_dict = self.get_shap_values()

        for group, values in shap_dict.items():
            plt.title(group)
            shap.summary_plot(values, feature_names=self.features)

    def plot_shap_dependence(self, treatment_group, feature_idx, shap_dict=None, interaction_idx='auto', **kwargs):
        if shap_dict is None:
            shap_dict = self.get_shap_values()

        shap_values = shap_dict[treatment_group]

        shap.dependence_plot(feature_idx, shap_values, self.X, interaction_index=interaction_idx,
                             feature_names=self.features, **kwargs)
