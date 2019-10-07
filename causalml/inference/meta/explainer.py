import pandas as pd
import numpy as np
from eli5.sklearn import PermutationImportance
import shap
import matplotlib.pyplot as plt

VALID_METHODS = ('gini', 'permutation', 'shapley')
LEARNER_TYPES = ('S', 'T', 'X', 'R')


class Explainer(object):
    def __init__(self, method, models, control_name, learner_type='S',
                 X=None, treatment=None, y=None, features=None,
                 normalize=True, override_checks=False):
        self.method = method
        self.models = models
        self.control_name = control_name
        self.learner_type = learner_type
        self.X = X
        self.treatment = treatment
        self.y = y
        self.features = features
        self.normalize = normalize

        if not override_checks:
            self.check_conditions()

        if self.features is None:
            if self.method == 'shapley' and self.X is not None:
                num_features = self.X.shape[1]
                self.features = ['Feature_{:03d}'.format(i) for i in range(num_features)]

    def check_conditions(self):
        assert self.learner_type in LEARNER_TYPES
        assert self.method in VALID_METHODS, 'Current supported methods: {}'.format(', '.join(VALID_METHODS))

        if self.method in ('gini', 'shapley'):
            conds = [hasattr(mod, "feature_importances_") for mod in self.models.values()]
            assert all(conds),\
                "All models must have .feature_importances_ attribute if method = {}".format(self.method)

        if self.method in ('permutation', 'shapley'):
            assert all([arr is not None for arr in (self.X, self.treatment, self.y)]), \
                "X, treatment, and y must be provided if method = {}".format(self.method)

    def get_importance(self):
        assert self.method in ('gini', 'permutation')
        if self.method == 'gini':
            importance_dict = self.gini_importance()
        elif self.method == 'permutation':
            importance_dict = self.perm_importance()

        if self.learner_type == 'S':
            # remove the 0th feature: is_treatment
            importance_dict = {group: array[1:] for group, array in importance_dict.items()}

        if self.features is None:
            num_features = list(importance_dict.values())[0].shape[0]
            self.features = ['Feature_{:03d}'.format(i) for i in range(num_features)]

        importance_dict = {group: pd.Series(array, index=self.features).sort_values(ascending=False)
                           for group, array in importance_dict.items()}
        return importance_dict

    def gini_importance(self):
        importance_dict = {group: mod.feature_importances_ for group, mod in self.models.items()}
        if self.normalize:
            for group in self.models:
                importance_dict[group] = importance_dict[group] / importance_dict[group].sum()

        return importance_dict

    def perm_importance(self):
        importance_dict = {}
        for group, mod in self.models.items():
            mask = (self.treatment == group) | (self.treatment == self.control_name)
            X_filt = self.X[mask]
            y_filt = self.y[mask]
            if self.learner_type == 'S':
                w = (self.treatment[mask] == group).astype(int)
                X_filt = np.hstack((w.reshape((-1, 1)), X_filt))

            perm_fitter = PermutationImportance(mod, cv='prefit')
            perm_fitter.fit(X_filt, y_filt)
            importance_dict[group] = perm_fitter.feature_importances_

        return importance_dict

    def get_shap_values(self):
        shap_dict = {}
        for group, mod in self.models.items():
            mask = (self.treatment == group) | (self.treatment == self.control_name)
            X_filt = self.X[mask]
            y_filt = self.y[mask]
            if self.learner_type == 'S':
                w = (self.treatment[mask] == group).astype(int)
                X_filt = np.hstack((w.reshape((-1, 1)), X_filt))

            explainer = shap.TreeExplainer(mod)
            shap_values = explainer.shap_values(X_filt, y_filt)
            if self.learner_type == 'S':
                # remove the 0th feature: is_treatment
                shap_values = shap_values[:, 1:]
            shap_dict[group] = shap_values

        return shap_dict

    def plot_importance(self, importance_dict=None, title_prefix=''):
        if importance_dict is None:
            importance_dict = self.get_importance()
        for group, series in importance_dict.items():
            plt.figure()
            series.plot(kind='bar', figsize=(12, 8))
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
        mask = (self.treatment == treatment_group) | (self.treatment == self.control_name)
        X_filt = self.X[mask]

        shap.dependence_plot(feature_idx, shap_values, X_filt, interaction_index=interaction_idx,
                             feature_names=self.features, **kwargs)
