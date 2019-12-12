import numpy as np


def check_treatment_vector(treatment, control_name=None):
    n_unique_treatments = np.unique(treatment).shape[0]
    assert n_unique_treatments > 1, \
        'Treatment vector must have at least two levels.'
    if control_name is not None:
        assert control_name in treatment, \
            'Control group level {} not found in treatment vector.'.format(control_name)


def check_p_conditions(p, t_groups):
    assert isinstance(p, (np.ndarray, dict)), \
        'p must be an np.ndarray or dict type'
    if isinstance(p, np.ndarray):
        assert t_groups.shape[0] == 1, \
            'If p is passed as an np.ndarray, there must be only 1 unique non-control group in the treatment vector.'


def check_explain_conditions(method, models, X=None, treatment=None, y=None):
    valid_methods = ['gini', 'permutation', 'shapley']
    assert method in valid_methods, 'Current supported methods: {}'.format(', '.join(valid_methods))

    if method in ('gini', 'shapley'):
        conds = [hasattr(mod, "feature_importances_") for mod in models]
        assert all(conds), "Both models must have .feature_importances_ attribute if method = {}".format(method)

    if method in ('permutation', 'shapley'):
        assert all([arr is not None for arr in (X, treatment, y)]), \
                "X, treatment, and y must be provided if method = {}".format(method)
