import numpy as np

from packaging import version
from xgboost import __version__ as xgboost_version


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


def clean_xgboost_objective(objective):
    """
    Translate objective to be compatible with loaded xgboost version

    Args
    ----

    objective : string
        The objective to translate.

    Returns
    -------
    The translated objective, or original if no translation was required.
    """
    compat_before_v83 = {'reg:squarederror': 'reg:linear'}
    compat_v83_or_later = {'reg:linear': 'reg:squarederror'}
    if version.parse(xgboost_version) < version.parse('0.83'):
        if objective in compat_before_v83:
            objective = compat_before_v83[objective]
    else:
        if objective in compat_v83_or_later:
            objective = compat_v83_or_later[objective]
    return objective


def get_xgboost_objective_metric(objective):
    """
    Get the xgboost version-compatible objective and evaluation metric from a potentially version-incompatible input.

    Args
    ----

    objective : string
        An xgboost objective that may be incompatible with the installed version.

    Returns
    -------
    A tuple with the translated objective and evaluation metric.
    """
    def clean_dict_keys(orig):
        return {clean_xgboost_objective(k): v for (k, v) in orig.items()}

    metric_mapping = clean_dict_keys({
        'rank:pairwise': 'auc',
        'reg:squarederror': 'rmse',
    })

    objective = clean_xgboost_objective(objective)

    assert (objective in metric_mapping), \
        'Effect learner objective must be one of: ' + ", ".join(metric_mapping)
    return objective, metric_mapping[objective]
