import numpy as np

from packaging import version
from xgboost import (
    __version__ as xgboost_version,
    XGBRegressor
)


def check_control_in_treatment(treatment, control_name):
    if np.unique(treatment).shape[0] == 2:
        assert control_name in treatment, \
            'If treatment vector has 2 unique values, one of them must be the control (specify in init step).'


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


def xgb_with_valid_objective(xgb_constructor=XGBRegressor):
    """
    Wrapper for xgboost constructors that avoids warnings from deprecated default arguments
    that exist in some version 0.90.

    Returns
    -------
    A new xgboost object
    """
    valid_objective = clean_xgboost_objective('reg:squarederror')
    return xgb_constructor(objective=valid_objective)
