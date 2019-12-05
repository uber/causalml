import pandas as pd
import numpy as np


def convert_pd_to_np(*args):
    pd_types = (pd.DataFrame, pd.Series)
    output = [obj.values if isinstance(obj, pd_types) else obj for obj in args]
    if len(output) == 1:
        return output[0]
    else:
        return output


def check_control_in_treatment(treatment, control_name):
    if np.unique(treatment).shape[0] == 2:
        assert control_name in treatment, \
            'If treatment vector has 2 unique values, one of them must be the control (specify in init step).'


def check_p_conditions(p, t_groups):
    assert isinstance(p, (np.ndarray, pd.Series, dict)), \
        'p must be an np.ndarray, pd.Series, or dict type'
    if isinstance(p, (np.ndarray, pd.Series)):
        assert t_groups.shape[0] == 1, \
            'If p is passed as an np.ndarray, there must be only 1 unique non-control group in the treatment vector.'
