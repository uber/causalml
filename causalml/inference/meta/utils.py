import pandas as pd
import numpy as np


def convert_pd_to_np(*args):
    output = [obj.to_numpy() if hasattr(obj, "to_numpy") else obj for obj in args]
    return output if len(output) > 1 else output[0]


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
