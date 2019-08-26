import numpy as np


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
