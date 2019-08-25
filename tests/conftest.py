import numpy as np
import pytest

from causalml.dataset import synthetic_data
from causalml.dataset import make_uplift_classification

from .const import RANDOM_SEED, N_SAMPLE, TREATMENT_NAMES, CONVERSION


@pytest.fixture(scope='module')
def generate_regression_data():

    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = synthetic_data(mode=1, n=N_SAMPLE, p=8, sigma=.1)

        return data

    yield _generate_data


@pytest.fixture(scope='module')
def generate_classification_data():

    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = make_uplift_classification(n_samples=N_SAMPLE,
                                              treatment_name=TREATMENT_NAMES,
                                              y_name=CONVERSION,
                                              random_seed=RANDOM_SEED)

        return data

    yield _generate_data
