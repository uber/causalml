import numpy as np
import pytest

from causalml.dataset import synthetic_data
from .const import RANDOM_SEED, N_SAMPLE


@pytest.fixture(scope='module')
def generate_regression_data():

    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = synthetic_data(mode=1, n=N_SAMPLE, p=8, sigma=.1)

        return data

    yield _generate_data
