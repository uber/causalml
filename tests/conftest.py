import numpy as np
import pytest

from causalml.dataset import synthetic_data
from causalml.dataset import make_uplift_classification

from .const import (
    RANDOM_SEED,
    N_SAMPLE,
    TREATMENT_NAMES,
    CONVERSION,
    DELTA_UPLIFT_INCREASE_DICT,
    N_UPLIFT_INCREASE_DICT,
)


@pytest.fixture(scope="module")
def generate_regression_data(mode: int = 1, p: int = 8, sigma: float = 0.1):
    generated = False

    def _generate_data(mode: int = mode, p: int = p, sigma: float = sigma):
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = synthetic_data(mode=mode, n=N_SAMPLE, p=p, sigma=sigma)

        return data

    yield _generate_data


@pytest.fixture(scope="module")
def generate_classification_data():
    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = make_uplift_classification(
                n_samples=N_SAMPLE,
                treatment_name=TREATMENT_NAMES,
                y_name=CONVERSION,
                random_seed=RANDOM_SEED,
            )

        return data

    yield _generate_data


@pytest.fixture(scope="module")
def generate_classification_data_two_treatments():
    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            data = make_uplift_classification(
                n_samples=N_SAMPLE,
                treatment_name=TREATMENT_NAMES[0:2],
                y_name=CONVERSION,
                random_seed=RANDOM_SEED,
                delta_uplift_increase_dict=DELTA_UPLIFT_INCREASE_DICT,
                n_uplift_increase_dict=N_UPLIFT_INCREASE_DICT,
            )

        return data

    yield _generate_data


def pytest_addoption(parser):
    parser.addoption("--runtf", action="store_true", default=False, help="run tf tests")
    parser.addoption(
        "--runtorch", action="store_true", default=False, help="run torch tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "tf: mark test as tf to run")
    config.addinivalue_line("markers", "torch: mark test as torch to run")


def pytest_collection_modifyitems(config, items):

    skip_tf = False if config.getoption("--runtf") else True
    skip_torch = False if config.getoption("--runtorch") else True

    for item in items:
        if "tf" in item.keywords and skip_tf:
            item.add_marker(pytest.mark.skip(reason="need --runtf option to run"))
        if "torch" in item.keywords and skip_torch:
            item.add_marker(pytest.mark.skip(reason="need --runtorch option to run"))
