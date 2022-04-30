import numpy as np
from causalml.feature_selection.filters import FilterSelect

from .const import RANDOM_SEED, CONVERSION


def test_filter_f(generate_classification_data):
    # generate uplift classification data
    np.random.seed(RANDOM_SEED)
    df, X_names = generate_classification_data()
    y_name = CONVERSION

    # test F filter
    method = "F"
    filter_f = FilterSelect()
    f_imp = filter_f.get_importance(
        df, X_names, y_name, method, treatment_group="treatment1"
    )

    # each row represents the rank and importance score of each feature
    # and spot check if it's sorted properly
    assert f_imp.shape[0] == len(X_names)
    assert f_imp["rank"].values[0] == 1
    assert f_imp["score"].values[0] >= f_imp["score"].values[1]


def test_filter_lr(generate_classification_data):
    # generate uplift classification data
    np.random.seed(RANDOM_SEED)
    df, X_names = generate_classification_data()
    y_name = CONVERSION

    # test LR filter
    method = "LR"
    filter_obj = FilterSelect()
    imp = filter_obj.get_importance(
        df, X_names, y_name, method, treatment_group="treatment1"
    )

    # each row represents the rank and importance score of each feature
    # and spot check if it's sorted properly
    assert imp.shape[0] == len(X_names)
    assert imp["rank"].values[0] == 1
    assert imp["score"].values[0] >= imp["score"].values[1]


def test_filter_kl(generate_classification_data):
    # generate uplift classification data
    np.random.seed(RANDOM_SEED)
    df, X_names = generate_classification_data()
    y_name = CONVERSION

    # test KL filter
    method = "KL"
    filter_obj = FilterSelect()
    imp = filter_obj.get_importance(
        df, X_names, y_name, method, treatment_group="treatment1"
    )

    # each row represents the rank and importance score of each feature
    # and spot check if it's sorted properly
    assert imp.shape[0] == len(X_names)
    assert imp["rank"].values[0] == 1
    assert imp["score"].values[0] >= imp["score"].values[1]
