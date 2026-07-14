import numpy as np
import pytest

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


@pytest.mark.parametrize("method", ["LR", "KL", "ED", "Chi"])
def test_filter_rejects_non_binary_outcome(generate_classification_data, method):
    """Regression test for uber/causalml#349.

    The likelihood-ratio (``LR``) and divergence-based (``KL``/``ED``/``Chi``)
    filters model a binary outcome. A multi-class (or continuous) outcome
    previously produced meaningless statistics silently; it should now raise a
    clear ``ValueError``.
    """
    np.random.seed(RANDOM_SEED)
    df, X_names = generate_classification_data()
    df = df.copy()
    # Turn the binary conversion outcome into a 4-class outcome.
    df[CONVERSION] = np.random.randint(0, 4, size=df.shape[0])

    filter_obj = FilterSelect()
    with pytest.raises(ValueError, match="binary"):
        filter_obj.get_importance(
            df, X_names, CONVERSION, method, treatment_group="treatment1"
        )


def test_filter_f_accepts_continuous_outcome(generate_classification_data):
    """``filter_F`` (OLS F-test) tolerates a continuous outcome (uber/causalml#349)."""
    np.random.seed(RANDOM_SEED)
    df, X_names = generate_classification_data()
    df = df.copy()
    df[CONVERSION] = np.random.normal(size=df.shape[0])

    filter_obj = FilterSelect()
    f_imp = filter_obj.get_importance(
        df, X_names, CONVERSION, "F", treatment_group="treatment1"
    )
    assert f_imp.shape[0] == len(X_names)
