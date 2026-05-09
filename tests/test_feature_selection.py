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


@pytest.mark.parametrize("method", ["LR", "KL", "ED", "Chi"])
def test_filter_rejects_non_binary_outcome(generate_classification_data, method):
    """Regression test for uber/causalml#349.

    Before the fix, passing a non-binary outcome to ``filter_LR`` /
    ``filter_D`` (KL/ED/Chi) would silently mis-handle it: ``Logit`` would
    raise a ``PerfectSeparationError`` deep inside statsmodels, and the
    bin-based filters' ``_GetNodeSummary`` would only count ``y == 0``
    and ``y == 1`` rows — producing meaningless scores when the label set
    is e.g. ``{0, 1, 2, 3}``.

    After the fix, the public entry points raise a clear ``ValueError``
    that names the offending column and points the user at the limitation.
    """
    np.random.seed(RANDOM_SEED)
    df, X_names = generate_classification_data()
    y_name = CONVERSION

    # Replace the binary outcome with a multi-class label set that includes
    # 0 and 1 (so a naive value-counts check could miss the gap).
    df = df.copy()
    df[y_name] = np.random.randint(0, 4, size=df.shape[0])

    filter_obj = FilterSelect()
    with pytest.raises(ValueError, match="binary"):
        filter_obj.get_importance(
            df, X_names, y_name, method, treatment_group="treatment1"
        )


def test_filter_f_accepts_continuous_outcome(generate_classification_data):
    """``filter_F`` uses OLS and is documented to tolerate continuous y;
    the binary-outcome guard introduced for #349 must not affect it."""
    np.random.seed(RANDOM_SEED)
    df, X_names = generate_classification_data()
    y_name = CONVERSION
    df = df.copy()
    df[y_name] = np.random.rand(df.shape[0])  # continuous outcome

    filter_obj = FilterSelect()
    imp = filter_obj.get_importance(
        df, X_names, y_name, "F", treatment_group="treatment1"
    )
    assert imp.shape[0] == len(X_names)


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
