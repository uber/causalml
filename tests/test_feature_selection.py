import numpy as np
import pandas as pd
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


def test_filter_kl_bin_with_no_conversions():
    """A bin where nobody converted must not turn the whole score into NaN.

    ``_kl_divergence`` clamps the control probability away from 0 and 1 but not
    the treatment one, so ``pk = 0`` gives ``0 * log(0 / qk)`` -> NaN, and the
    NaN then propagates through the bin sum to the feature's score and rank.
    """
    n = 400
    x = np.arange(n) % 20 * 1.0
    df = pd.DataFrame(
        {
            "x": x,
            "treatment_group_key": np.where(
                np.arange(n) % 2 == 0, "control", "treatment"
            ),
            # conversions only happen in the upper half of x, so the lower bins
            # have no converters in either arm
            CONVERSION: ((x >= 10) & (np.arange(n) % 3 == 0)).astype(int),
        }
    )

    imp = FilterSelect().filter_D(
        data=df, features=["x"], y_name=CONVERSION, n_bins=5, method="KL"
    )

    assert np.isfinite(imp["score"].values[0])
    assert imp["rank"].values[0] == 1


@pytest.mark.parametrize(
    "pk, qk",
    [(0.0, 0.5), (1.0, 0.5), (0.0, 0.001), (1.0, 0.001), (0.0, 0.999)],
)
def test_kl_divergence_at_the_limits(pk, qk):
    """``pk`` at 0 or 1 must give the limit, not an epsilon-shifted approximation.

    Clamping ``pk`` keeps the result finite but biased low, by ~0.8% at
    ``qk = 0.001``. ``_kl_divergence`` in
    ``causalml/inference/tree/_uplift/_criterion.pyx`` already takes the limits
    directly; this keeps the two implementations agreeing.

    These pairs cover direct callers of the scalar. ``filter_D`` cannot emit a
    degenerate ``pk`` against a non-degenerate ``qk``; see
    ``test_kl_divergence_identical_arms`` for the pairs it does reach.
    """
    expected = -np.log(1 - qk) if pk == 0 else -np.log(qk)

    assert FilterSelect._kl_divergence(pk, qk) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("value", [0.0, 1.0, 0.3])
def test_kl_divergence_identical_arms(value):
    """Two arms that behaved the same have no divergence between them.

    ``_GetNodeSummary`` smooths a missing count to 1, so ``pk`` only reaches 0
    or 1 when the whole bin is degenerate, which puts ``qk`` at the same value.
    (0, 0) and (1, 1) are therefore the degenerate pairs ``filter_D`` actually
    produces, and clamping ``qk`` away from ``pk`` before comparing them would
    report a difference of about 1e-6 between identical arms.
    """
    assert FilterSelect._kl_divergence(value, value) == 0.0


def test_filter_kl_bin_where_everyone_converted():
    """``pk = 1`` is the other NaN trigger: a bin in which every unit converted."""
    n = 400
    x = np.arange(n) % 20 * 1.0
    df = pd.DataFrame(
        {
            "x": x,
            "treatment_group_key": np.where(
                np.arange(n) % 2 == 0, "control", "treatment"
            ),
            # the upper bins convert in both arms, so their treatment probability
            # is exactly 1
            CONVERSION: (x >= 10).astype(int),
        }
    )

    imp = FilterSelect().filter_D(
        data=df, features=["x"], y_name=CONVERSION, n_bins=5, method="KL"
    )

    # every bin is identical across the two arms, so there is no divergence
    # for the feature to report
    assert imp["score"].values[0] == 0
    assert imp["rank"].values[0] == 1
