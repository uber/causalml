import numpy as np
import pandas as pd
import pytest
from numpy import isclose
from causalml.metrics.classification import logloss
from causalml.metrics.regression import mape, smape
from causalml.metrics.visualize import qini_score


def test_qini_score():
    test_df = pd.DataFrame(
        {"y": [0, 0, 0, 0, 1, 0, 0, 1, 1, 1], "w": [0] * 5 + [1] * 5}
    )

    good_uplift = [_ / 10 for _ in range(0, 5)]
    bad_uplift = [1] + [0] * 9
    test_df["learner_1"] = good_uplift * 2
    # learner_2 is a bad model because it gives zero for almost all rows of data
    test_df["learner_2"] = bad_uplift

    # get qini score for 2 models in the single calling of qini_score
    full_result = qini_score(test_df)

    # get qini score for learner_1 separately
    learner_1_result = qini_score(test_df[["y", "w", "learner_1"]])

    # get qini score for learner_2 separately
    learner_2_result = qini_score(test_df[["y", "w", "learner_2"]])

    # for each learner, its qini score should stay same no matter calling with another model or calling separately
    assert isclose(full_result["learner_1"], learner_1_result["learner_1"])
    assert isclose(full_result["learner_2"], learner_2_result["learner_2"])


def test_logloss_does_not_modify_prediction():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.0, 1.0, 0.2, 0.8, 0.9])
    p_before = p.copy()

    logloss(y, p)

    # logloss clips p into (0, 1) and must do so on a copy, not on the caller's array
    assert np.array_equal(p, p_before)


def test_logloss_clips_degenerate_predictions():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])

    assert np.isfinite(logloss(y, p))


def test_mape_raises_when_every_target_is_zero():
    y = np.zeros(100)
    p = np.zeros(100)

    with pytest.raises(ValueError):
        mape(y, p)


def test_mape_ignores_zero_targets_but_keeps_the_rest():
    y = np.array([0.0, 2.0, 0.0, 8.0])
    p = np.array([0.0, 1.8, 0.5, 7.0])

    # Only the two non-zero targets contribute: |1 - 1.8/2| and |1 - 7/8|.
    assert isclose(mape(y, p), 0.1125)


def test_smape_is_zero_when_target_and_prediction_are_both_zero():
    y = np.zeros(100)
    p = np.zeros(100)

    # A zero prediction for a zero target is exact, not undefined.
    assert smape(y, p) == 0.0


def test_smape_does_not_nan_on_a_single_zero_pair():
    y = np.array([0.0, 2.0, 0.0, 8.0])
    p = np.array([0.0, 1.8, 0.5, 7.0])

    result = smape(y, p)
    assert not np.isnan(result)
    assert isclose(result, 0.5596491228070176)


def test_smape_unchanged_for_non_degenerate_input():
    y = np.array([1.0, 2.0, 4.0, 8.0])
    p = np.array([1.1, 1.8, 4.4, 7.0])

    assert isclose(smape(y, p), 0.1072681704260651)
