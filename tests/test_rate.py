from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest

from causalml.metrics.rate import get_toc, rate_score, plot_toc


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(42)
    n = 500
    tau = rng.normal(1.0, 0.5, n)
    return pd.DataFrame(
        {
            "y": rng.normal(0, 1, n),
            "w": rng.integers(0, 2, n),
            "tau": tau,
            "perfect_model": tau,
            "random_model": rng.normal(0, 1, n),
        }
    )


@pytest.fixture
def rct_df():
    rng = np.random.default_rng(0)
    n = 2000
    tau = rng.uniform(0, 2, n)
    w = rng.integers(0, 2, n)
    y = tau * w + rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "y": y,
            "w": w,
            "score": tau + rng.normal(0, 0.1, n),
        }
    )


def test_get_toc_errors_on_nan():
    df = pd.DataFrame(
        [[0, np.nan, 0.5], [1, np.nan, 0.1], [1, 1, 0.4], [0, 1, 0.3], [1, 1, 0.2]],
        columns=["w", "y", "pred"],
    )

    with pytest.raises(Exception):
        get_toc(df)


def test_get_toc_returns_dataframe(synthetic_df):
    toc = get_toc(synthetic_df, treatment_effect_col="tau")
    assert isinstance(toc, pd.DataFrame)


def test_get_toc_index_bounds(synthetic_df):
    toc = get_toc(synthetic_df, treatment_effect_col="tau")
    assert toc.index[0] == 0.0
    assert toc.index[-1] == 1.0


def test_get_toc_starts_at_zero(synthetic_df):
    toc = get_toc(synthetic_df, treatment_effect_col="tau")
    assert np.allclose(toc.iloc[0].values, 0.0)


def test_get_toc_model_cols_present(synthetic_df):
    toc = get_toc(synthetic_df, treatment_effect_col="tau")
    assert "perfect_model" in toc.columns
    assert "random_model" in toc.columns


def test_get_toc_perfect_model_positive_at_low_q(synthetic_df):
    toc = get_toc(synthetic_df, treatment_effect_col="tau")
    assert toc.loc[toc.index <= 0.15, "perfect_model"].mean() > 0


def test_get_toc_observed_outcomes(rct_df):
    toc = get_toc(rct_df, outcome_col="y", treatment_col="w")
    assert isinstance(toc, pd.DataFrame)
    assert "score" in toc.columns


def test_rate_score_returns_series(synthetic_df):
    result = rate_score(synthetic_df, treatment_effect_col="tau")
    assert isinstance(result, pd.Series)


def test_rate_score_autoc_perfect_model_positive(synthetic_df):
    scores = rate_score(synthetic_df, treatment_effect_col="tau", weighting="autoc")
    assert scores["perfect_model"] > 0


def test_rate_score_qini_perfect_model_positive(synthetic_df):
    scores = rate_score(synthetic_df, treatment_effect_col="tau", weighting="qini")
    assert scores["perfect_model"] > 0


def test_rate_score_perfect_beats_random(synthetic_df):
    scores = rate_score(synthetic_df, treatment_effect_col="tau", weighting="autoc")
    assert scores["perfect_model"] > scores["random_model"]


def test_rate_score_invalid_weighting_raises(synthetic_df):
    with pytest.raises(AssertionError):
        rate_score(synthetic_df, treatment_effect_col="tau", weighting="invalid")


def test_rate_score_constant_tau_near_zero():
    rng = np.random.default_rng(99)
    n = 2000
    df = pd.DataFrame(
        {
            "tau": np.ones(n),
            "random_score": rng.normal(0, 1, n),
        }
    )
    scores = rate_score(df, treatment_effect_col="tau", weighting="autoc")
    assert abs(scores["random_score"]) < 0.2


def test_rate_score_all_finite(synthetic_df):
    scores = rate_score(synthetic_df, treatment_effect_col="tau")
    assert np.isfinite(scores.values).all()


def test_rate_score_observed_outcomes(rct_df):
    scores = rate_score(rct_df, outcome_col="y", treatment_col="w")
    assert isinstance(scores, pd.Series)
    assert np.isfinite(scores.values).all()


def test_plot_toc(synthetic_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    ax = plot_toc(synthetic_df, treatment_effect_col="tau")
    assert isinstance(ax, plt.Axes)
