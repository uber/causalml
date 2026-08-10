"""Bootstrap confidence intervals for auuc_score() and qini_score()."""

import numpy as np
import pandas as pd
import pytest

from causalml.metrics.visualize import auuc_score, qini_score

from .const import RANDOM_SEED

N_BOOTSTRAP = 100


def _uplift_df(n=1500, seed=RANDOM_SEED):
    """RCT with a heterogeneous effect, one informative and one useless ranking."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    w = rng.integers(0, 2, n)
    tau = x
    return pd.DataFrame(
        {
            "y": 0.5 * x + w * tau + rng.normal(0, 0.5, n),
            "w": w,
            "tau": tau,
            "informative_model": x + rng.normal(0, 0.3, n),
            "useless_model": rng.uniform(0, 1, n),
        }
    )


@pytest.fixture
def uplift_df():
    return _uplift_df()


@pytest.mark.parametrize("score_fn", [auuc_score, qini_score])
def test_default_call_is_unchanged(uplift_df, score_fn):
    """return_ci defaults to False, so existing callers keep their Series."""
    result = score_fn(uplift_df)

    assert isinstance(result, pd.Series)
    assert "informative_model" in result.index


@pytest.mark.parametrize(
    ("score_fn", "expected"),
    [
        (qini_score, {"qini", "se", "ci_lower", "ci_upper", "p_value"}),
        (auuc_score, {"auuc", "se", "ci_lower", "ci_upper"}),
    ],
)
def test_return_ci_returns_a_dataframe(uplift_df, score_fn, expected):
    result = score_fn(
        uplift_df,
        return_ci=True,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_SEED,
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == expected
    assert result.index.name == "model"
    assert "informative_model" in result.index


@pytest.mark.parametrize("score_fn", [auuc_score, qini_score])
def test_point_estimate_is_untouched_by_the_bootstrap(uplift_df, score_fn):
    """Adding an interval must not move the number it is an interval around."""
    point = score_fn(uplift_df)
    with_ci = score_fn(
        uplift_df,
        return_ci=True,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_SEED,
    )
    column = with_ci.columns[0]

    for model in point.index:
        assert with_ci.loc[model, column] == pytest.approx(point[model])


@pytest.mark.parametrize("score_fn", [auuc_score, qini_score])
def test_interval_brackets_the_estimate(uplift_df, score_fn):
    result = score_fn(
        uplift_df,
        return_ci=True,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_SEED,
    )
    column = result.columns[0]

    assert (result["ci_lower"] < result[column]).all()
    assert (result["ci_upper"] > result[column]).all()
    assert (result["se"] > 0).all()


def test_a_useless_ranking_is_not_called_better_than_random(uplift_df):
    """Negative control.

    A ranking built from noise carries no information about who benefits, so its
    Qini score must not be separated from zero.
    """
    result = qini_score(
        uplift_df,
        return_ci=True,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_SEED,
    )
    row = result.loc["useless_model"]

    assert row["p_value"] > 0.05
    assert row["ci_lower"] < 0 < row["ci_upper"]


def test_an_informative_ranking_is_separated_from_random(uplift_df):
    """Planted signal: the control above must not be passing for lack of power."""
    result = qini_score(
        uplift_df,
        return_ci=True,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_SEED,
    )
    row = result.loc["informative_model"]

    assert row["qini"] > 0
    assert row["p_value"] < 0.05
    assert row["ci_lower"] > 0


def test_auuc_reports_no_p_value_because_its_null_is_not_zero():
    """A random ranking scores about 0.5 on AUUC, not 0.

    So a two-sided test of H0: AUUC = 0 would reject for every model ever
    scored and mean nothing. This pins the reason the AUUC frame has no
    p_value column while the Qini frame does.
    """
    scores = [
        auuc_score(_uplift_df(n=1500, seed=seed))["useless_model"] for seed in range(12)
    ]

    assert np.mean(scores) == pytest.approx(0.5, abs=0.02)

    result = auuc_score(
        _uplift_df(), return_ci=True, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_SEED
    )
    assert "p_value" not in result.columns


def test_qini_null_really_is_zero():
    """The other half of the same claim: Qini is already differenced against
    random, so zero is the right null for it."""
    scores = [
        qini_score(_uplift_df(n=1500, seed=seed))["useless_model"] for seed in range(12)
    ]

    assert np.mean(scores) == pytest.approx(0.0, abs=0.02)


@pytest.mark.parametrize("score_fn", [auuc_score, qini_score])
def test_standard_error_shrinks_as_the_sample_grows(score_fn):
    """Quadrupling the data should roughly halve the standard error.

    A standard error that ignores the sample size, or scales by the wrong
    power of n, fails here.
    """
    small = score_fn(
        _uplift_df(n=1000),
        return_ci=True,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_SEED,
    )
    large = score_fn(
        _uplift_df(n=4000),
        return_ci=True,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_SEED,
    )
    ratio = small.loc["informative_model", "se"] / large.loc["informative_model", "se"]

    assert 1.4 < ratio < 2.8


@pytest.mark.parametrize("score_fn", [auuc_score, qini_score])
def test_same_random_state_reproduces_the_interval(uplift_df, score_fn):
    first = score_fn(
        uplift_df, return_ci=True, n_bootstrap=50, random_state=RANDOM_SEED
    )
    again = score_fn(
        uplift_df, return_ci=True, n_bootstrap=50, random_state=RANDOM_SEED
    )
    other = score_fn(uplift_df, return_ci=True, n_bootstrap=50, random_state=7)

    pd.testing.assert_frame_equal(first, again)
    assert not np.allclose(first["se"].to_numpy(), other["se"].to_numpy())


@pytest.mark.parametrize("score_fn", [auuc_score, qini_score])
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_bootstrap": 0}, "n_bootstrap"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
    ],
)
def test_invalid_bootstrap_arguments_raise(uplift_df, score_fn, kwargs, match):
    with pytest.raises(ValueError, match=match):
        score_fn(uplift_df, return_ci=True, random_state=RANDOM_SEED, **kwargs)


@pytest.mark.parametrize("score_fn", [auuc_score, qini_score])
def test_tmle_path_refuses_rather_than_silently_refitting(uplift_df, score_fn):
    """Bootstrapping the TMLE path would refit a learner on every draw.

    Better to say so than to hang for an hour.
    """
    with pytest.raises(ValueError, match="tmle"):
        score_fn(uplift_df, tmle=True, return_ci=True, n_bootstrap=N_BOOTSTRAP)


@pytest.mark.parametrize("score_fn", [auuc_score, qini_score])
def test_bootstrap_arguments_are_ignored_when_ci_is_not_requested(uplift_df, score_fn):
    """No silent cost, and no silent validation, on the default path."""
    result = score_fn(uplift_df, n_bootstrap=0, alpha=5.0)

    assert isinstance(result, pd.Series)


@pytest.mark.parametrize("score_fn", [auuc_score, qini_score])
def test_single_draw_is_rejected_rather_than_producing_a_nan_standard_error(
    uplift_df, score_fn
):
    """A single bootstrap draw has no standard error: ``np.std(ddof=1)`` is NaN.

    Left to run, that NaN reached the interval bounds while the p-value was computed
    from ``se > 0 else np.inf`` and came out 0.0 — a certain-looking answer produced
    by the absence of a measurement.
    """
    with pytest.raises(ValueError, match="at least 2"):
        score_fn(uplift_df, return_ci=True, n_bootstrap=1, random_state=RANDOM_SEED)


def test_degenerate_resample_reports_no_p_value():
    """When every draw lands on the same value the standard error is 0.

    There is no p-value to report there, and NaN says so; dividing by zero would
    have said p = 0.0 instead.
    """
    from causalml.metrics.rate import _bootstrap_score_ci

    df = pd.DataFrame({"y": np.arange(40)})
    constant = lambda resampled: pd.Series({"model_a": 1.0})  # noqa: E731

    out = _bootstrap_score_ci(
        df,
        constant,
        pd.Series({"model_a": 1.0}),
        "score",
        n_bootstrap=5,
        alpha=0.05,
        random_state=RANDOM_SEED,
        p_value=True,
    )

    assert out["se"].iloc[0] == 0
    assert np.isnan(out["p_value"].iloc[0])
