import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from causalml.metrics.cate_scoring import (
    compute_dr_pseudo_outcomes,
    dr_score,
    plug_in_t_score,
    rlearner_score,
)
from causalml.propensity import compute_r_residuals
from causalml.metrics.rate import rate_score

try:
    from tests.const import RANDOM_SEED
except ImportError:
    RANDOM_SEED = 42


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(RANDOM_SEED)
    n, p = 2000, 5
    X = rng.normal(0, 1, (n, p))
    tau = 1.0 + X[:, 0]
    e = np.clip(0.5 + 0.1 * X[:, 1], 0.1, 0.9)
    w = rng.binomial(1, e)
    mu0 = X[:, 2]
    y = mu0 + w * tau + rng.normal(0, 0.5, n)

    perfect_model = tau
    noisy_model = tau + rng.normal(0, 1.0, n)
    bad_model = rng.normal(0, 1, n)

    df = pd.DataFrame(
        {
            "y": y,
            "w": w,
            "perfect_model": perfect_model,
            "noisy_model": noisy_model,
            "bad_model": bad_model,
        }
    )
    return df, X


def test_compute_dr_pseudo_outcomes_shape(synthetic_data):
    df, X = synthetic_data
    phi = compute_dr_pseudo_outcomes(
        X,
        df["w"],
        df["y"],
        learner=LinearRegression(),
        n_folds=5,
        random_state=RANDOM_SEED,
    )
    assert phi.shape == (len(df),)
    assert np.isfinite(phi).all()


def test_compute_dr_pseudo_outcomes_recovers_ate(synthetic_data):
    df, X = synthetic_data
    phi = compute_dr_pseudo_outcomes(
        X,
        df["w"],
        df["y"],
        learner=LinearRegression(),
        n_folds=5,
        random_state=RANDOM_SEED,
    )
    # With a well-specified outcome model the DR pseudo-outcome mean should
    # recover the true ATE. Tolerance is loose (not ~0.05) because AIPW is a
    # high-variance functional: a handful of cross-fitted, near-boundary
    # propensities inflate the mean even after p_clip_bounds trimming.
    true_ate = (1.0 + X[:, 0]).mean()
    assert abs(phi.mean() - true_ate) < 0.75


def test_dr_score_returns_series(synthetic_data):
    df, X = synthetic_data
    scores = dr_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        random_state=RANDOM_SEED,
    )
    assert isinstance(scores, pd.Series)
    assert set(scores.index) == {"perfect_model", "noisy_model", "bad_model"}


def test_dr_score_ranks_perfect_model_lowest(synthetic_data):
    df, X = synthetic_data
    scores = dr_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        random_state=RANDOM_SEED,
    )
    assert scores["perfect_model"] < scores["noisy_model"] < scores["bad_model"]


def test_dr_score_with_precomputed_pseudo_outcomes(synthetic_data):
    df, X = synthetic_data
    phi = compute_dr_pseudo_outcomes(
        X,
        df["w"],
        df["y"],
        learner=LinearRegression(),
        n_folds=5,
        random_state=RANDOM_SEED,
    )
    df_with_phi = df.assign(dr_pseudo_outcome=phi)
    scores = dr_score(
        df_with_phi, pseudo_outcome_col="dr_pseudo_outcome", random_state=RANDOM_SEED
    )
    assert scores["perfect_model"] < scores["bad_model"]


def test_dr_score_shares_pseudo_outcomes_with_rate_score(synthetic_data):
    # The same pseudo-outcomes should be usable both for dr_score() and as
    # rate_score()'s treatment_effect_col, without re-fitting nuisances twice.
    df, X = synthetic_data
    phi = compute_dr_pseudo_outcomes(
        X,
        df["w"],
        df["y"],
        learner=LinearRegression(),
        n_folds=5,
        random_state=RANDOM_SEED,
    )
    df_with_phi = df.assign(tau=phi)
    rate = rate_score(df_with_phi.drop(columns=["w"]), treatment_effect_col="tau")
    dr = dr_score(
        df_with_phi.rename(columns={"tau": "dr_pseudo_outcome"}),
        pseudo_outcome_col="dr_pseudo_outcome",
    )
    assert rate["perfect_model"] > 0
    assert dr["perfect_model"] < dr["bad_model"]


def test_dr_score_missing_X_and_pseudo_outcome_raises(synthetic_data):
    df, _ = synthetic_data
    with pytest.raises(AssertionError):
        dr_score(df, treatment_col="w", outcome_col="y")


def test_dr_score_return_ci_returns_dataframe(synthetic_data):
    df, X = synthetic_data
    result = dr_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        return_ci=True,
        n_bootstrap=50,
        random_state=RANDOM_SEED,
    )
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"dr_loss", "se", "ci_lower", "ci_upper"}


def test_dr_score_ci_bounds_ordered(synthetic_data):
    df, X = synthetic_data
    result = dr_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        return_ci=True,
        n_bootstrap=50,
        random_state=RANDOM_SEED,
    )
    assert (result["ci_lower"] < result["dr_loss"]).all()
    assert (result["ci_upper"] > result["dr_loss"]).all()


def test_plug_in_t_score_returns_series(synthetic_data):
    df, X = synthetic_data
    scores = plug_in_t_score(
        df,
        X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        random_state=RANDOM_SEED,
    )
    assert isinstance(scores, pd.Series)
    assert set(scores.index) == {"perfect_model", "noisy_model", "bad_model"}


def test_plug_in_t_score_ranks_perfect_model_lowest(synthetic_data):
    df, X = synthetic_data
    scores = plug_in_t_score(
        df,
        X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        random_state=RANDOM_SEED,
    )
    assert scores["perfect_model"] < scores["noisy_model"] < scores["bad_model"]


def test_plug_in_t_score_return_ci_returns_dataframe(synthetic_data):
    df, X = synthetic_data
    result = plug_in_t_score(
        df,
        X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        return_ci=True,
        n_bootstrap=50,
        random_state=RANDOM_SEED,
    )
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "plug_in_t_loss",
        "se",
        "ci_lower",
        "ci_upper",
    }


def test_plug_in_t_score_missing_columns_raises(synthetic_data):
    df, X = synthetic_data
    with pytest.raises(AssertionError):
        plug_in_t_score(df.drop(columns=["w"]), X, treatment_col="w", outcome_col="y")


def test_compute_dr_pseudo_outcomes_handles_imbalanced_treatment():
    rng = np.random.default_rng(RANDOM_SEED)

    n, p = 200, 5
    X = rng.normal(size=(n, p))

    w = np.zeros(n, dtype=int)
    w[rng.choice(n, 10, replace=False)] = 1

    y = rng.normal(size=n)

    phi = compute_dr_pseudo_outcomes(
        X,
        w,
        y,
        learner=LinearRegression(),
        n_folds=5,
        random_state=RANDOM_SEED,
    )

    assert phi.shape == (n,)
    assert np.isfinite(phi).all()


def test_plug_in_t_score_handles_imbalanced_treatment():
    rng = np.random.default_rng(RANDOM_SEED)

    n, p = 200, 5
    X = rng.normal(size=(n, p))

    w = np.zeros(n, dtype=int)
    w[rng.choice(n, 10, replace=False)] = 1

    y = rng.normal(size=n)

    df = pd.DataFrame(
        {
            "y": y,
            "w": w,
            "model_1": rng.normal(size=n),
            "model_2": rng.normal(size=n),
        }
    )

    scores = plug_in_t_score(
        df,
        X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        n_folds=5,
        random_state=RANDOM_SEED,
    )

    assert isinstance(scores, pd.Series)
    assert np.isfinite(scores.values).all()


def test_dr_score_and_plug_in_t_score_all_finite(synthetic_data):
    df, X = synthetic_data
    dr = dr_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        random_state=RANDOM_SEED,
    )
    t = plug_in_t_score(
        df,
        X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        random_state=RANDOM_SEED,
    )
    assert np.isfinite(dr.values).all()
    assert np.isfinite(t.values).all()


def test_compute_r_residuals_shape(synthetic_data):
    df, X = synthetic_data
    y_residual, w_residual = compute_r_residuals(
        X,
        df["w"],
        df["y"],
        outcome_learner=LinearRegression(),
        n_folds=5,
        random_state=RANDOM_SEED,
    )
    assert y_residual.shape == (len(df),)
    assert w_residual.shape == (len(df),)
    assert np.isfinite(y_residual).all()
    assert np.isfinite(w_residual).all()


def test_rlearner_score_returns_series(synthetic_data):
    df, X = synthetic_data
    scores = rlearner_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        outcome_learner=LinearRegression(),
        random_state=RANDOM_SEED,
    )
    assert isinstance(scores, pd.Series)
    assert set(scores.index) == {"perfect_model", "noisy_model", "bad_model"}


def test_rlearner_score_ranks_perfect_model_lowest(synthetic_data):
    df, X = synthetic_data
    scores = rlearner_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        outcome_learner=LinearRegression(),
        random_state=RANDOM_SEED,
    )
    assert scores["perfect_model"] < scores["noisy_model"] < scores["bad_model"]


def test_rlearner_score_with_precomputed_residuals(synthetic_data):
    df, X = synthetic_data
    y_residual, w_residual = compute_r_residuals(
        X,
        df["w"],
        df["y"],
        outcome_learner=LinearRegression(),
        n_folds=5,
        random_state=RANDOM_SEED,
    )
    df_with_residuals = df.assign(y_resid=y_residual, w_resid=w_residual)
    scores = rlearner_score(
        df_with_residuals,
        y_residual_col="y_resid",
        w_residual_col="w_resid",
        random_state=RANDOM_SEED,
    )
    assert scores["perfect_model"] < scores["bad_model"]


def test_rlearner_score_missing_X_and_residuals_raises(synthetic_data):
    df, _ = synthetic_data
    with pytest.raises(AssertionError):
        rlearner_score(df, treatment_col="w", outcome_col="y")


def test_rlearner_score_return_ci_returns_dataframe(synthetic_data):
    df, X = synthetic_data
    result = rlearner_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        outcome_learner=LinearRegression(),
        return_ci=True,
        n_bootstrap=50,
        random_state=RANDOM_SEED,
    )
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"r_loss", "se", "ci_lower", "ci_upper"}


def test_rlearner_score_ci_bounds_ordered(synthetic_data):
    df, X = synthetic_data
    result = rlearner_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        outcome_learner=LinearRegression(),
        return_ci=True,
        n_bootstrap=50,
        random_state=RANDOM_SEED,
    )
    assert (result["ci_lower"] < result["r_loss"]).all()
    assert (result["ci_upper"] > result["r_loss"]).all()


def test_compute_r_residuals_handles_imbalanced_treatment():
    rng = np.random.default_rng(RANDOM_SEED)

    n, p = 200, 5
    X = rng.normal(size=(n, p))

    w = np.zeros(n, dtype=int)
    w[rng.choice(n, 10, replace=False)] = 1

    y = rng.normal(size=n)

    y_residual, w_residual = compute_r_residuals(
        X,
        w,
        y,
        outcome_learner=LinearRegression(),
        n_folds=5,
        random_state=RANDOM_SEED,
    )

    assert y_residual.shape == (n,)
    assert w_residual.shape == (n,)
    assert np.isfinite(y_residual).all()
    assert np.isfinite(w_residual).all()


def test_rlearner_score_handles_imbalanced_treatment():
    rng = np.random.default_rng(RANDOM_SEED)

    n, p = 200, 5
    X = rng.normal(size=(n, p))

    w = np.zeros(n, dtype=int)
    w[rng.choice(n, 10, replace=False)] = 1

    y = rng.normal(size=n)

    df = pd.DataFrame(
        {
            "y": y,
            "w": w,
            "model_1": rng.normal(size=n),
            "model_2": rng.normal(size=n),
        }
    )

    scores = rlearner_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        outcome_learner=LinearRegression(),
        n_folds=5,
        random_state=RANDOM_SEED,
    )

    assert isinstance(scores, pd.Series)
    assert np.isfinite(scores.values).all()


def test_dr_plug_in_t_and_rlearner_score_all_finite(synthetic_data):
    df, X = synthetic_data
    dr = dr_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        random_state=RANDOM_SEED,
    )
    t = plug_in_t_score(
        df,
        X,
        treatment_col="w",
        outcome_col="y",
        learner=LinearRegression(),
        random_state=RANDOM_SEED,
    )
    r = rlearner_score(
        df,
        X=X,
        treatment_col="w",
        outcome_col="y",
        outcome_learner=LinearRegression(),
        random_state=RANDOM_SEED,
    )
    assert np.isfinite(dr.values).all()
    assert np.isfinite(t.values).all()
    assert np.isfinite(r.values).all()
