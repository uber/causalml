from copy import deepcopy
import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from causalml.propensity import compute_propensity_score, compute_r_residuals

logger = logging.getLogger("causalml")

# Sentinel default for `learner` in compute_dr_pseudo_outcomes() / dr_score() /
# plug_in_t_score(). These functions fit per-fold, per-arm (with the default
# n_folds=5, that's 10 fits per call, each on a fraction of a single arm), so a
# single fixed-size default learner is either oversized for small slices or
# undersized for large ones. Using a sentinel instead of a pre-built estimator
# lets _make_fold_learner() build one sized to each fit's actual row count --
# see _sized_default_learner(). It also avoids the mutable-default-argument
# pitfall of instantiating one shared estimator at import time.
_AUTO = "auto"


def _sized_default_learner(n_samples, min_child_samples=20, max_leaves=64):
    """Build a default LGBMRegressor with num_leaves capped to what n_samples supports.

    LightGBM's own default ``min_child_samples`` (20) means a tree can't grow
    past roughly ``n_samples // min_child_samples`` leaves anyway -- asking for
    more just produces "No further splits with positive gain" log noise rather
    than a more expressive model. This module fits per-fold and, for
    ``compute_dr_pseudo_outcomes``/``plug_in_t_score``, per-arm on top of that
    (e.g. the default ``n_folds=5`` means 10 fits per call, each on well under
    a full arm's worth of rows), so a fixed ``num_leaves=64`` is frequently
    past what a single fit's slice can support. Sizing it to the slice keeps
    the model honest about what it can learn from that slice.

    Args:
        n_samples (int): number of training rows this learner will be fit on
        min_child_samples (int, optional): min_child_samples for the learner;
            also used to size num_leaves. Default 20 (LightGBM's own default).
        max_leaves (int, optional): ceiling on num_leaves regardless of
            n_samples, matching the previous fixed default. Default 64.

    Returns:
        (LGBMRegressor): a fresh, unfitted default learner
    """
    num_leaves = int(np.clip(n_samples // min_child_samples, 2, max_leaves))
    return LGBMRegressor(
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        learning_rate=0.05,
        n_estimators=300,
        verbose=-1,
    )


def _make_fold_learner(resolved_learner, n_samples):
    """Materialize a fresh, unfitted learner for a single fold/arm fit.

    If ``resolved_learner`` is the ``_AUTO`` sentinel (the caller didn't pass
    an explicit learner), build a default learner sized to ``n_samples`` via
    ``_sized_default_learner()``. Otherwise, deep-copy the caller-supplied
    learner so that different folds/arms don't share fitted state.

    Args:
        resolved_learner: either the ``_AUTO`` sentinel or an estimator
            already isolated per arm by ``_resolve_outcome_learners()``
        n_samples (int): number of rows this learner is about to be fit on

    Returns:
        An unfitted estimator ready for ``.fit()``.
    """
    if resolved_learner is _AUTO:
        return _sized_default_learner(n_samples)
    return deepcopy(resolved_learner)


def _resolve_outcome_learners(
    learner, control_outcome_learner, treatment_outcome_learner
):
    """Resolve control/treatment outcome learners from constructor-style args.

    Mirrors the ``learner`` / ``control_*_learner`` / ``treatment_*_learner``
    convention used throughout ``causalml.inference.meta`` (e.g. ``BaseDRLearner``,
    ``BaseTLearner``).

    Args:
        learner (model or the ``_AUTO`` sentinel, optional): a model used for
            both control and treatment outcome regressions if the
            group-specific learners are not given. ``_AUTO`` defers learner
            construction to ``_make_fold_learner()`` at fit time.
        control_outcome_learner (model, optional): a model to estimate outcomes
            in the control group
        treatment_outcome_learner (model, optional): a model to estimate outcomes
            in the treatment group

    Returns:
        (tuple): the resolved (control_outcome_learner, treatment_outcome_learner),
        each either an estimator or the ``_AUTO`` sentinel
    """
    if (control_outcome_learner is None) != (treatment_outcome_learner is None):
        raise ValueError(
            "Specify both `control_outcome_learner` and `treatment_outcome_learner`, "
            "or neither."
        )
    if learner is None and control_outcome_learner is None:
        raise ValueError(
            "Either `learner` or both `control_outcome_learner` and "
            "`treatment_outcome_learner` must be specified."
        )

    def _default():
        return _AUTO if learner is _AUTO else deepcopy(learner)

    _control_outcome_learner = (
        control_outcome_learner if control_outcome_learner is not None else _default()
    )
    _treatment_outcome_learner = (
        treatment_outcome_learner
        if treatment_outcome_learner is not None
        else _default()
    )
    return _control_outcome_learner, _treatment_outcome_learner


def compute_dr_pseudo_outcomes(
    X,
    treatment,
    y,
    p=None,
    learner=_AUTO,
    control_outcome_learner=None,
    treatment_outcome_learner=None,
    n_folds=5,
    p_clip_bounds=(0.02, 0.98),
    random_state=None,
):
    """Construct cross-fitted doubly-robust (AIPW) pseudo-outcomes for CATE evaluation.

    For each unit i, the pseudo-outcome is

        phi_i = (w_i - e(X_i)) / (e(X_i) * (1 - e(X_i))) * (y_i - mu_w(X_i)) + mu_1(X_i) - mu_0(X_i)

    where ``e`` is the propensity score and ``mu_0``/``mu_1`` are the control/treatment
    outcome regressions. Under either correct propensity or correct outcome-model
    specification, ``E[phi_i | X_i]`` is an unbiased estimate of the true CATE
    ``tau(X_i)`` (Kennedy, 2023), which is why ``phi`` can stand in for the unobserved
    ground-truth treatment effect when scoring fitted CATE models.

    Nuisance models (propensity and outcome regressions) are cross-fitted with
    ``n_folds``-fold splitting so that ``phi_i`` is always constructed from models
    that did not see unit i during training. This is the same doubly-robust
    formula used internally by ``BaseDRLearner.fit()``.

    This is a standalone helper so the pseudo-outcomes can be computed once and
    reused across multiple scoring calls -- e.g. passed to ``dr_score()`` directly,
    or to ``rate_score(..., treatment_effect_col=...)`` for RATE on observational
    data -- without re-fitting nuisance models for each.

    Args:
        X (numpy.ndarray or pandas.DataFrame): a feature matrix
        treatment (numpy.ndarray or pandas.Series): a binary treatment indicator (0 or 1)
        y (numpy.ndarray or pandas.Series): an outcome vector
        p (numpy.ndarray or pandas.Series, optional): propensity scores. If None,
            they are estimated in-fold via ``causalml.propensity.compute_propensity_score``
            (``ElasticNetPropensityModel`` by default)
        learner (model, optional): a model used for both control and treatment outcome
            regressions if the group-specific learners below are not given. If
            left as the default, a fresh ``LGBMRegressor`` is built for each
            fold/arm fit with ``num_leaves`` capped to what that fit's row
            count can support (see ``_sized_default_learner()``), rather than
            a single fixed-size model shared across every fit
        control_outcome_learner (model, optional): a model to estimate outcomes
            in the control group
        treatment_outcome_learner (model, optional): a model to estimate outcomes
            in the treatment group
        n_folds (int, optional): number of cross-fitting folds. Default 5.
        p_clip_bounds (tuple, optional): lower and upper bounds for clipping
            propensity scores before they're used as AIPW weights. The default
            ``ElasticNetPropensityModel`` clips to ``(1e-3, 1 - 1e-3)`` internally
            for numerical stability of the model itself, but that's too permissive
            once the score is *inverted* here: a handful of near-boundary,
            cross-fitted propensities (e.g. 0.001, arising from isotonic
            calibration on a single fold) can produce AIPW weights in the
            hundreds and dominate the mean. Tighter trimming bounds the
            variance at the cost of some bias for units with extreme propensity;
            (0.02, 0.98) is a reasonable default for that trade-off. Default
            (0.02, 0.98).
        random_state (int or None, optional): random seed for the fold splitter.
            Default None.

    Returns:
        (numpy.ndarray): the cross-fitted DR pseudo-outcomes, one per row of ``X``
    """
    _control_outcome_learner, _treatment_outcome_learner = _resolve_outcome_learners(
        learner, control_outcome_learner, treatment_outcome_learner
    )

    X = np.asarray(X)
    treatment = np.asarray(treatment)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float) if p is not None else None

    n = X.shape[0]
    phi = np.empty(n, dtype=float)

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    for train_idx, test_idx in cv.split(X, treatment):
        w_train = treatment[train_idx]
        w_test = treatment[test_idx]

        X_train_c = X[train_idx][w_train == 0]
        y_train_c = y[train_idx][w_train == 0]
        mu_c = _make_fold_learner(_control_outcome_learner, X_train_c.shape[0])
        mu_c.fit(X_train_c, y_train_c)

        X_train_t = X[train_idx][w_train == 1]
        y_train_t = y[train_idx][w_train == 1]
        mu_t = _make_fold_learner(_treatment_outcome_learner, X_train_t.shape[0])
        mu_t.fit(X_train_t, y_train_t)

        mu_c_pred = mu_c.predict(X[test_idx])
        mu_t_pred = mu_t.predict(X[test_idx])

        if p is None:
            p_test, _ = compute_propensity_score(
                X=X[train_idx],
                treatment=w_train,
                X_pred=X[test_idx],
                treatment_pred=w_test,
            )
        else:
            p_test = p[test_idx]
        # Trim regardless of source (estimated or user-supplied): see the
        # p_clip_bounds docstring above for why this is tighter than the
        # propensity model's own internal clipping.
        p_test = np.clip(p_test, p_clip_bounds[0], p_clip_bounds[1])

        y_test = y[test_idx]
        mu_w_pred = np.where(w_test == 1, mu_t_pred, mu_c_pred)
        phi[test_idx] = (
            (w_test - p_test) / (p_test * (1 - p_test)) * (y_test - mu_w_pred)
            + mu_t_pred
            - mu_c_pred
        )

    return phi


def _bootstrap_loss_ci(sq_err, score_name, return_ci, n_bootstrap, alpha, random_state):
    """Half-sample bootstrap confidence intervals over a precomputed squared-error frame.

    Shared by dr_score/plug_in_t_score (sq_err = (tau_hat - pseudo_outcome) ** 2)
    and rlearner_score (sq_err = (y_residual - w_residual * tau_hat) ** 2) -- the
    loss formula differs per metric, but once reduced to a per-model squared
    error the bootstrap procedure is identical.

    Lower scores are better in all cases: this is a loss, not a similarity score.

    Args:
        sq_err (pandas.DataFrame): per-row, per-model squared error
        score_name (str): name used for the returned Series/column
        return_ci (bool): whether to return bootstrap confidence intervals
        n_bootstrap (int): number of half-sample bootstrap iterations
        alpha (float): significance level for confidence intervals
        random_state (int or None): random seed for the bootstrap sampler

    Returns:
        If return_ci=False: (pandas.Series): loss for each model column
        If return_ci=True: (pandas.DataFrame): loss, se, and CI bounds per model column
    """
    model_cols = list(sq_err.columns)
    loss = sq_err.mean(axis=0)
    loss.name = score_name

    if not return_ci:
        return loss

    n = len(sq_err)
    m = n // 2
    rng = np.random.default_rng(random_state)
    boot_losses = {model: [] for model in model_cols}
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=m, replace=False)
        boot_loss = sq_err.iloc[idx].mean(axis=0)
        for model in model_cols:
            boot_losses[model].append(boot_loss[model])

    z_crit = stats.norm.ppf(1 - alpha / 2)
    results = []
    for model in model_cols:
        point = loss[model]
        boot = np.array(boot_losses[model])
        se = np.std(boot, ddof=1)
        results.append(
            {
                "model": model,
                score_name: point,
                "se": se,
                "ci_lower": point - z_crit * se,
                "ci_upper": point + z_crit * se,
            }
        )

    return pd.DataFrame(results).set_index("model")


def _score_against_pseudo_outcome(
    df,
    pseudo_outcome,
    model_cols,
    score_name,
    return_ci,
    n_bootstrap,
    alpha,
    random_state,
):
    """MSE-against-pseudo-outcome loss, for dr_score / plug_in_t_score.

    See _bootstrap_loss_ci for the shared bootstrap-CI mechanics.

    Args:
        df (pandas.DataFrame): a data frame with model CATE estimates as columns
        pseudo_outcome (numpy.ndarray): the CATE proxy to score models against
        model_cols (list): the columns of df holding model CATE estimates
        score_name, return_ci, n_bootstrap, alpha, random_state: see
            _bootstrap_loss_ci

    Returns:
        See _bootstrap_loss_ci.
    """
    sq_err = (df[model_cols].sub(pseudo_outcome, axis=0)) ** 2
    return _bootstrap_loss_ci(
        sq_err, score_name, return_ci, n_bootstrap, alpha, random_state
    )


def dr_score(
    df,
    X=None,
    treatment_col="w",
    outcome_col="y",
    pseudo_outcome_col=None,
    p=None,
    learner=_AUTO,
    control_outcome_learner=None,
    treatment_outcome_learner=None,
    n_folds=5,
    p_clip_bounds=(0.02, 0.98),
    return_ci=False,
    n_bootstrap=200,
    alpha=0.05,
    random_state=None,
):
    """Score fitted CATE models via the doubly-robust (DR) pseudo-outcome loss.

    Following Kennedy (2023), this constructs cross-fitted AIPW pseudo-outcomes
    ``phi`` (see ``compute_dr_pseudo_outcomes()``) and scores each candidate CATE
    model by its mean squared error against ``phi``:

        DR loss(tau_hat) = mean((tau_hat(X) - phi) ** 2)

    Lower is better. Unlike held-out outcome MSE, this measures accuracy of the
    *treatment effect* estimate rather than the outcome level, without requiring
    access to counterfactual outcomes. Mahajan et al. (2024) found DR-based
    metrics dominate across 78 benchmark datasets for CATE model selection.

    Pseudo-outcomes can either be supplied directly (via ``pseudo_outcome_col``,
    e.g. computed once with ``compute_dr_pseudo_outcomes()`` and reused across
    multiple scoring calls or shared with ``rate_score()``) or computed internally
    from ``X``, ``treatment_col``, and ``outcome_col``.

    Note:
        Comparing several candidate CATE models in one ``df`` without an
        explicit ``learner``/``pseudo_outcome_col`` re-fits the DR nuisance
        models (``n_folds`` folds x 2 arms) on every call to ``dr_score()``,
        even though the pseudo-outcomes don't depend on which model columns
        are being scored. Precompute them once with
        ``compute_dr_pseudo_outcomes()`` and pass the result via
        ``pseudo_outcome_col`` instead.

    Args:
        df (pandas.DataFrame): a data frame with fitted CATE model estimates as
            columns, plus either ``pseudo_outcome_col`` or both ``outcome_col``
            and ``treatment_col``
        X (numpy.ndarray or pandas.DataFrame, optional): feature matrix used to
            fit the DR nuisance models. Required unless ``pseudo_outcome_col`` is
            already present in ``df``
        treatment_col (str, optional): the column name for the treatment
            indicator (0 or 1). Ignored if ``pseudo_outcome_col`` is provided
        outcome_col (str, optional): the column name for the actual outcome.
            Ignored if ``pseudo_outcome_col`` is provided
        pseudo_outcome_col (str, optional): the column name of pre-computed DR
            pseudo-outcomes (e.g. from ``compute_dr_pseudo_outcomes()``). If given
            and present in ``df``, nuisance models are not re-fit
        p (numpy.ndarray or pandas.Series, optional): propensity scores. Only
            used when pseudo-outcomes are computed internally
        learner (model, optional): a model for both control and treatment outcome
            regressions if the group-specific learners below are not given. If
            left as the default, see ``compute_dr_pseudo_outcomes()`` for how
            the default learner is sized
        control_outcome_learner (model, optional): a model to estimate outcomes
            in the control group
        treatment_outcome_learner (model, optional): a model to estimate outcomes
            in the treatment group
        n_folds (int, optional): number of cross-fitting folds for nuisance
            estimation. Default 5
        p_clip_bounds (tuple, optional): bounds for clipping propensity scores
            used as AIPW weights when pseudo-outcomes are computed internally.
            See ``compute_dr_pseudo_outcomes()`` for why this is tighter than a
            propensity model's own internal clipping. Ignored if
            ``pseudo_outcome_col`` is provided. Default (0.02, 0.98)
        return_ci (bool, optional): whether to return bootstrap confidence
            intervals. Default False
        n_bootstrap (int, optional): number of half-sample bootstrap iterations.
            Only used when return_ci=True. Default 200
        alpha (float, optional): significance level for confidence intervals.
            Only used when return_ci=True. Default 0.05
        random_state (int or None, optional): random seed for cross-fitting and
            the bootstrap sampler. Default None

    Returns:
        If return_ci=False: (pandas.Series): DR loss for each model column (lower is better)
        If return_ci=True: (pandas.DataFrame): DR loss, standard error, and confidence interval bounds for each model column
    """
    have_pseudo_outcome = (
        pseudo_outcome_col is not None and pseudo_outcome_col in df.columns
    )
    assert have_pseudo_outcome or (X is not None), (
        "Either `pseudo_outcome_col` (present in df) or `X` "
        "(to compute pseudo-outcomes internally) must be provided."
    )

    model_cols = [
        c
        for c in df.columns
        if c not in (outcome_col, treatment_col, pseudo_outcome_col)
    ]

    if have_pseudo_outcome:
        pseudo_outcome = df[pseudo_outcome_col].to_numpy()
    else:
        assert (
            outcome_col in df.columns and treatment_col in df.columns
        ), "{} and {} must be present in df to compute DR pseudo-outcomes.".format(
            outcome_col, treatment_col
        )
        pseudo_outcome = compute_dr_pseudo_outcomes(
            X=X,
            treatment=df[treatment_col],
            y=df[outcome_col],
            p=p,
            learner=learner,
            control_outcome_learner=control_outcome_learner,
            treatment_outcome_learner=treatment_outcome_learner,
            n_folds=n_folds,
            p_clip_bounds=p_clip_bounds,
            random_state=random_state,
        )

    return _score_against_pseudo_outcome(
        df=df,
        pseudo_outcome=pseudo_outcome,
        model_cols=model_cols,
        score_name="dr_loss",
        return_ci=return_ci,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )


def plug_in_t_score(
    df,
    X,
    treatment_col="w",
    outcome_col="y",
    learner=_AUTO,
    control_outcome_learner=None,
    treatment_outcome_learner=None,
    n_folds=5,
    return_ci=False,
    n_bootstrap=200,
    alpha=0.05,
    random_state=None,
):
    """Score fitted CATE models against a cross-fitted plug-in T-learner proxy.

    Fits a simple T-learner -- separate control/treatment outcome regressions --
    with ``n_folds``-fold cross-fitting, and uses ``mu_1(X) - mu_0(X)`` on each
    held-out fold as a proxy for the true CATE. Candidate models are then scored
    by mean squared error against this proxy:

        T-loss(tau_hat) = mean((tau_hat(X) - (mu_1(X) - mu_0(X))) ** 2)

    Lower is better. This is a simpler baseline than ``dr_score()`` -- it isn't
    doubly robust and is biased under a misspecified outcome model -- but Mahajan
    et al. (2024) found it is never dominated across their benchmark datasets
    despite its simplicity, making it a useful complement to DR-based scoring
    rather than a replacement.

    Note:
        Comparing several candidate CATE models in one ``df`` without an
        explicit ``learner`` re-fits the plug-in T-learner nuisance models
        (``n_folds`` folds x 2 arms) on every call, even though the proxy
        doesn't depend on which model columns are being scored. If you're
        calling this repeatedly for the same ``X``/``treatment_col``/
        ``outcome_col``, consider computing the proxy once yourself and
        reusing it, the way ``compute_dr_pseudo_outcomes()`` lets you do for
        ``dr_score()``.

    Args:
        df (pandas.DataFrame): a data frame with fitted CATE model estimates as columns
        X (numpy.ndarray or pandas.DataFrame): feature matrix used to fit the
            plug-in T-learner nuisance models
        treatment_col (str, optional): the column name for the treatment
            indicator (0 or 1)
        outcome_col (str, optional): the column name for the actual outcome
        learner (model, optional): a model for both control and treatment outcome
            regressions if the group-specific learners below are not given. If
            left as the default, see ``compute_dr_pseudo_outcomes()`` for how
            the default learner is sized
        control_outcome_learner (model, optional): a model to estimate outcomes
            in the control group
        treatment_outcome_learner (model, optional): a model to estimate outcomes
            in the treatment group
        n_folds (int, optional): number of cross-fitting folds. Default 5
        return_ci (bool, optional): whether to return bootstrap confidence
            intervals. Default False
        n_bootstrap (int, optional): number of half-sample bootstrap iterations.
            Only used when return_ci=True. Default 200
        alpha (float, optional): significance level for confidence intervals.
            Only used when return_ci=True. Default 0.05
        random_state (int or None, optional): random seed for cross-fitting and
            the bootstrap sampler. Default None

    Returns:
        If return_ci=False: (pandas.Series): plug-in T-learner loss for each model column (lower is better)
        If return_ci=True: (pandas.DataFrame): loss, standard error, and confidence interval bounds for each model column
    """
    assert (
        outcome_col in df.columns and treatment_col in df.columns
    ), "{} and {} must be present in df.".format(outcome_col, treatment_col)

    _control_outcome_learner, _treatment_outcome_learner = _resolve_outcome_learners(
        learner, control_outcome_learner, treatment_outcome_learner
    )

    X_arr = np.asarray(X)
    treatment = df[treatment_col].to_numpy()
    y = df[outcome_col].to_numpy(dtype=float)

    n = X_arr.shape[0]
    tau_proxy = np.empty(n, dtype=float)

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    for train_idx, test_idx in cv.split(X_arr, treatment):
        w_train = treatment[train_idx]

        X_train_c = X_arr[train_idx][w_train == 0]
        y_train_c = y[train_idx][w_train == 0]
        mu_c = _make_fold_learner(_control_outcome_learner, X_train_c.shape[0])
        mu_c.fit(X_train_c, y_train_c)

        X_train_t = X_arr[train_idx][w_train == 1]
        y_train_t = y[train_idx][w_train == 1]
        mu_t = _make_fold_learner(_treatment_outcome_learner, X_train_t.shape[0])
        mu_t.fit(X_train_t, y_train_t)

        tau_proxy[test_idx] = mu_t.predict(X_arr[test_idx]) - mu_c.predict(
            X_arr[test_idx]
        )

    model_cols = [c for c in df.columns if c not in (outcome_col, treatment_col)]

    return _score_against_pseudo_outcome(
        df=df,
        pseudo_outcome=tau_proxy,
        model_cols=model_cols,
        score_name="plug_in_t_loss",
        return_ci=return_ci,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )


def rlearner_score(
    df,
    X=None,
    treatment_col="w",
    outcome_col="y",
    y_residual_col=None,
    w_residual_col=None,
    outcome_learner=None,
    propensity_learner=None,
    n_folds=5,
    return_ci=False,
    n_bootstrap=200,
    alpha=0.05,
    random_state=None,
):
    """Score fitted CATE models via the R-loss (Nie & Wager, 2021).

    R-loss(tau_hat) = mean[((y - m(X)) - (w - e(X)) * tau_hat(X)) ** 2]

    where m(X) = E[Y|X] and e(X) = E[W|X] are cross-fitted nuisance regressions
    (see causalml.propensity.compute_r_residuals()). This is the loss
    BaseRLearner.fit() already minimizes internally to train its own effect
    model; exposing it standalone gives R-loss-based comparison across
    arbitrary fitted CATE models -- EconML RScorer parity. Lower is better.

    R-score complements dr_score() and plug_in_t_score() on the CATE-accuracy
    axis (as opposed to rate_score()'s targeting/ranking axis); Mahajan et al.
    (2024) found DR-loss dominates and plug-in-T is never dominated, with
    R-loss not the standout of the three -- useful as a third opinion,
    particularly for parity with EconML workflows already using RScorer.

    Residuals can be supplied directly (e.g. precomputed once with
    compute_r_residuals() and reused across scoring calls) via
    y_residual_col / w_residual_col, or computed internally from X,
    treatment_col, and outcome_col.

    Args:
        df (pandas.DataFrame): a data frame with fitted CATE model estimates as
            columns, plus either y_residual_col/w_residual_col or both
            outcome_col and treatment_col
        X (numpy.ndarray or pandas.DataFrame, optional): feature matrix for the
            R-loss nuisance models. Required unless residual columns are given
        treatment_col (str, optional): treatment indicator column (0 or 1).
            Ignored if residual columns are provided
        outcome_col (str, optional): outcome column. Ignored if residual
            columns are provided
        y_residual_col (str, optional): precomputed y - m_hat(X) column
        w_residual_col (str, optional): precomputed w - e_hat(X) column
        outcome_learner (model, optional): model for E[Y|X]. Required unless
            residual columns are provided
        propensity_learner (PropensityModel, optional): passed to
            compute_r_residuals(). Defaults to ElasticNetPropensityModel
        n_folds (int, optional): cross-fitting folds. Default 5
        return_ci (bool, optional): whether to return bootstrap CIs. Default False
        n_bootstrap (int, optional): half-sample bootstrap iterations. Default 200
        alpha (float, optional): CI significance level. Default 0.05
        random_state (int or None, optional): random seed. Default None

    Returns:
        If return_ci=False: (pandas.Series): R-loss for each model column (lower is better)
        If return_ci=True: (pandas.DataFrame): R-loss, se, and CI bounds per model column
    """
    have_residuals = (
        y_residual_col is not None
        and y_residual_col in df.columns
        and w_residual_col is not None
        and w_residual_col in df.columns
    )
    assert have_residuals or (X is not None and outcome_learner is not None), (
        "Either `y_residual_col`/`w_residual_col` (present in df) or `X` and "
        "`outcome_learner` (to compute residuals internally) must be provided."
    )

    model_cols = [
        c
        for c in df.columns
        if c not in (outcome_col, treatment_col, y_residual_col, w_residual_col)
    ]

    if have_residuals:
        y_residual = df[y_residual_col].to_numpy()
        w_residual = df[w_residual_col].to_numpy()
    else:
        assert (
            outcome_col in df.columns and treatment_col in df.columns
        ), "{} and {} must be present in df to compute R-loss residuals.".format(
            outcome_col, treatment_col
        )
        y_residual, w_residual = compute_r_residuals(
            X=X,
            treatment=df[treatment_col],
            y=df[outcome_col],
            outcome_learner=outcome_learner,
            propensity_learner=propensity_learner,
            n_folds=n_folds,
            random_state=random_state,
        )

    sq_err = (df[model_cols].mul(w_residual, axis=0).sub(y_residual, axis=0)) ** 2
    return _bootstrap_loss_ci(
        sq_err, "r_loss", return_ci, n_bootstrap, alpha, random_state
    )
