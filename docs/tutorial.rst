========================
A First Causal Analysis
========================

This tutorial trains four CATE estimators on the same data and then does the
part that is genuinely hard in causal ML: deciding which of them to believe.
The data is the IHDP benchmark, where the per-unit ground truth is known -- so
the validation methods you would use on real data can themselves be checked
against the right answer. Each step names the User Guide page that covers it
in depth. The dataset downloads on first use and is cached locally (see
:doc:`datasets`).

Step 1: State the question
==========================

The Infant Health and Development Program (IHDP) benchmark
:cite:`hill2011bayesian` starts from a real randomized trial of home visits
for premature infants, with a child's cognitive test score as the outcome. Two
modifications made it the standard testbed for heterogeneous-effect
estimation: a nonrandom subset of the treated group was removed, so treatment
assignment is confounded the way observational data is, and the outcomes are
simulated from the real covariates -- so both potential outcomes, and
therefore every unit's true effect, are known. Each of its 100 replications
simulates new outcomes for the same 747 units and draws its own 672/75
train-test split.

Step 2: Load the data
=====================

.. code-block:: python

    import numpy as np
    import pandas as pd
    from causalml.dataset import fetch_ihdp

    RS = 42
    np.random.seed(RS)

    train = fetch_ihdp(replication=0, split="train")
    test = fetch_ihdp(replication=0, split="test")
    X_tr = pd.DataFrame(train.data, columns=train.feature_names)
    w_tr, y_tr, tau_tr = train.treatment, train.target, train.tau
    X_te = pd.DataFrame(test.data, columns=test.feature_names)
    w_te, y_te, tau_te = test.treatment, test.target, test.tau

    print(X_tr.shape, w_tr.sum(), X_te.shape, w_te.sum())
    print("true ATE: %.3f | sd of tau: %.3f" % (tau_tr.mean(), tau_tr.std()))

.. code-block:: text

    (672, 25) 123 (75, 25) 16
    true ATE: 4.012 | sd of tau: 0.866

672 training units with 123 treated, 25 covariates, and -- because this is a
benchmark -- the true effect of every unit, held aside for scoring.

Step 3: Check overlap
=====================

Estimation needs treated and untreated units throughout the covariate space
(see :ref:`Checking Overlap <validation:Checking Overlap>`). Estimate each
unit's probability of treatment -- the propensity score -- and compare its
distribution across groups. One practical note: on this data the model's
default cross-validation grid selects a penalty that collapses every score to
the treated share, so the grid is widened explicitly.

.. code-block:: python

    from causalml.propensity import ElasticNetPropensityModel

    pm = ElasticNetPropensityModel(Cs=np.logspace(0, 3, 8), random_state=RS)
    pm.fit(X_tr, w_tr)
    p_tr, p_te = pm.predict(X_tr), pm.predict(X_te)

.. image:: ./_static/img/tutorial_overlap.png
    :width: 629
    :alt: Propensity-score distributions with the treatment group shifted right and a mass of control units near zero.

The two distributions overlap over most of the range -- estimation is possible
-- but they are far from identical: the confounding introduced by removing
part of the treated group is exactly what this picture shows, and a spike of
near-zero-propensity controls marks a region with almost no treated
counterparts. On a randomized experiment this plot is flat.

Step 4: Train four estimators
=============================

Four meta-learners (see :doc:`methodology`), all wrapping the same
gradient-boosted base learner so the comparison is about the learner
*strategies*. The X- and R-learners consume the propensity score:

.. code-block:: python

    from xgboost import XGBRegressor
    from causalml.inference.meta import (
        BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor,
    )

    base = lambda: XGBRegressor(random_state=RS)
    learners = {
        "S-learner": BaseSRegressor(learner=base()),
        "T-learner": BaseTRegressor(learner=base()),
        "X-learner": BaseXRegressor(learner=base()),
        "R-learner": BaseRRegressor(learner=base()),
    }

    print("ATE (truth: %.3f):" % tau_tr.mean())
    for name, m in learners.items():
        if name in ("X-learner", "R-learner"):
            ate, lb, ub = m.estimate_ate(X=X_tr, treatment=w_tr, y=y_tr, p=p_tr)
        elif name == "S-learner":
            ate, lb, ub = m.estimate_ate(X=X_tr, treatment=w_tr, y=y_tr, return_ci=True)
        else:
            ate, lb, ub = m.estimate_ate(X=X_tr, treatment=w_tr, y=y_tr)
        print("  %-10s %.2f (%.2f, %.2f)" % (name, ate, lb, ub))

.. code-block:: text

    ATE (truth: 4.012):
      S-learner  3.88 (3.79, 3.98)
      T-learner  3.96 (3.84, 4.08)
      X-learner  4.16 (4.07, 4.24)
      R-learner  4.18 (4.16, 4.19)

All four land near the truth (how the intervals are computed is cataloged in
:doc:`inference`). Note the R-learner: the narrowest interval of the four, and
the only one that excludes the true value. Precision is not accuracy, and
nothing on this table says which estimator to trust -- that takes the next two
steps.

Step 5: Evaluate against the ground truth
=========================================

Predict each unit's effect on the held-out split and score it against the
truth: PEHE (precision in estimating heterogeneous effects -- the root mean
squared error of the per-unit estimates) and :func:`~causalml.metrics.ate_error`:

.. code-block:: python

    from causalml.metrics import pehe, ate_error

    cate_te = {}
    for name, m in learners.items():
        if name == "X-learner":
            cate_te[name] = m.predict(X=X_te, p=p_te).flatten()
        else:
            cate_te[name] = m.predict(X=X_te).flatten()

    for name, c in cate_te.items():
        print("  %-10s PEHE %.3f  ate_error %+.3f"
              % (name, pehe(tau_te, c, squared=False), ate_error(tau_te, c)))

.. code-block:: text

      S-learner  PEHE 0.722  ate_error +0.044
      T-learner  PEHE 0.994  ate_error +0.019
      X-learner  PEHE 0.883  ate_error +0.164
      R-learner  PEHE 2.855  ate_error +1.329

With ground truth, evaluation is just measurement: the R-learner -- with this
base learner and these defaults -- is failing on both metrics, and the other
three are close to each other. On real data there is no such measurement,
which is the situation the next step simulates.

Step 6: Validate as if the truth were unknown
=============================================

Everything in this step uses only what real data provides: covariates,
treatment, outcome, and the models' predictions. The validation losses score
each model's predictions against a proxy for the true effect built by
cross-fitting on the held-out data -- the doubly robust (DR) pseudo-outcome
loss and the plug-in T-learner loss (see
:ref:`Model Selection with Validation Losses <validation:Model Selection with Validation Losses>`):

.. code-block:: python

    from causalml.metrics import dr_score, plug_in_t_score, rate_score

    df = pd.DataFrame({"y": y_te, "w": w_te, **cate_te})
    print(dr_score(df, X=X_te, outcome_col="y", treatment_col="w", p=p_te,
                   learner=XGBRegressor(random_state=RS),
                   return_ci=True, random_state=RS).round(3))
    print(plug_in_t_score(df, X=X_te, outcome_col="y", treatment_col="w",
                          learner=XGBRegressor(random_state=RS),
                          return_ci=True, random_state=RS).round(3))

.. code-block:: text

               dr_loss      se  ci_lower  ci_upper
    model
    S-learner   39.307  23.815    -7.369    85.984
    T-learner   40.219  23.369    -5.583    86.021
    X-learner   37.947  22.495    -6.142    82.036
    R-learner   44.041  20.881     3.115    84.968

               plug_in_t_loss     se  ci_lower  ci_upper
    model
    S-learner           2.247  0.635     1.003     3.491
    T-learner           2.967  0.855     1.292     4.642
    X-learner           1.773  0.272     1.241     2.305
    R-learner           9.281  1.517     6.307    12.255

Both losses, knowing nothing of the truth, reproduce its verdict: the
R-learner is worst by a wide margin, and the other three sit within each
other's uncertainty. (They rank the X-learner first where the truth ranks the
S-learner first -- differences inside the top group are within the intervals,
and the losses cannot resolve them. What they reliably do is catch the failing
model.)

The ranking metrics tell a different story about sample size:

.. code-block:: python

    print(rate_score(df, outcome_col="y", treatment_col="w",
                     return_ci=True, random_state=RS).round(3))

.. code-block:: text

                rate     se  ci_lower  ci_upper  p_value
    model
    S-learner  0.316  0.230    -0.134     0.767    0.168
    T-learner -0.197  0.170    -0.530     0.136    0.246
    X-learner -0.166  0.191    -0.541     0.208    0.385
    R-learner  0.283  0.225    -0.157     0.724    0.208

Every :ref:`RATE <methodology:RATE>` interval spans zero (the Qini scores look
the same): 75 validation rows carry too little information for rank-based
metrics to separate anything, even models the losses separate cleanly. On data
this small, lean on the validation losses. The next step gives the ranking
metrics the sample size they need.

Step 7: Visualize the ranking with gain and TOC curves
======================================================

To see what the ranking metrics measure, pool the replication's 747 units and
re-split them in half, so the validation side has 374 rows instead of 75, and
refit the four learners on the other half:

.. code-block:: python

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from causalml.metrics import auuc_score, plot_gain, plot_toc

    X_all = pd.DataFrame(np.vstack([train.data, test.data]),
                         columns=train.feature_names)
    w_all = np.concatenate([train.treatment, test.treatment])
    y_all = np.concatenate([train.target, test.target])

    fit_idx, val_idx = train_test_split(np.arange(len(y_all)), test_size=0.5,
                                        random_state=RS, stratify=w_all)
    pm_v = ElasticNetPropensityModel(Cs=np.logspace(0, 3, 8), random_state=RS)
    pm_v.fit(X_all.iloc[fit_idx], w_all[fit_idx])
    p_fit = pm_v.predict(X_all.iloc[fit_idx])
    p_val = pm_v.predict(X_all.iloc[val_idx])

    cate_val = {}
    for name, m in learners.items():
        if name in ("X-learner", "R-learner"):
            m.fit(X=X_all.iloc[fit_idx], treatment=w_all[fit_idx],
                  y=y_all[fit_idx], p=p_fit)
        else:
            m.fit(X=X_all.iloc[fit_idx], treatment=w_all[fit_idx],
                  y=y_all[fit_idx])
        c = (m.predict(X=X_all.iloc[val_idx], p=p_val) if name == "X-learner"
             else m.predict(X=X_all.iloc[val_idx]))
        cate_val[name] = c.flatten()

    df_val = pd.DataFrame({"y": y_all[val_idx], "w": w_all[val_idx], **cate_val})
    print(auuc_score(df_val, outcome_col="y", treatment_col="w",
                     return_ci=True, random_state=RS).round(3))

.. code-block:: text

                auuc     se  ci_lower  ci_upper
    model
    S-learner  0.535  0.013     0.510     0.560
    T-learner  0.529  0.013     0.505     0.554
    X-learner  0.527  0.014     0.500     0.554
    R-learner  0.530  0.013     0.504     0.555

The AUUC (area under the cumulative gain curve, normalized so 0.5 is random
targeting) sits a hair above 0.5 for all four models, and the gain curves show
why:

.. code-block:: python

    fig, ax = plt.subplots(figsize=(7, 4.2))
    plot_gain(df_val, outcome_col="y", treatment_col="w", ax=ax)

.. image:: ./_static/img/tutorial_gain.png
    :width: 629
    :alt: Cumulative gain curves for the four learners sitting just above the random diagonal.

Every curve hugs the random diagonal, because most units share a similar
effect: ranking cannot beat random by much when there is little spread to
exploit. Whatever advantage exists is concentrated in the top-ranked few
percent, which is exactly what the TOC curve isolates -- TOC(q) is the excess
effect among the top-q fraction over the overall ATE:

.. code-block:: python

    fig, ax = plt.subplots(figsize=(7, 4.2))
    plot_toc(df_val, outcome_col="y", treatment_col="w", ax=ax)

.. image:: ./_static/img/tutorial_toc.png
    :width: 629
    :alt: TOC curves spiking to about 1.5 in the top few percent of ranked units and decaying toward zero.

The units the models rank highest show an excess effect of about 1.5 on top of
the overall 4.0 -- a real, if narrow, targeting opportunity that the gain
curve's scale hides. :ref:`RATE <methodology:RATE>` with its default
``autoc`` weighting integrates the TOC with weight :math:`1/q`, emphasizing
precisely this region:

.. code-block:: python

    print(rate_score(df_val, outcome_col="y", treatment_col="w",
                     return_ci=True, random_state=RS).round(3))

.. code-block:: text

                rate     se  ci_lower  ci_upper  p_value
    model
    S-learner  0.437  0.294    -0.139     1.012    0.137
    T-learner  0.302  0.274    -0.236     0.840    0.272
    X-learner  0.381  0.191     0.006     0.757    0.046
    R-learner  0.196  0.231    -0.256     0.648    0.395

With five times the validation rows of Step 6, every point estimate is now
positive and the X-learner's RATE excludes zero (p = 0.046): its
prioritization demonstrably beats treating at random, a conclusion AUUC could
not reach because the signal lives in a region AUUC weights no more than any
other. This division of labor -- gain curves for the overall picture, TOC and
RATE for concentrated heterogeneity -- is the reason both exist.

Step 8: Stress the assumptions
==============================

Every estimate above leans on unconfoundedness: that the 25 covariates capture
everything driving both treatment and outcome. Sensitivity analysis (see
:ref:`Validation with Sensitivity Analysis <validation:Validation with Sensitivity Analysis>`)
perturbs the analysis and re-estimates. Replacing the treatment with random
noise -- the placebo test -- should destroy the effect; adding a random
covariate or halving the sample should not:

.. code-block:: python

    from causalml.metrics.sensitivity import Sensitivity

    df_s = X_tr.assign(treatment=w_tr, outcome=y_tr, p=p_tr)
    sens = Sensitivity(df=df_s, inference_features=list(X_tr.columns),
                       p_col="p", treatment_col="treatment",
                       outcome_col="outcome",
                       learner=BaseXRegressor(learner=XGBRegressor(random_state=RS)))
    print(sens.sensitivity_analysis(
        methods=["Placebo Treatment", "Random Cause", "Subset Data"],
        sample_size=0.5).to_string())

.. code-block:: text

                              Method     ATE   New ATE  New ATE LB  New ATE UB
    0              Placebo Treatment  4.1561  0.169624    0.059923    0.279326
    1                   Random Cause  4.1561  4.050544    3.969506    4.131583
    2  Subset Data(sample size @0.5)  4.1561  4.073714    3.953690    4.193737

The placebo collapses the estimate by 96% -- not exactly to zero, since a
flexible learner finds some structure even in noise, but to the far side of
negligible -- while the other two perturbations barely move it. The analysis
is behaving the way a real effect should.

Step 9: The replication protocol
================================

IHDP results are published as a mean and standard error *across* replications
-- a single replication is not comparable to a published number. The loop is
the unit of comparison:

.. code-block:: python

    rows = []
    for rep in range(10):
        train = fetch_ihdp(replication=rep, split="train")
        test = fetch_ihdp(replication=rep, split="test")
        ...  # refit the four learners, score PEHE on the test split
    print(pd.DataFrame(rows).groupby("model")["pehe"].agg(["mean", "sem"]).round(3))

.. code-block:: text

                mean    sem
    model
    R-learner  4.876  1.858
    S-learner  3.298  2.068
    T-learner  3.286  2.001
    X-learner  3.524  2.031

The means dwarf the replication-0 numbers because a few replications simulate
heavy-tailed outcomes that dominate the average -- and with standard errors
this size, ten replications establish no significant differences. The
:doc:`benchmark leaderboard <examples/benchmark_leaderboard>` is the canonical
version of this loop; published tables run all 100 replications.

Where to next
=============

* Which estimator fits your problem: :doc:`choosing_an_estimator`
* The mathematics of each method: :doc:`methodology`
* The full evaluation workflow: :doc:`validation`
* What uncertainty each estimator reports: :doc:`inference`
* Interpreting a fitted model: :doc:`interpretation`
