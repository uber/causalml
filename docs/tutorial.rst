========================
A First Causal Analysis
========================

This tutorial walks one real analysis end to end, on a dataset where the
per-unit ground truth is known -- so every estimate below can be checked
against the right answer, including the heterogeneous ones. Each step names
the User Guide page that covers it in depth. The dataset downloads on first
use and is cached locally (see :doc:`datasets`).

Step 1: State the question
==========================

The Twins benchmark :cite:`louizos2017causal` covers 11,400 same-sex twin
births from US birth records. The treatment is being the heavier twin of the
pair and the outcome is one-year mortality. Because *both* twins are observed,
both potential outcomes are measured rather than simulated: the true effect
for every pair, and therefore the true average effect, is known. The question:
what is the effect of higher birth weight on survival, and does it vary across
pairs?

Step 2: Load the data
=====================

.. code-block:: python

    import numpy as np
    import pandas as pd
    from causalml.dataset import fetch_twins

    RS = 42
    np.random.seed(RS)

    data = fetch_twins(random_state=RS)
    X = pd.DataFrame(data.data, columns=data.feature_names)
    treatment = np.asarray(data.treatment)
    y = 1 - np.asarray(data.target)   # survival = 1 - mortality
    tau_true = -np.asarray(data.tau)  # effect on survival

    print(X.shape, treatment.sum())
    print("true ATE: %.4f" % tau_true.mean())
    print("diff in means: %.4f" % (y[treatment == 1].mean() - y[treatment == 0].mean()))

.. code-block:: text

    (11400, 30) 5640
    true ATE: 0.0161
    diff in means: 0.0189

The recorded outcome is mortality; the code models survival so that a positive
effect is a benefit, which is the orientation the ranking metrics in Step 6
assume. The lighter twin's one-year mortality is 17.8%, matching the figure
reported by :cite:`louizos2017causal`, and being the heavier twin raises
survival by 1.61 percentage points on average. The loader reveals one twin per
pair by a fair coin (``random_state`` fixes the coin), so the difference in
means, 1.89 points, is a valid but noisy estimate of that truth.

Step 3: Check overlap
=====================

Estimation needs both treated and untreated units throughout the covariate
space (see :ref:`Checking Overlap <validation:Checking Overlap>`). Estimate
each unit's probability of treatment -- the propensity score -- and compare
its distribution across groups:

.. code-block:: python

    from causalml.propensity import ElasticNetPropensityModel

    pm = ElasticNetPropensityModel(random_state=RS)
    p_hat = pm.fit_predict(X, treatment)
    print("%.3f - %.3f" % (p_hat.min(), p_hat.max()))

.. code-block:: text

    0.495 - 0.495

.. image:: ./_static/img/tutorial_overlap.png
    :width: 629
    :alt: Propensity scores concentrated at a single value for both twins, as expected under randomized assignment.

The estimated propensity is the same for every unit: no covariate predicts
which twin was revealed, exactly what a coin flip should produce. On
observational data this picture instead shows two separated distributions, and
this step is where problems announce themselves. Because the assignment
probability is known here, the estimators below receive it directly:

.. code-block:: python

    p = np.full(len(y), treatment.mean())

Step 4: Estimate the average effect
===================================

Start with a transparent baseline -- a :ref:`T-Learner <methodology:T-Learner>`
with linear regression, which fits one regression per group and differences
the predictions -- then a more flexible :ref:`X-Learner
<methodology:X-Learner>` with gradient-boosted trees. (The outcome is binary;
the ``*Classifier`` learner variants accept classification base learners, but
a regressor on a binary outcome estimates the same risk difference and keeps
this tutorial short.)

.. code-block:: python

    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from causalml.inference.meta import BaseTRegressor, BaseXRegressor

    tl = BaseTRegressor(learner=LinearRegression())
    print(tl.estimate_ate(X=X, treatment=treatment, y=y))

    xl = BaseXRegressor(learner=XGBRegressor(random_state=RS))
    print(xl.estimate_ate(X=X, treatment=treatment, y=y, p=p))

.. code-block:: text

    (array([0.01767856]), array([0.00610093]), array([0.02925618]))
    (array([0.01737203]), array([0.01086719]), array([0.02387686]))

Each call returns the ATE with a confidence interval (how these are computed
is cataloged in :doc:`inference`). Both intervals bracket the true value of
0.0161 -- on most datasets there is no such number to check against, which is
what the :doc:`validation <validation>` machinery is for.

Step 5: Estimate heterogeneous effects
======================================

The same fitted X-learner produces a per-unit CATE estimate:

.. code-block:: python

    cate = xl.fit_predict(X=X, treatment=treatment, y=y, p=p).flatten()
    print("mean %.4f sd %.4f" % (cate.mean(), cate.std()))

.. code-block:: text

    mean 0.0174 sd 0.1495

The spread is nine times the average effect -- but spread in the *estimates*
is not evidence of real heterogeneity. Whether any of it is signal is the next
step's question.

Step 6: Evaluate against the ground truth
=========================================

Refit on a training split and evaluate the held-out split. With ground truth
available, PEHE (the root mean squared error of the per-unit effect estimates)
and ``ate_error`` measure accuracy directly; the Qini coefficient and
:ref:`RATE <methodology:RATE>` are the tests available on real data, where no
truth exists (see :doc:`validation`):

.. code-block:: python

    from sklearn.model_selection import train_test_split
    from causalml.metrics import pehe, ate_error, qini_score, rate_score

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, random_state=RS,
                              stratify=treatment)
    p_tr = np.full(len(tr), treatment[tr].mean())
    p_te = np.full(len(te), treatment[tr].mean())

    xl_ho = BaseXRegressor(learner=XGBRegressor(random_state=RS))
    xl_ho.fit(X=X.iloc[tr], treatment=treatment[tr], y=y[tr], p=p_tr)
    cate_te = xl_ho.predict(X=X.iloc[te], p=p_te).flatten()

    print("PEHE: %.4f" % pehe(tau_true[te], cate_te, squared=False))
    print("PEHE, predicting zero: %.4f" % pehe(tau_true[te], np.zeros(len(te)), squared=False))
    print("ate_error: %.4f" % ate_error(tau_true[te], cate_te))

    df = pd.DataFrame({"y": y[te], "w": treatment[te], "X-learner": cate_te})
    print(rate_score(df, outcome_col="y", treatment_col="w",
                     return_ci=True, random_state=RS))

.. code-block:: text

    PEHE: 0.3540
    PEHE, predicting zero: 0.3231
    ate_error: 0.0056

                   rate       se  ci_lower  ci_upper   p_value
    model
    X-learner -0.246639  0.08943 -0.421918  -0.07136  0.005817

Both verdicts are bad, and they agree. The model's PEHE is *worse* than
predicting zero effect for everyone, and its RATE is significantly
*negative*: units it ranks as high-benefit actually gained less than average.
(The absolute PEHE level is dominated by irreducible noise -- each true
``tau`` is a difference of two binary outcomes, so values of :math:`\pm 1` are
common and no estimator can score far below the zero baseline here; PEHE is
read comparatively.) The diagnosis is overfitting: a flexible learner fit the
individual noise. The remedy is regularization, chosen exactly as
:ref:`Model Selection with Validation Losses <validation:Model Selection with Validation Losses>`
prescribes:

.. code-block:: python

    xl_reg = BaseXRegressor(learner=XGBRegressor(
        max_depth=2, n_estimators=100, learning_rate=0.05,
        min_child_weight=100, random_state=RS))
    xl_reg.fit(X=X.iloc[tr], treatment=treatment[tr], y=y[tr], p=p_tr)
    cate_reg = xl_reg.predict(X=X.iloc[te], p=p_te).flatten()

    print("PEHE: %.4f" % pehe(tau_true[te], cate_reg, squared=False))
    print("ate_error: %.4f" % ate_error(tau_true[te], cate_reg))
    df["X-reg"] = cate_reg
    print(rate_score(df[["y", "w", "X-reg"]], outcome_col="y",
                     treatment_col="w", return_ci=True, random_state=RS))

.. code-block:: text

    PEHE: 0.3233
    ate_error: 0.0016

               rate        se  ci_lower  ci_upper  p_value
    model
    X-reg  0.027949  0.084229 -0.13714  0.193038  0.74096

The regularized model stops adding noise (PEHE at the zero baseline, a 3x
smaller ATE error) and its RATE is indistinguishable from zero. The honest
conclusion: higher birth weight raises survival by about 1.6 percentage
points, and neither model finds heterogeneity in that effect that survives a
held-out test -- a common and publishable finding, not a failure.

Step 7: Interpret the model
===========================

Permutation importance shows which covariates the CATE model relies on (see
:doc:`interpretation`):

.. code-block:: python

    print(xl.get_importance(X=X, tau=cate, method="permutation",
                            features=X.columns, random_state=RS).head(4))

.. code-block:: text

    gestat      0.376208
    wtgain      0.259032
    dmage       0.125790
    nprevist    0.120816
    dtype: float64

Gestation length, maternal weight gain, maternal age and the number of
prenatal visits drive the model's estimates. Read this descriptively: Step 6
found no validated heterogeneity, so these are properties of the fitted model,
not established effect modifiers.

Step 8: Stress the assumptions
==============================

Sensitivity analysis (see
:ref:`Validation with Sensitivity Analysis <validation:Validation with Sensitivity Analysis>`)
perturbs the analysis and re-estimates. Replacing the treatment with random
noise -- the placebo test -- should destroy the effect:

.. code-block:: python

    from causalml.metrics.sensitivity import Sensitivity

    df_s = X.assign(treatment=treatment, outcome=y, p=p)
    sens = Sensitivity(df=df_s, inference_features=list(X.columns),
                       p_col="p", treatment_col="treatment",
                       outcome_col="outcome",
                       learner=BaseXRegressor(learner=XGBRegressor(random_state=RS)))
    print(sens.sensitivity_analysis(
        methods=["Placebo Treatment", "Random Cause", "Subset Data"],
        sample_size=0.5).to_string())

.. code-block:: text

                              Method       ATE   New ATE  New ATE LB  New ATE UB
    0              Placebo Treatment  0.017372 -0.000768   -0.007422    0.005885
    1                   Random Cause  0.017372  0.015239    0.009209    0.021269
    2  Subset Data(sample size @0.5)  0.017372  0.011045    0.003613    0.018476

The placebo estimate is centered on zero, and neither adding a random
covariate nor halving the sample changes the conclusion.

Where to next
=============

* Which estimator fits your problem: :doc:`choosing_an_estimator`
* The mathematics of each method: :doc:`methodology`
* The full evaluation workflow: :doc:`validation`
* What uncertainty each estimator reports: :doc:`inference`
