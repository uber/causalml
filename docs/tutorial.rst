========================
A First Causal Analysis
========================

This tutorial walks one real analysis end to end: estimating the effect of a
job-training program on earnings, on data where the right answer is known. Each
step names the User Guide page that covers it in depth. The dataset downloads
on first use and is cached locally (see :doc:`datasets`).

Because the data come from a randomized experiment, the simple
difference in means is a valid benchmark -- which is exactly what makes this a
good first dataset: every estimate below can be checked against it.

Step 1: State the question
==========================

The National Supported Work (NSW) demonstration randomly assigned
disadvantaged workers to a job-training program. The question: what was the
effect of training (the treatment) on 1978 earnings (the outcome)? The
experimental difference in means, about $1,794, is the number an estimator
should recover :cite:`lalonde1986evaluating`.

Step 2: Load the data
=====================

.. code-block:: python

    import numpy as np
    import pandas as pd
    from causalml.dataset import fetch_lalonde

    RS = 42
    np.random.seed(RS)

    data = fetch_lalonde()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    treatment = np.asarray(data.treatment)
    y = np.asarray(data.target)

    print(X.shape, treatment.sum())
    print(y[treatment == 1].mean() - y[treatment == 0].mean())

.. code-block:: text

    (445, 8) 185
    1794.3

445 workers, 185 treated; the covariates are age, education, race/ethnicity
indicators, marital status, a no-degree indicator, and earnings in 1974 and
1975.

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

.. image:: ./_static/img/tutorial_overlap.png
    :width: 629
    :alt: Overlapping propensity-score histograms for the treatment and control groups.

The two distributions overlap over their whole range, as they should in a
randomized experiment. Because the assignment probability is known here, the
estimators below receive it directly rather than an estimate of it:

.. code-block:: python

    p = np.full(len(y), treatment.mean())

Step 4: Estimate the average effect
===================================

Start with a transparent baseline -- a :ref:`T-Learner <methodology:T-Learner>`
with linear regression, which fits one regression per group and differences the
predictions -- then a more flexible :ref:`X-Learner <methodology:X-Learner>`
with gradient-boosted trees:

.. code-block:: python

    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from causalml.inference.meta import BaseTRegressor, BaseXRegressor

    tl = BaseTRegressor(learner=LinearRegression())
    print(tl.estimate_ate(X=X, treatment=treatment, y=y))

    xl = BaseXRegressor(learner=XGBRegressor(random_state=RS))
    print(xl.estimate_ate(X=X, treatment=treatment, y=y, p=p))

.. code-block:: text

    (array([1622.06]), array([335.90]), array([2908.22]))
    (array([1865.90]), array([1138.36]), array([2593.43]))

Each call returns the ATE with a confidence interval (how these are computed
is cataloged in :doc:`inference`). Both intervals bracket the experimental
benchmark of $1,794.

Step 5: Estimate heterogeneous effects
======================================

The same fitted X-learner produces a per-unit CATE estimate:

.. code-block:: python

    cate = xl.fit_predict(X=X, treatment=treatment, y=y, p=p)
    print(cate.mean(), cate.std())

.. code-block:: text

    1865.9 6119.3

The spread is large relative to the mean -- but spread in the *estimates* is
not evidence of real heterogeneity. Whether any of it is signal is a question
for held-out evaluation.

Step 6: Evaluate on held-out data
=================================

Refit on a training split and score the held-out split with the Qini
coefficient and :ref:`RATE <methodology:RATE>` (see :doc:`validation` for the
full evaluation workflow):

.. code-block:: python

    from sklearn.model_selection import train_test_split
    from causalml.metrics import qini_score, rate_score

    X_tr, X_te, w_tr, w_te, y_tr, y_te = train_test_split(
        X, treatment, y, test_size=0.3, random_state=RS, stratify=treatment
    )
    xl_ho = BaseXRegressor(learner=XGBRegressor(random_state=RS))
    xl_ho.fit(X=X_tr, treatment=w_tr, y=y_tr, p=np.full(len(y_tr), w_tr.mean()))
    cate_te = xl_ho.predict(X=X_te, p=np.full(len(y_te), w_tr.mean())).flatten()

    df = pd.DataFrame({"y": y_te, "w": w_te, "X-learner": cate_te})
    print(qini_score(df, outcome_col="y", treatment_col="w",
                     return_ci=True, random_state=RS))
    print(rate_score(df, outcome_col="y", treatment_col="w",
                     return_ci=True, random_state=RS))

.. code-block:: text

                   qini        se  ci_lower  ci_upper   p_value
    model
    X-learner -0.648355  1.480567 -3.550212  2.253503  0.661452

                     rate           se     ci_lower     ci_upper   p_value
    model
    X-learner  247.592902  1033.931236 -1778.875084  2274.060888  0.810743

Both confidence intervals include zero: on 134 held-out units there is no
evidence that the model's prioritization beats treating at random. That is a
legitimate and common finding -- at this sample size, an insignificant result
can mean low power rather than no heterogeneity -- and it says the CATE
estimates from Step 5 should not drive targeting decisions on their own.

Step 7: Interpret the model
===========================

Permutation importance shows which covariates the CATE model relies on (see
:doc:`interpretation`):

.. code-block:: python

    print(xl.get_importance(X=X, tau=cate, method="permutation",
                            features=X.columns, random_state=RS))

.. code-block:: text

    age         0.296605
    re75        0.215293
    educ        0.180431
    re74        0.167443
    marr        0.025297
    black       0.023638
    hisp        0.004632
    nodegree    0.000000

Age and prior earnings drive the model's estimates. Read this descriptively:
Step 6 found no validated heterogeneity, so these are properties of the fitted
model, not established effect modifiers.

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

                              Method          ATE      New ATE  New ATE LB   New ATE UB
    0              Placebo Treatment  1865.895497   747.669383  -12.269935  1507.608700
    1                   Random Cause  1865.895497  1284.882820  747.662586  1822.103053
    2  Subset Data(sample size @0.5)  1865.895497  2088.609120  976.685229  3200.533011

The placebo interval includes zero, and adding a random covariate or halving
the sample moves the estimate without changing the conclusion. At n=445 these
checks are coarse; on larger data they tighten accordingly.

Where to next
=============

* Which estimator fits your problem: :doc:`choosing_an_estimator`
* The mathematics of each method: :doc:`methodology`
* The full evaluation workflow: :doc:`validation`
* What uncertainty each estimator reports: :doc:`inference`
