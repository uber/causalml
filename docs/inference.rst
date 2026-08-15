==========================
Uncertainty Quantification
==========================

Every number CausalML produces is an estimate, and most of them can carry a
confidence interval. This page catalogs, per estimator family and per metric,
what uncertainty measure is available, how it is computed, and how to request
it. It consolidates an API surface that is otherwise spread across the
individual classes.

Average treatment effects
=========================

**Meta-learners.** ``estimate_ate()`` returns the triple ``(ate, lb, ub)``.
The default interval is analytic, based on the lower-bound formula (7) of
:cite:`imbens2009recent`; passing ``bootstrap_ci=True`` (with ``n_bootstraps``
and ``bootstrap_size``) replaces it with a bootstrap interval. The
:ref:`DRIV learner <methodology:Doubly Robust Instrumental Variable (DRIV) learner>`
follows the same interface.

**TMLE.** ``TMLELearner.estimate_ate(..., return_ci=True)`` reports the ATE
with a confidence interval; :ref:`the methodology section
<methodology:Targeted maximum likelihood estimation (TMLE) for ATE>` describes
the estimator.

**Causal trees.** ``CausalTreeRegressor.estimate_ate()`` also returns
``(ate, lb, ub)``.

CATE estimates
==============

**Meta-learners.** ``fit_predict(..., return_ci=True)`` returns per-unit CATE
estimates with lower and upper bounds from a bootstrap over refits:
``n_bootstraps`` controls the number of refits and ``bootstrap_size`` the
resample size, so the cost is roughly ``n_bootstraps`` times the single fit.

**Causal forests.** ``CausalRandomForestRegressor.calculate_error(X_train,
X_test)`` returns an unbiased sampling variance for each prediction, computed
with the infinitesimal jackknife of `Wager, Hastie and Efron (2014)
<https://arxiv.org/abs/1311.4555>`_ as implemented in `forestci
<https://github.com/scikit-learn-contrib/forest-confidence-interval>`_.

**Uplift trees.** ``UpliftTreeClassifier`` and
``UpliftRandomForestClassifier`` do not report uncertainty for their
estimates.

Evaluation metrics
==================

The evaluation functions accept ``return_ci=True`` and then return a DataFrame
with the score, its bootstrap standard error and confidence interval bounds
per model column (``n_bootstrap``, ``alpha`` and ``random_state`` control the
bootstrap). Pass ``random_state`` whenever the numbers will be reported: the
bootstrap otherwise draws from the global NumPy state and will not reproduce.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Function
     - Uncertainty reported
     - Notes
   * - ``auuc_score``
     - bootstrap SE and CI
     - no p-value by design; compare models by CI overlap
   * - ``qini_score``
     - bootstrap SE, CI and p-value
     - p-value tests ranking better than random
   * - ``rate_score``
     - half-sample bootstrap SE, CI and p-value
     - the heterogeneity test; see :ref:`RATE <methodology:RATE>`
   * - ``dr_score`` / ``plug_in_t_score``
     - bootstrap SE and CI on the loss
     - lower loss is better; see
       :ref:`Model Selection with Validation Losses <validation:Model Selection with Validation Losses>`

What an interval does and does not cover
========================================

These intervals quantify sampling uncertainty under each estimator's
assumptions -- most importantly unconfoundedness. They say nothing about bias
from an unmeasured confounder; that question belongs to
:ref:`sensitivity analysis <validation:Validation with Sensitivity Analysis>`.
And a per-unit CATE interval that excludes zero is not yet evidence of
targetable heterogeneity across units -- that claim needs the held-out tests
in :doc:`validation`.
