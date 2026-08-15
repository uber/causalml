=====================
Choosing an Estimator
=====================

CausalML implements many estimators because no single one dominates: they
differ in the outcome and treatment types they accept, the data they are
designed for, and what they report. This page maps a problem to a shortlist.
The mathematics of each method lives in the :doc:`methodology`; the API details
live in the :doc:`API Reference <causalml>`.

Start from the data
===================

**Was the treatment randomized?**

* **Yes, and compliance was perfect.** Any estimator below applies, and the
  assignment probability is known -- pass it as ``p`` instead of estimating it.
  For a binary conversion outcome where the goal is targeting segments or
  interpretable rules, start with the uplift trees. For per-unit CATE
  estimates with confidence intervals, start with the meta-learners.

* **Yes, but some units did not comply.** The randomized assignment is an
  instrument for the treatment actually received. Use the
  :ref:`DRIV learner <methodology:Doubly Robust Instrumental Variable (DRIV) learner>`
  (``BaseDRIVLearner``) to estimate the effect on compliers, or
  :ref:`2SLS <methodology:2-Stage Least Squares (2SLS)>` for a linear model.

* **No -- the data are observational.** Estimation requires that every
  confounder (a variable driving both treatment and outcome) is measured, and
  that treated and untreated units overlap (see
  :ref:`Checking Overlap <validation:Checking Overlap>`). Prefer the
  estimators that model the treatment assignment explicitly: the
  :ref:`X-Learner <methodology:X-Learner>`,
  :ref:`R-Learner <methodology:R-Learner>` and
  :ref:`DR learner <methodology:Doubly Robust (DR) learner>` all accept a
  propensity score ``p`` and estimate one internally when it is omitted.
  Validate with :doc:`sensitivity analysis <validation>` afterwards.

* **No, and an important confounder is unmeasured.** With an instrument, use
  the IV estimators above. With proxy variables for the hidden confounder,
  :ref:`CEVAE <methodology:CEVAE>` models it as a latent variable. Without
  either, no estimator in this package (or any other) identifies the effect.

**What do you need out of the model?**

* **Only the average effect (ATE)** -- :ref:`TMLE
  <methodology:Targeted maximum likelihood estimation (TMLE) for ATE>`,
  :ref:`IPTW <methodology:Inverse probability of treatment weighting>`,
  :ref:`matching <methodology:Matching>`, or any meta-learner's
  ``estimate_ate()``.
* **Per-unit effects (CATE)** -- meta-learners, causal trees and forests, or
  the neural models.
* **Segments and rules you can read** -- uplift trees, with
  :ref:`visualization <interpretation:Uplift Tree Visualization>`.
* **Who to treat under constraints** -- estimate CATE first, then use
  ``PolicyLearner`` or the
  :ref:`value optimization methods <methodology:Value optimization methods>`.

Capability matrix
=================

.. list-table::
   :header-rows: 1
   :widths: 26 17 15 16 16 10

   * - Estimator (classes)
     - Outcome type
     - Treatment
     - Observational data
     - Uncertainty
     - Extra install
   * - S/T/X/R meta-learners (``BaseSRegressor`` ... ``BaseRClassifier``)
     - continuous (``*Regressor``) or binary (``*Classifier``)
     - binary or multiple discrete
     - yes; X/R use a propensity score
     - ATE CI; bootstrap CATE CI
     - --
   * - DR learner (``BaseDRRegressor``/``BaseDRClassifier``)
     - continuous or binary
     - binary or multiple discrete
     - yes, doubly robust
     - ATE CI; bootstrap CATE CI
     - --
   * - DRIV learner (``BaseDRIVLearner``)
     - continuous
     - binary, with an instrument
     - yes, given an instrument
     - ATE CI; bootstrap CATE CI
     - --
   * - Uplift trees (``UpliftTreeClassifier``, ``UpliftRandomForestClassifier``)
     - binary or multi-class
     - binary or multiple discrete
     - designed for randomized data
     - --
     - --
   * - Causal trees (``CausalTreeRegressor``, ``CausalRandomForestRegressor``)
     - continuous
     - binary
     - yes
     - ATE CI (tree); per-prediction variance (forest)
     - --
   * - ``DragonNet``
     - continuous or binary
     - binary
     - yes
     - --
     - ``tf`` or ``jax``
   * - ``CEVAE``
     - continuous or binary
     - binary
     - yes, with proxies for a hidden confounder
     - --
     - ``torch`` or ``jax``
   * - 2SLS (``IVRegressor``)
     - continuous
     - continuous or binary, with an instrument
     - yes, given an instrument
     - coefficient SE
     - --
   * - ``TMLELearner`` (ATE only)
     - continuous
     - binary
     - yes, with a propensity score
     - ATE CI
     - --

How the uncertainty is computed, per estimator, is cataloged in
:doc:`inference`.

Two rules of thumb
==================

* **Start simple, then justify complexity.** A T-learner with a linear base
  learner is a transparent baseline; adopt a more flexible estimator when
  held-out evaluation (:doc:`validation`) shows it ranks units better.
* **Do not choose by in-sample fit.** CATE models cannot be scored against an
  observed label. Compare candidates with the validation losses and ranking
  metrics on held-out data, as described in
  :ref:`Model Selection with Validation Losses <validation:Model Selection with Validation Losses>`.
