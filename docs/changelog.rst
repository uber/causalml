.. :changelog:

Changelog
=========

You can find the latest changes in the `GitHub releases <https://github.com/uber/causalml/releases>`_

Unreleased
----------

New Features
~~~~~~~~~~~~
* **Benchmark dataset loaders and ground-truth metrics (v1.0 M3).** ``fetch_lalonde``,
  ``fetch_ihdp`` and ``fetch_twins`` download a benchmark from its original source on
  first use and cache it under ``~/causalml-data`` (``CAUSALML_DATA`` overrides), with
  SHA256 verification so a file that has moved or been truncated raises instead of
  being parsed as data. ``download_if_missing=False`` stays offline, and
  ``clear_data_dir()`` empties the cache. None of the datasets are redistributed with
  CausalML; :doc:`datasets` records where each comes from and on what terms, including
  the four benchmarks that are documented rather than shipped.

  ``causalml.metrics`` gains the three metrics that need a known truth: ``pehe``
  (Hill's precision in estimating heterogeneous effects, with ``squared=False`` for the
  root), ``ate_error`` for the absolute error in the average effect, and
  ``policy_risk`` for the expected loss of the treat-if-positive policy on randomized
  data. Each docstring names what it requires — the ranking metrics stay the only ones
  computable on ordinary observational data.

  ``docs/examples/benchmark_leaderboard.ipynb`` is the leaderboard itself: running it
  end to end regenerates every published number. It reports three tables rather than
  one, because what is measurable is a property of how each dataset was made.

* **`CausalTreeRegressor` and `CausalRandomForestRegressor` accept** ``ccp_alpha="cv"``
  **(#584).** ``honesty=True`` supplies held-out leaf estimation; this setting adds the
  two remaining pieces of :cite:`athey2016recursive`. The splitting objective's variance penalty is scaled by
  ``1 + N_structure / N_estimation`` — the paper's factor of 2 at an even split — and tree
  size is selected by ``cv_folds``-fold cross-validation over the cost-complexity path,
  scoring each candidate subtree with that same objective evaluated on the held-out fold.

  Without it the tree grows until ``min_samples_leaf`` / ``min_group_samples`` stop it, and
  the variance penalty only ranks candidate splits instead of choosing tree size, which is
  the job it does in the paper. Held-out CATE RMSE for a single tree on a weak effect
  (16 seeds, ``min_samples_leaf=25``, paired by seed):

  .. list-table::
     :header-rows: 1

     * - Noise
       - ``honesty`` only
       - ``ccp_alpha="cv"``
       - Change
       - Leaves
     * - ``sigma=0.1``
       - 0.1212
       - 0.1192
       - -1.7% (n.s.)
       - 46 → 14
     * - ``sigma=0.5``
       - 0.2128
       - 0.1676
       - **-21.2%**
       - 46 → 13
     * - ``sigma=1.0``
       - 0.3499
       - 0.2287
       - **-34.6%**
       - 46 → 10
     * - ``sigma=2.0``
       - 0.7011
       - 0.2977
       - **-57.5%**
       - 46 → 8

  Better in 16 of 16 seeds at every noise level above 0.1, ``p < 0.0001``; at
  ``sigma=0.1`` there is no overfitting to remove and the two are equivalent. The cost is
  ``cv_folds`` extra fits per tree, roughly 5x fit time. Off by default because it changes
  fitted trees.

  This applies to a single tree. On ``CausalRandomForestRegressor`` the same option
  measured no gain: held-out CATE RMSE went from 0.084 to 0.130 at ``sigma=0.5`` (+55%,
  0 of 10 paired seeds better) and was unchanged at ``sigma=2.0``. Averaging across trees
  already removes the variance the cross-validated pruning targets, so pruning each tree to
  a few leaves adds bias and reduces ensemble diversity, at a larger cost when noise is
  lower. Prefer it on ``CausalTreeRegressor``.

  ``ccp_alpha="cv"`` requires ``honesty=True`` and raises otherwise, since the
  cross-validation scores candidate subtrees with the honest objective.

* **`CausalTreeRegressor.estimate_ate` and `BaseDRIVLearner.estimate_ate` accept**
  ``pretrain`` **(#517).** ``pretrain=True`` estimates from the already-fitted model
  instead of refitting on ``X``, so the rows passed in can be held out from the fit.
  The default refits on the rows it then estimates from, which leaves the estimate
  carrying whatever the model overfit. Fitting on half a randomized draw and
  estimating on the other half gave a mean absolute ATE error of 0.099 against 0.114
  in-sample, over 20 seeds on ``CausalTreeRegressor``.

  This completes the parameter across the package: the meta-learners have had it
  since #511. ``TMLELearner.estimate_ate`` is the remaining one without it and stays
  that way, since the class has no ``fit`` and so no fitted model to reuse.

  ``BaseDRIVLearner`` requires ``p`` when ``pretrain=True``: the propensity stored by
  ``fit`` is indexed by the rows it was fit on, while the standard error indexes it by
  the rows passed here. ``assignment`` and ``pZ`` are only used to fit and go unused.

* **`UpliftTreeClassifier` can prune inside** ``fit`` **(#1003).** ``prune_fraction``
  (default ``None``, off) holds out that fraction of the rows stratified on
  (treatment, outcome), grows the tree on the rest, and runs the existing ``prune()``
  on the holdout with ``min_gain`` / ``prune_rule``. Reaching validation-based pruning
  previously meant splitting the data, fitting, and calling ``prune()`` as a second
  step. ``prune()`` is unchanged for callers who manage their own holdout, and
  ``n_nodes_before_pruning_`` records the size pruning started from.

  The pruning rows are removed before the honest split, so neither the split search nor
  the estimation half sees them. Pruning renumbers nodes and promotes internal nodes to
  leaves, so the honest re-estimation and the per-node group counts are redone against
  the pruned tree.

  ``fit`` now rejects a held-out fraction outside ``(0, 1)`` — ``prune_fraction`` on the
  uplift tree and ``estimation_sample_size`` on both the uplift and causal trees —
  raising a ``ValueError`` that names the parameter instead of ``train_test_split``'s
  ``test_size``. ``prune_fraction=0.0`` was read as off.

  On ``make_uplift_classification`` (n=3000, 6 seeds, ``max_depth=None``,
  ``min_samples_leaf=20``), held-out qini went from -1.74 unpruned to -1.44 with
  ``prune_fraction=0.3``, and to 0.83 combined with ``honesty=True``. One simulated
  design, so the magnitudes are indicative.

Bug Fixes
~~~~~~~~~
* **`CausalTreeRegressor.estimate_ate` understated the standard error (#1006).** The
  interval came from ``dhat.std() / n``, which divides by ``n`` rather than ``sqrt(n)``
  and measures how the predicted effect varies across units instead of the sampling
  variability of its mean. At a nominal 95% over 100 randomized-trial draws
  (``n=1000``), the interval contained the true ATE 0 times.

  It is now the standard error of the estimate's influence function, which adds the
  model's residuals on the observed outcomes to the spread of the predicted effects —
  the quantity ``BaseTLearner.estimate_ate`` already computes as a three-term
  variance. Coverage on the same draws is 88%, the remainder being that ``estimate_ate``
  fits and estimates on the same rows (#517). Point estimates are unchanged; only the
  interval moves, and it is wider.

* **`UpliftTreeClassifier.prune` left `_node_group_counts` stale (#1003).** ``prune()``
  replaced ``tree_`` without rebuilding the per-node group counts, which are indexed by
  node id and read by the uplift-score p-value and the plot's ``group_size``. The stale
  array is longer than the pruned tree, so it returned another node's counts instead of
  raising. ``prune()`` now carries the counts over to the pruned node ids, so a
  standalone call is fixed too, not only the fit-time path. A surviving node is reached
  by the same rows either way, so the values transfer unchanged; the remap replays the
  pruner's traversal and matches a full recomputation exactly.

Behavior Changes
~~~~~~~~~~~~~~~~
* **`CausalTreeRegressor` and `CausalRandomForestRegressor` now estimate leaves honestly by
  default (#584).** The new ``honesty`` parameter defaults to ``True``: the sample is split
  in two, the tree structure is grown on one half and each leaf's per-group outcome means
  are re-estimated on the other, so a leaf value no longer inherits the selection bias of
  the split search that produced it :cite:`athey2016recursive`.
  ``estimation_sample_size`` (default ``0.5``) sets the held-out fraction, and the split is
  stratified on treatment. Both are the names ``UpliftTreeClassifier`` already uses, and
  the default matches ``grf``'s ``honesty = TRUE`` and EconML's ``honest=True``.

  **This changes fitted models and predictions.** An existing script gets a different tree,
  different leaf values and different CATE estimates without any edit. Pass
  ``honesty=False`` to keep the previous behavior.

  Measured over 200 replications on data with no treatment effect anywhere, with the tree
  structure held fixed across both arms: in-sample leaves reported a mean absolute
  estimated effect of 0.163 versus 0.095 for honest leaves, a 42% reduction in spurious
  heterogeneity. The trade is variance: each half sees only part of the data, so an honest
  tree is shallower and noisier per leaf, and at small sample sizes an individual estimate
  can be worse. Use ``honesty=False`` when halving the sample leaves too little to split
  on.

  On the forest, honesty composes with ``bootstrap`` rather than replacing it: the
  bootstrap counts arrive as each tree's ``sample_weight`` and weight both the structure
  fit and the leaf re-estimation, so this is not the subsample-without-replacement scheme
  of :cite:`athey2019generalized`.

* **`UpliftRandomForestClassifier` now defaults to** ``n_jobs=None`` **(one worker) instead of**
  ``n_jobs=-1`` **(#991).** Trees are fitted in parallel and each concurrent fit holds its own
  working set, so peak memory grew in proportion to the machine's core count — the same fit
  needed more memory on a larger machine. On a 100k × 440 benchmark, peak memory was 8.8× the
  input array at ``n_jobs=-1`` (10 cores) versus 1.4× at ``n_jobs=1``.

  Fitted models and predictions are unchanged; only peak memory and wall time move. Existing
  code gets slower and much lighter without any edit. Pass ``n_jobs=-1`` explicitly to restore
  the previous behavior. Note that ``n_estimators`` does not affect peak memory once ``n_jobs``
  is fixed.

  This matches ``CausalRandomForestRegressor`` and scikit-learn's forests, which use the same
  across-trees parallelism and default to serial.

* **`causalml.features.load_data` now returns** ``numpy.ndarray`` **instead of**
  ``numpy.matrix`` **(#978).** This was the only function in the package that returned a
  ``numpy.matrix``. Code that relied on matrix semantics — ``*`` as matrix multiplication,
  or an always-2-D result from indexing — needs ``np.asmatrix()`` around the call or,
  preferably, the equivalent ``ndarray`` operation. ``numpy.matrix`` carries a
  ``PendingDeprecationWarning`` upstream.

  The same change fixes an ``AssertionError`` raised whenever the one-hot encoder produced
  no columns: with an all-numeric feature list, a constant categorical column, or a
  high-cardinality column whose levels all fall below ``min_obs``. All three now return a
  feature matrix instead of raising (#978, #979).

Deprecations
~~~~~~~~~~~~
* **Positional** ``treatment`` **and** ``y`` **are deprecated (#854).** In v1.0 every
  learner's positional argument order becomes ``fit(X, y, treatment, ...)``, matching
  the scikit-learn convention so a CausalML learner can be a ``Pipeline`` step. Passing
  ``treatment`` or ``y`` by position now emits a ``FutureWarning``; positional order is
  unchanged in this release, so nothing breaks yet.

  The fix is to pass them by keyword — ``learner.fit(X=X, treatment=treatment, y=y)`` —
  which is order-independent and therefore correct both before and after the flip.

  This one cannot be caught at runtime: ``y`` and ``treatment`` are both same-length
  arrays, so a call left in the old positional form will silently train on swapped
  arguments at v1.0 rather than raising ``TypeError``. See the
  :ref:`migration guide <fit-argument-order>` for the per-family table and for the
  signatures that are not a plain swap (``IVRegressor.fit`` and ``BaseDRIVLearner``).

  The warning is live for three minor releases — 0.18.0, 0.19.0 and 0.20.0 — before
  the flip in v1.0, roughly nine months.

  By @jeongyoonlee in https://github.com/uber/causalml/pull/975

Bug Fixes
~~~~~~~~~
* **macOS without the OpenMP runtime now reports an actionable error (#908).**
  The installation guide now documents the required OpenMP runtime and
  installation options.

Release Schedule
~~~~~~~~~~~~~~~~
Quarterly, targeting:

.. list-table::
   :header-rows: 1

   * - Release
     - Target
     - Argument order
   * - 0.18.0
     - September 2026
     - ``fit(X, treatment, y, ...)`` — deprecation warning added
   * - 0.19.0
     - December 2026
     - unchanged, warning continues
   * - 0.20.0
     - March 2027
     - unchanged, warning continues
   * - 1.0.0
     - June 2027
     - ``fit(X, y, treatment, ...)`` — the flip, shim removed

Dates are targets, not commitments. The ordering is fixed: no signature moves before
v1.0.

0.17.0 (Jul 2026)
-----------------
* Adds **scikit-learn 1.9 support**, resolving an ``import causalml.dataset`` failure present on the 0.16.0 wheel (#926).
* Adds **native Polars** ``DataFrame``/``Series``/``LazyFrame`` support across all meta-learners, and a **JAX/flax.nnx backend** for DragonNet.
* Makes the meta-learners **scikit-learn compliant** (``BaseEstimator``), so ``clone()`` and ``get_params()`` work.
* Adds the **RATE** evaluation metric and post-fit confidence intervals for ``BaseTLearner``, plus numerous bug fixes.

New Features
~~~~~~~~~~~~
* Add native Polars DataFrame, Series, and LazyFrame support for all meta-learners by @aman-coder03 in https://github.com/uber/causalml/pull/901
* Polish Polars support and add bootstrap CI test coverage by @aman-coder03 in https://github.com/uber/causalml/pull/921
* Add JAX/flax.nnx backend for DragonNet by @xrhd in https://github.com/uber/causalml/pull/918
* docs: document JAX backend for DragonNet by @xrhd in https://github.com/uber/causalml/pull/919
* Make meta-learners scikit-learn compliant via BaseEstimator by @aman-coder03 in https://github.com/uber/causalml/pull/912
* Add Rank-weighted Average Treatment Effect (RATE) metric by @aman-coder03 in https://github.com/uber/causalml/pull/887
* Add ``rate_score()`` with bootstrap confidence intervals and p-values by @aman-coder03 in https://github.com/uber/causalml/pull/890
* Add AIPW docstring warning to ``get_toc()`` and ``rate_score()`` by @jeongyoonlee in https://github.com/uber/causalml/pull/891
* Add post-fit confidence intervals to ``BaseTLearner`` via ``store_bootstraps`` and ``return_ci`` by @aman-coder03 in https://github.com/uber/causalml/pull/886
* Support NaN values in UpliftTree and UpliftRandomForest by @aman-coder03 in https://github.com/uber/causalml/pull/860

scikit-learn 1.9 Support
~~~~~~~~~~~~~~~~~~~~~~~~~~
* Fix for compatibility with sklearn v1.9.0 by @jakevdp in https://github.com/uber/causalml/pull/903
* Fix CausalRandomForestRegressor.fit() on scikit-learn 1.9 by @jeongyoonlee in https://github.com/uber/causalml/pull/907
* Support CausalRandomForestRegressor.calculate_error() on scikit-learn >= 1.9 by @jeongyoonlee in https://github.com/uber/causalml/pull/909

Bug Fixes
~~~~~~~~~
* Fix #904: Prevent deepcopy of fitted templates in bootstrap and correct predict validation ordering by @Saurav-Gupta-9741 in https://github.com/uber/causalml/pull/910
* Optimised training, inference and memory for metalearners in multitreatment settings by @Ic3fr0g in https://github.com/uber/causalml/pull/896
* Fix UpliftRandomForest predict shape mismatch with multiple treatments by @jeongyoonlee in https://github.com/uber/causalml/pull/884
* Fix uplift tree p-value NaN from division by zero by @jeongyoonlee in https://github.com/uber/causalml/pull/882
* Fix CausalRandomForestRegressor predicting inf from division by zero by @jeongyoonlee in https://github.com/uber/causalml/pull/883
* Fix SensitivityPlaceboTreatment ignoring actual treatment groups by @jeongyoonlee in https://github.com/uber/causalml/pull/880
* Fix seed parameter TypeError in BaseDRLearner bootstrap CI by @mohsinm-dev in https://github.com/uber/causalml/pull/879
* Fix ValueError on read-only arrays in BaseSLearner.predict() by @mohsinm-dev in https://github.com/uber/causalml/pull/878
* Add input validation to auuc_score for missing model columns by @jeongyoonlee in https://github.com/uber/causalml/pull/881
* Make xgboost optional in synthetic dataset generation by @Si-ra-kri in https://github.com/uber/causalml/pull/872
* Bug Fix: use iloc to index pd.Series by @bekojuniranjan in https://github.com/uber/causalml/pull/877

Build / CI
~~~~~~~~~~
* Make Cython line tracing opt-in to keep release wheels fast by @HSJung93 in https://github.com/uber/causalml/pull/914
* Remove the PyPI token from GitHub Actions in favor of the Trusted Publishing by @jeongyoonlee in https://github.com/uber/causalml/pull/871
* Upgrade GitHub Actions for Node 24 compatibility by @salmanmkc in https://github.com/uber/causalml/pull/874
* Upgrade GitHub Actions to latest versions by @salmanmkc in https://github.com/uber/causalml/pull/875
* ci: declare workflow-level ``contents: read`` on 4 workflows by @arpitjain099 in https://github.com/uber/causalml/pull/900

Breaking Changes
~~~~~~~~~~~~~~~~~
* **Meta-learner** ``__init__`` **signatures (#912):** to become scikit-learn ``BaseEstimator`` s, each learner now stores its constructor arguments verbatim and builds models in ``fit()``. Most visibly, ``XGBRRegressor`` no longer accepts arbitrary ``**kwargs`` — pass XGBoost parameters via the explicit ``xgb_kwargs=<dict>`` argument.

0.16.0 (Feb 2026)
-----------------
* **BREAKING CHANGE:** This release upgrades from manylinux2014 to manylinux_2_28 for Linux wheel distribution.
* Pre-built wheels now require glibc 2.28 or later (Ubuntu 20.04+, RHEL 8+, Debian 10+).
* Users on older Linux distributions (Ubuntu 18.04, RHEL 7, etc.) must build from source.
* Removes scipy version pin, enabling compatibility with both scipy 1.16.x and 1.17.x.

Updates
~~~~~~~
* Upgrade to manylinux_2_28 and remove scipy version constraints by @jeongyoonlee in https://github.com/uber/causalml/pull/869
* Upgrade cibuildwheel to v3.3.1 and remove deprecated macos-13 runner by @jeongyoonlee in https://github.com/uber/causalml/pull/867
* Fix Ubuntu packaging failure - scipy manylinux compatibility by @jeongyoonlee in https://github.com/uber/causalml/pull/865
* Fix Ubuntu packaging failure by aligning cibuildwheel config with Python version requirement by @jeongyoonlee in https://github.com/uber/causalml/pull/864

Breaking Changes
~~~~~~~~~~~~~~~~
* **Linux wheel compatibility:** Pre-built wheels require glibc 2.28+ (manylinux_2_28)

  * ✅ **Supported:** Ubuntu 20.04+, RHEL/CentOS 8+, Debian 10+, Fedora 32+
  * ⚠️ **Requires source build:** Ubuntu 18.04, RHEL 7, Ubuntu 16.04, Debian 9

* **Python version:** Minimum Python version is 3.11 (updated from 3.9)

Migration Notes
~~~~~~~~~~~~~~~
If you are on an older Linux distribution:

1. **Check your glibc version:** ``ldd --version``
2. **If glibc < 2.28:** Install from source instead of using pip wheels:

   .. code-block:: bash

       git clone https://github.com/uber/causalml.git
       cd causalml
       pip install -e .

3. **Recommended:** Upgrade to a modern Linux distribution (Ubuntu 20.04+, RHEL 8+)

0.15.1 (Apr 2024)
-----------------
* This release fixes the build failure on macOS and a few bugs in ``UpliftTreeClassifier``.
* We have two new contributors, @lee-junseok and @IanDelbridge. Thanks for your contributions!

Updates
~~~~~~~
* Relax ``pandas`` version requirement by @jeongyoonlee in https://github.com/uber/causalml/pull/743
* Remove undefined variables in ``match.__main__()`` by @jeongyoonlee in https://github.com/uber/causalml/pull/749
* Fix ``distr_plot_single_sim()`` by @jeongyoonlee in https://github.com/uber/causalml/pull/750
* Add ``with_std``, ``with_counts`` to ``create_table_one`` by @lee-junseok in https://github.com/uber/causalml/pull/748
* fix stratified sampling call by @IanDelbridge in https://github.com/uber/causalml/pull/756
* 20240207 honest leaf size by @IanDelbridge in https://github.com/uber/causalml/pull/753
* 757: add ``return_ci=True`` in sensitivity by @lee-junseok in https://github.com/uber/causalml/pull/758
* Update sensitivity tests with more meta-learners by @jeongyoonlee in https://github.com/uber/causalml/pull/759
* manually specify ``multiprocessing`` use ``fork`` in ``setup.py`` by @IanDelbridge in https://github.com/uber/causalml/pull/754

New contributors
~~~~~~~~~~~~~~~~
* @lee-junseok made their first contribution in https://github.com/uber/causalml/pull/748
* @IanDelbridge made their first contribution in https://github.com/uber/causalml/pull/756

0.15.0 (Feb 2024)
-----------------
* In this release, we revamped documentation, cleaned up dependencies, and improved installation - in addition to the long list of bug fixes.
* We have three new contributors, @peterloleungyau, @SuperBo, and @ZiJiaW, who submitted their first PRs to CausalML. @erikcs also contributed to @ras44's PR #729 to add the wrapper for his MAQ implementation to CausalML. Thanks for your contributions!

Updates
~~~~~~~
* Update python-publish.yml by @jeongyoonlee in https://github.com/uber/causalml/pull/673
* Add build.[os, tools.python] to .readthedocs.yml by @jeongyoonlee in https://github.com/uber/causalml/pull/676
* Update notebook example with causal trees interpretation by @alexander-pv in https://github.com/uber/causalml/pull/683
* Remove the numpy and pandas version restriction in pyproject.toml by @jeongyoonlee in https://github.com/uber/causalml/pull/681
* Add governance documents by @jeongyoonlee in https://github.com/uber/causalml/pull/688
* Update GOVERNANCE.md by @ras44 in https://github.com/uber/causalml/pull/691
* Dev/governance docs to snake-case by @ras44 in https://github.com/uber/causalml/pull/693
* Reduce sklearn dependency in causalml by @alexander-pv in https://github.com/uber/causalml/pull/686
* Update MAINTAINERS.md by @jeongyoonlee in https://github.com/uber/causalml/pull/696
* Modified to speed up UpliftTreeClassifier.growDecisionTreeFrom. by @peterloleungyau in https://github.com/uber/causalml/pull/695
* Update README.md by @ras44 in https://github.com/uber/causalml/pull/698
* Add notebook examples to docs by @jeongyoonlee in https://github.com/uber/causalml/pull/697
* resolves change requests in #166 by @ras44 in https://github.com/uber/causalml/pull/701
* Fix the readthedocs build error by @jeongyoonlee in https://github.com/uber/causalml/pull/702
* Replace Stack and PriorityHeap with cpp stack/heap methods in trees by @SuperBo in https://github.com/uber/causalml/pull/700
* Hotfix for #701 by @jeongyoonlee in https://github.com/uber/causalml/pull/705
* Dev/699 win build fix by @ras44 in https://github.com/uber/causalml/pull/710
* expose n_jobs for rlearner by @ZiJiaW in https://github.com/uber/causalml/pull/714
* minimal fix to resolve #707 by @ras44 in https://github.com/uber/causalml/pull/720
* Add Python 3.10, 3.11, 3.12 to the testing by @cclauss in https://github.com/uber/causalml/pull/454
* Remove Python 3.12 from the build tests in python-test.yaml by @jeongyoonlee in https://github.com/uber/causalml/pull/726
* fix plot_std_diffs, add bal_tol, condense to one plot by @ras44 in https://github.com/uber/causalml/pull/723
* Dev/677 documentation by @ras44 in https://github.com/uber/causalml/pull/725
* documentation updates by @ras44 in https://github.com/uber/causalml/pull/728
* resolves #730, docs clean conda install by @ras44 in https://github.com/uber/causalml/pull/731
* minimal wrapper of MAQ #662 by @ras44 in https://github.com/uber/causalml/pull/729
* Temporary fix for causal trees missing values support #733 by @alexander-pv in https://github.com/uber/causalml/pull/734
* resolves #639, credit due to Dong Liu by @ras44 in https://github.com/uber/causalml/pull/722

New contributors
~~~~~~~~~~~~~~~~
* @peterloleungyau made their first contribution in https://github.com/uber/causalml/pull/695
* @SuperBo made their first contribution in https://github.com/uber/causalml/pull/700
* @ZiJiaW made their first contribution in https://github.com/uber/causalml/pull/714


0.14.1 (Aug 2023)
-----------------
* This release mainly addressed installation issues and updated documentation accordingly.
* We have 4 new contributors. @bsaunders27, @xhulianoThe1, @zpppy, and @bsaunders23. Thanks for your contributions!

Updates
~~~~~~~
* Update the python-publish workflow file to fix the package publish Gi… by @jeongyoonlee in https://github.com/uber/causalml/pull/633
* Update Cython dependency by @alexander-pv in https://github.com/uber/causalml/pull/640
* Fix for builds on Mac M1 infrastructure by @bsaunders27 in https://github.com/uber/causalml/pull/641
* code cleanups by @xhulianoThe1 in https://github.com/uber/causalml/pull/634
* support valid error early stopping by @zpppy in https://github.com/uber/causalml/pull/614
* fix: update to ``envs/`` conda build for precompiled M1 installs by @bsaunders27 in https://github.com/uber/causalml/pull/646
* Installation updates to README and .github/workflows by @ras44 in https://github.com/uber/causalml/pull/637
* fix: simulate_randomized_trial by @bsaunders23 in https://github.com/uber/causalml/pull/656
* issue 252 by @vincewu51 in https://github.com/uber/causalml/pull/660
* ras44/651 graph viz, resolves #651 by @ras44 in https://github.com/uber/causalml/pull/661
* linted with black by @ras44 in https://github.com/uber/causalml/pull/663
* Fix issue 650 by @vincewu51 in https://github.com/uber/causalml/pull/659
* Install graphviz in the workflow builds by @jeongyoonlee in https://github.com/uber/causalml/pull/668
* Update docs/installation.rst by @jeongyoonlee in https://github.com/uber/causalml/pull/667
* Schedule monthly PyPI install tests by @jeongyoonlee in https://github.com/uber/causalml/pull/670

New contributors
~~~~~~~~~~~~~~~~
* @bsaunders27 made their first contribution in https://github.com/uber/causalml/pull/641
* @xhulianoThe1 made their first contribution in https://github.com/uber/causalml/pull/634
* @zpppy made their first contribution in https://github.com/uber/causalml/pull/614
* @bsaunders23 made their first contribution in https://github.com/uber/causalml/pull/656


0.14.0 (July 2023)
------------------
- CausalML surpassed `2MM downloads <https://pepy.tech/project/causalml>`_ on PyPI and `4,100 stars <https://github.com/uber/causalml/stargazers>`_ on GitHub. Thanks for choosing CausalML and supporting us on GitHub.
- We have 7 new contributors: @darthtrevino, @ras44, @AbhishekVermaDH, @joel-mcmurry, @AlxClt, @kklein, and @volico. Thanks for your contributions!

Updates
~~~~~~~
- Fix the readthedocs build failure by @jeongyoonlee in https://github.com/uber/causalml/pull/545
- Add ``pyproject.toml`` with basic build dependencies for PEP518 compliance by @darthtrevino in https://github.com/uber/causalml/pull/553
- bump ``numpy`` from 1.20.3 to 1.23.2 in ``environment-py38.yml`` #338 by @ras44 in https://github.com/uber/causalml/pull/550
- CausalTree split criterions fix and fit optimization by @alexander-pv in https://github.com/uber/causalml/pull/557
- fixing math notations for proper rendering by @AbhishekVermaDH in https://github.com/uber/causalml/pull/558
- Update ``methodology.rst`` by @joel-mcmurry in https://github.com/uber/causalml/pull/568
- Causal trees bootstrapping and ``max_leaf_nodes`` fixes with minor update by @alexander-pv in https://github.com/uber/causalml/pull/583
- Fix #596 by @AlxClt in https://github.com/uber/causalml/pull/597
- Add ``**kwargs`` to ``Explainer.plot_shap_values()`` by @jeongyoonlee in https://github.com/uber/causalml/pull/603
- Make the Adam optimization optional and learning rate/epochs configurable in DragonNet by @jeongyoonlee in https://github.com/uber/causalml/pull/604
- Fix bug in variance calculation in drivlearner. by @huigangchen in https://github.com/uber/causalml/pull/606
- Bug Fix in Dragonnet: Adam parameter name lr depreciation by @huigangchen in https://github.com/uber/causalml/pull/617
- Fix AttributeError in builds with ``numpy>=1.24`` and ``pandas>=2.0`` by @jeongyoonlee in https://github.com/uber/causalml/pull/631
- Pass on ``**kwargs`` in ``plot_shap_values`` of base meta leaner by @kklein in https://github.com/uber/causalml/pull/627
- Bump ``scipy`` from 1.4.1 to 1.10.0 by @dependabot in https://github.com/uber/causalml/pull/629
- Feature/ttest criterion by @volico in https://github.com/uber/causalml/pull/570
- Added Interaction Tree (IT), Causal Inference Tree (CIT), and Invariant DDP (IDDP) by @jroessler in https://github.com/uber/causalml/pull/562
- Causal trees option to return counterfactual outcomes by @alexander-pv in https://github.com/uber/causalml/pull/623

New contributors
~~~~~~~~~~~~~~~~
- @darthtrevino made their first contribution in https://github.com/uber/causalml/pull/553
- @ras44 made their first contribution in https://github.com/uber/causalml/pull/550
- @AbhishekVermaDH made their first contribution in https://github.com/uber/causalml/pull/558
- @joel-mcmurry made their first contribution in https://github.com/uber/causalml/pull/568
- @AlxClt made their first contribution in https://github.com/uber/causalml/pull/597
- @kklein made their first contribution in https://github.com/uber/causalml/pull/627
- @volico made their first contribution in https://github.com/uber/causalml/pull/570


0.13.0 (Sep 2022)
-----------------
- CausalML surpassed `1MM downloads <https://pepy.tech/project/causalml>`_ on PyPI and `3,200 stars <https://github.com/uber/causalml/stargazers>`_ on GitHub. Thanks for choosing CausalML and supporting us on GitHub.
- We have 7 new contributors @saiwing-yeung, @lixuan12315, @aldenrogers, @vincewu51, @AlkanSte, @enzoliao, and @alexander-pv. Thanks for your contributions!
- @alexander-pv revamped `CausalTreeRegressor` and added `CausalRandomForestRegressor` with more seamless integration with `scikit-learn`'s Cython tree module. He also added integration with `shap` for causal tree/ random forest interpretation. Please check out the `example notebook <https://github.com/uber/causalml/blob/master/docs/examples/causal_trees_interpretation.ipynb>`_.
- We dropped the support for Python 3.6 and removed its test workflow.

Updates
~~~~~~~
- Fix typo ``(% -> $)`` by @saiwing-yeung in https://github.com/uber/causalml/pull/488
- Add function for calculating PNS bounds by @t-tte in https://github.com/uber/causalml/pull/482
- Fix hard coding bug by @t-tte in https://github.com/uber/causalml/pull/492
- Update README of ``conda`` install and instruction of maintain in ``conda-forge`` by @ppstacy in https://github.com/uber/causalml/pull/485
- Update ``examples.rst`` by @lixuan12315 in https://github.com/uber/causalml/pull/496
- Fix incorrect ``effect_learner_objective`` in ``XGBRRegressor`` by @jeongyoonlee in https://github.com/uber/causalml/pull/504
- Fix Filter F doesn't work with latest ``statsmodels``' F test f-value format by @paullo0106 in https://github.com/uber/causalml/pull/505
- Exclude tests in ``setup.py`` by @aldenrogers in https://github.com/uber/causalml/pull/508
- Enabling higher orders feature importance for F filter and LR filter by @zhenyuz0500 in https://github.com/uber/causalml/pull/509
- Ate pretrain 0506 by @vincewu51 in https://github.com/uber/causalml/pull/511
- Update ``methodology.rst`` by @AlkanSte in https://github.com/uber/causalml/pull/518
- Fix the bug of incorrect result in qini for multiple models by @enzoliao in https://github.com/uber/causalml/pull/520
- Test ``get_qini()`` by @enzoliao in https://github.com/uber/causalml/pull/523
- Fixed typo in ``uplift_trees_with_synthetic_data.ipynb`` by @jroessler in https://github.com/uber/causalml/pull/531
- Remove Python 3.6 test from workflows by @jeongyoonlee in https://github.com/uber/causalml/pull/535
- Causal trees update by @alexander-pv in https://github.com/uber/causalml/pull/522
- Causal trees interpretation example by @alexander-pv in https://github.com/uber/causalml/pull/536


0.12.3 (Feb 2022)
-----------------
This patch is to release a version without the constraint for Shap to be abled to use for Conda.

Updates
~~~~~~~
- `#483 <https://github.com/uber/causalml/pull/483>`_ by @ppstacy: Modify the requirement version of Shap


0.12.2 (Feb 2022)
-----------------
This patch includes three updates by @tonkolviktor and @heiderich as follows. We also start using `black <https://black.readthedocs.io/en/stable/integrations/index.html>`_, a Python formatter. Please check out the updated `contribution guideline <https://github.com/uber/causalml/blob/master/CONTRIBUTING.md>`_ to learn how to use it.

Updates
~~~~~~~
- `#473 <https://github.com/uber/causalml/pull/477>`_ by @tonkolviktor: Open up the scipy dependency version
- `#476 <https://github.com/uber/causalml/pull/476>`_ by @heiderich: Use preferred backend for joblib instead of hard-coding it
- `#477 <https://github.com/uber/causalml/pull/477>`_ by @heiderich: Allow parallel prediction for UpliftRandomForestClassifier and make the joblib's preferred backend configurable


0.12.1 (Feb 2022)
-----------------
This patch includes two bug fixes for UpliftRandomForestClassifier as follows:

Updates
~~~~~~~
- `#462 <https://github.com/uber/causalml/pull/462>`_ by @paullo0106: Use the correct treatment_idx for fillTree() when applying validation data set
- `#468 <https://github.com/uber/causalml/pull/468>`_ by @jeongyoonlee: Switch the joblib backend for UpliftRandomForestClassifier to threading to avoid memory copy across trees


0.12.0 (Jan 2022)
-----------------
- CausalML surpassed `637K downloads <https://pepy.tech/project/causalml>`_ on PyPI and `2,500 stars <https://github.com/uber/causalml/stargazers>`_ on Github!
- We have 4 new community contributors, Luis (`@lgmoneda <https://github.com/lgmoneda>`_), Ravi (`@raviksharma <https://github.com/raviksharma>`_), Louis (`@LouisHernandez17 <https://github.com/LouisHernandez17>`_) and JackRab (`@JackRab <https://github.com/JackRab>`_). Thanks for the contribution!
- We refactored and speeded up UpliftTreeClassifier/UpliftRandomForestClassifier by 5x with Cython  (`#422 <https://github.com/uber/causalml/pull/422>`_ `#440 <https://github.com/uber/causalml/pull/440>`_ by @jeongyoonlee)
- We revamped our `API documentation <https://causalml.readthedocs.io/en/latest/about.html>`_, it now includes the latest methodology, references, installation, notebook examples, and graphs! (`#413 <https://github.com/uber/causalml/discussions/413>`_ by @huigangchen @t-tte @zhenyuz0500 @jeongyoonlee @paullo0106)
- Our team gave talks at `2021 Conference on Digital Experimentation @ MIT (CODE@MIT) <https://ide.mit.edu/events/2021-conference-on-digital-experimentation-mit-codemit/>`_, `Causal Data Science Meeting 2021 <https://www.causalscience.org/meeting/program/day-2/>`_,  and `KDD 2021 Tutorials <https://causal-machine-learning.github.io/kdd2021-tutorial/>`_ on CausalML introduction and applications. Please take a look if you missed them! Full list of publications and talks can be found here.

Updates
~~~~~~~
- Update documentation on Instrument Variable methods @huigangchen (`#447 <https://github.com/uber/causalml/pull/447>`_)
- Add benchmark simulation studies example notebook by @t-tte (`#443 <https://github.com/uber/causalml/pull/443>`_)
- Add sample_weight support for R-learner by @paullo0106 (`#425 <https://github.com/uber/causalml/pull/425>`_)
- Fix incorrect binning of numeric features in UpliftTreeClassifier by @jeongyoonlee (`#420 <https://github.com/uber/causalml/pull/420>`_)
- Update papers, talks, and publication info to README and refs.bib by @zhenyuz0500 (`#410 <https://github.com/uber/causalml/pull/410>`_ `#414 <https://github.com/uber/causalml/pull/414>`_ `#433 <https://github.com/uber/causalml/pull/433>`_)
- Add instruction for contributing.md doc by @jeongyoonlee (`#408 <https://github.com/uber/causalml/pull/408>`_)
- Fix incorrect feature importance calculation logic by @paullo0106 (`#406 <https://github.com/uber/causalml/pull/406>`_)
- Add parallel jobs support for NearestNeighbors search with n_jobs parameter by @paullo0106 (`#389 <https://github.com/uber/causalml/pull/389>`_)
- Fix bug in simulate_randomized_trial by @jroessler (`#385 <https://github.com/uber/causalml/pull/385>`_)
- Add GA pytest workflow by @ppstacy (`#380 <https://github.com/uber/causalml/pull/380>`_)



0.11.0 (2021-07-28)
-------------------
- CausalML surpassed `2K stars <https://github.com/uber/causalml/stargazers>`_!
- We have 3 new community contributors, Jannik (`@jroessler <https://github.com/jroessler>`_), Mohamed (`@ibraaaa <https://github.com/ibraaaa>`_), and Leo (`@lleiou <https://github.com/lleiou>`_). Thanks for the contribution!

Major Updates
~~~~~~~~~~~~~
- Make tensorflow dependency optional and add python 3.9 support by @jeongyoonlee (`#343 <https://github.com/uber/causalml/pull/343>`_)
- Add delta-delta-p (ddp) tree inference approach by @jroessler (`#327 <https://github.com/uber/causalml/pull/327>`_)
- Add conda env files for Python 3.6, 3.7, and 3.8 by @jeongyoonlee (`#324 <https://github.com/uber/causalml/pull/324>`_)

Minor Updates
~~~~~~~~~~~~~
- Fix inconsistent feature importance calculation in uplift tree by @paullo0106 (`#372 <https://github.com/uber/causalml/pull/372>`_)
- Fix filter method failure with NaNs in the data issue by @manojbalaji1 (`#367 <https://github.com/uber/causalml/pull/367>`_)
- Add automatic package publish by @jeongyoonlee (`#354 <https://github.com/uber/causalml/pull/354>`_)
- Fix typo in unit_selection optimization by @jeongyoonlee (`#347 <https://github.com/uber/causalml/pull/347>`_)
- Fix docs build failure by @jeongyoonlee (`#335 <https://github.com/uber/causalml/pull/335>`_)
- Convert pandas inputs to numpy in S/T/R Learners by @jeongyoonlee (`#333 <https://github.com/uber/causalml/pull/333>`_)
- Require scikit-learn as a dependency of setup.py by @ibraaaa (`#325 <https://github.com/uber/causalml/pull/325>`_)
- Fix AttributeError when passing in Outcome and Effect learner to R-Learner by @paullo0106 (`#320 <https://github.com/uber/causalml/pull/320>`_)
- Fix error when there is no positive class for KL Divergence filter by @lleiou (`#311 <https://github.com/uber/causalml/pull/311>`_)
- Add versions to cython and numpy in setup.py for requirements.txt accordingly by @maccam912 (`#306 <https://github.com/uber/causalml/pull/306>`_)



0.10.0 (2021-02-18)
-------------------
- CausalML surpassed `235,000 downloads <https://pepy.tech/project/causalml>`_!
- We have 5 new community contributors, Suraj (`@surajiyer <https://github.com/surajiyer>`_), Harsh (`@HarshCasper <https://github.com/HarshCasper>`_), Manoj (`@manojbalaji1 <https://github.com/manojbalaji1>`_), Matthew (`@maccam912 <https://github.com/maccam912>`_) and Václav (`@vaclavbelak <https://github.com/vaclavbelak>`_). Thanks for the contribution!

Major Updates
~~~~~~~~~~~~~
- Add Policy learner, DR learner, DRIV learner by @huigangchen (`#292 <https://github.com/uber/causalml/pull/292>`_)
- Add wrapper for CEVAE, a deep latent-variable and variational autoencoder based model by @ppstacy(`#276 <https://github.com/uber/causalml/pull/276>`_)

Minor Updates
~~~~~~~~~~~~~
- Add propensity_learner to R-learner by @jeongyoonlee (`#297 <https://github.com/uber/causalml/pull/297>`_)
- Add BaseLearner class for other meta-learners to inherit from without duplicated code by @jeongyoonlee (`#295 <https://github.com/uber/causalml/pull/295>`_)
- Fix installation issue for Shap>=0.38.1 by @paullo0106 (`#287 <https://github.com/uber/causalml/pull/287>`_)
- Fix import error for sklearn>= 0.24 by @jeongyoonlee (`#283 <https://github.com/uber/causalml/pull/283>`_)
- Fix KeyError issue in Filter method for certain dataset by @surajiyer (`#281 <https://github.com/uber/causalml/pull/281>`_)
- Fix inconsistent cumlift score calculation of multiple models by @vaclavbelak (`#273 <https://github.com/uber/causalml/pull/273>`_)
- Fix duplicate values handling in feature selection method by @manojbalaji1 (`#271 <https://github.com/uber/causalml/pull/271>`_)
- Fix the color spectrum of SHAP summary plot  for feature interpretations of meta-learners by @paullo0106 (`#269 <https://github.com/uber/causalml/pull/269>`_)
- Add IIA and value optimization related documentation by @t-tte (`#264 <https://github.com/uber/causalml/pull/264>`_)
- Fix StratifiedKFold arguments for propensity score estimation by @paullo0106 (`#262 <https://github.com/uber/causalml/pull/262>`_)
- Refactor the code with string format argument and is to compare object types, and change methods not using bound instance to static methods by @harshcasper (`#256 <https://github.com/uber/causalml/pull/256>`_, `#260 <https://github.com/uber/causalml/pull/260>`_)



0.9.0 (2020-10-23)
------------------
- CausalML won the 1st prize at the poster session in UberML'20
- DoWhy integrated CausalML starting v0.4 (`release note <https://github.com/microsoft/dowhy/releases/tag/v0.4>`_)
- CausalML team welcomes new project leadership, Mert Bay
- We have 4 new community contributors, Mario Wijaya (`@mwijaya3 <https://github.com/mwijaya3>`_), Harry Zhao (`@deeplaunch <https://github.com/deeplaunch>`_), Christophe (`@ccrndn <https://github.com/ccrndn>`_) and Georg Walther (`@waltherg <https://github.com/waltherg>`_). Thanks for the contribution!

Major Updates
~~~~~~~~~~~~~
- Add feature importance and its visualization to UpliftDecisionTrees and UpliftRF by @yungmsh (`#220 <https://github.com/uber/causalml/pull/220>`_)
- Add feature selection example with Filter methods by @paullo0106 (`#223 <https://github.com/uber/causalml/pull/223>`_)

Minor Updates
~~~~~~~~~~~~~
- Implement propensity model abstraction for common interface by @waltherg (`#223 <https://github.com/uber/causalml/pull/223>`_)
- Fix bug in BaseSClassifier and BaseXClassifier by @yungmsh and @ppstacy (`#217 <https://github.com/uber/causalml/pull/217>`_), (`#218 <https://github.com/uber/causalml/pull/218>`_)
- Fix parentNodeSummary for UpliftDecisionTrees by @paullo0106 (`#238 <https://github.com/uber/causalml/pull/238>`_)
- Add pd.Series for propensity score condition check by @paullo0106 (`#242 <https://github.com/uber/causalml/pull/242>`_)
- Fix the uplift random forest prediction output by @ppstacy (`#236 <https://github.com/uber/causalml/pull/236>`_)
- Add functions and methods to init for optimization module by @mwijaya3 (`#228 <https://github.com/uber/causalml/pull/228>`_)
- Install GitHub Stale App to close inactive issues automatically @jeongyoonlee (`#237 <https://github.com/uber/causalml/pull/237>`_)
- Update documentation by @deeplaunch, @ccrndn, @ppstacy(`#214 <https://github.com/uber/causalml/pull/214>`_, `#231 <https://github.com/uber/causalml/pull/231>`_, `#232 <https://github.com/uber/causalml/pull/232>`_)



0.8.0 (2020-07-17)
------------------
CausalML surpassed `100,000 downloads <https://pepy.tech/project/causalml>`_! Thanks for the support.

Major Updates
~~~~~~~~~~~~~
- Add value optimization to `optimize` by @t-tte (`#183 <https://github.com/uber/causalml/pull/183>`_)
- Add counterfactual unit selection to `optimize` by @t-tte (`#184 <https://github.com/uber/causalml/pull/184>`_)
- Add sensitivity analysis to `metrics` by @ppstacy (`#199 <https://github.com/uber/causalml/pull/199>`_, `#212 <https://github.com/uber/causalml/pull/212>`_)
- Add the `iv` estimator submodule and add 2SLS model to it by @huigangchen (`#201 <https://github.com/uber/causalml/pull/201>`_)

Minor Updates
~~~~~~~~~~~~~
- Add `GradientBoostedPropensityModel` by @yungmsh (`#193 <https://github.com/uber/causalml/pull/193>`_)
- Add covariate balance visualization by @yluogit (`#200 <https://github.com/uber/causalml/pull/200>`_)
- Fix bug in the X learner propensity model by @ppstacy (`#209 <https://github.com/uber/causalml/pull/209>`_)
- Update package dependencies by @jeongyoonlee (`#195 <https://github.com/uber/causalml/pull/195>`_, `#197 <https://github.com/uber/causalml/pull/197>`_)
- Update documentation by @jeongyoonlee, @ppstacy and @yluogit (`#181 <https://github.com/uber/causalml/pull/181>`_, `#202 <https://github.com/uber/causalml/pull/202>`_, `#205 <https://github.com/uber/causalml/pull/205>`_)



0.7.1 (2020-05-07)
------------------
Special thanks to our new community contributor, Katherine (`@khof312 <https://github.com/khof312>`_)!

Major Updates
~~~~~~~~~~~~~
- Adjust matching distances by a factor of the number of matching columns in propensity score matching by @yungmsh (`#157 <https://github.com/uber/causalml/pull/157>`_)
- Add TMLE-based AUUC/Qini/lift calculation and plotting by @ppstacy (`#165 <https://github.com/uber/causalml/pull/165>`_)

Minor Updates
~~~~~~~~~~~~~
- Fix typos and update documents by @paullo0106, @khof312, @jeongyoonlee (`#150 <https://github.com/uber/causalml/pull/150>`_, `#151 <https://github.com/uber/causalml/pull/151>`_, `#155 <https://github.com/uber/causalml/pull/155>`_, `#163 <https://github.com/uber/causalml/pull/163>`_)
- Fix error in `UpliftTreeClassifier.kl_divergence()` for `pk == 1 or 0` by @jeongyoonlee (`#169 <https://github.com/uber/causalml/pull/169>`_)
- Fix error in `BaseRRegressor.fit()` without propensity score input by @jeongyoonlee (`#170 <https://github.com/uber/causalml/pull/170>`_)


0.7.0 (2020-02-28)
------------------
Special thanks to our new community contributor, Steve (`@steveyang90 <https://github.com/steveyang90>`_)!

Major Updates
~~~~~~~~~~~~~
- Add a new `nn` inference submodule with `DragonNet` implementation by @yungmsh
- Add a new `feature selection` submodule with filter feature selection methods by @zhenyuz0500

Minor Updates
~~~~~~~~~~~~~
- Make propensity scores optional in all meta-learners by @ppstacy
- Replace `eli5` permutation importance with `sklearn`'s by @yluogit
- Replace `ElasticNetCV` with `LogisticRegressionCV` in `propensity.py` by @yungmsh
- Fix the normalized uplift curve plot with negative ATE by @jeongyoonlee
- Fix the TravisCI FOSSA error for PRs from forked repo by @steveyang90
- Add documentation about tree visualization by @zhenyuz0500

0.6.0 (2019-12-31)
------------------
Special thanks to our new community contributors, Fritz (`@fritzo <https://github.com/fritzo>`_), Peter (`@peterfoley <https://github.com/peterfoley>`_) and Tomasz (`@TomaszZamacinski <https://github.com/TomaszZamacinski>`_)!

- Improve `UpliftTreeClassifier`'s speed by 4 times by @jeongyoonlee
- Fix impurity computation in `CausalTreeRegressor` by @TomaszZamacinski
- Fix XGBoost related warnings by @peterfoley
- Fix typos and improve documentation by @peterfoley and @fritzo

0.5.0 (2019-11-26)
------------------
Special thanks to our new community contributors, Paul (`@paullo0106 <https://github.com/paullo0106>`_) and Florian (`@FlorianWilhelm <https://github.com/FlorianWilhelm>`_)!

- Add `TMLELearner`, targeted maximum likelihood estimator to `inference.meta` by @huigangchen
- Add an option to DGPs for regression to simulate imbalanced propensity distribution by @huigangchen
- Fix incorrect edge connections, and add more information in the uplift tree plot by @paullo0106
- Fix an installation error related to `Cython` and `numpy` by @FlorianWilhelm
- Drop Python 2 support from `setup.py` by @jeongyoonlee
- Update `causaltree.pyx` Cython code to be compatible with `scikit-learn>=0.21.0` by @jeongyoonlee

0.4.0 (2019-10-21)
------------------

- Add `uplift_tree_plot()` to `inference.tree` to visualize `UpliftTreeClassifier` by @zhenyuz0500
- Add the `Explainer` class to `inference.meta` to provide feature importances using `SHAP` and `eli5`'s `PermutationImportance` by @yungmsh
- Add bootstrap confidence intervals for the average treatment effect estimates of meta learners by @ppstacy

0.3.0 (2019-09-17)
------------------

- Extend meta-learners to support classification by @t-tte
- Extend meta-learners to support multiple treatments by @yungmsh
- Fix a bug in uplift curves and add Qini curves/scores to `metrics` by @jeongyoonlee
- Add `inference.meta.XGBRRegressor` with early stopping and ranking optimization by @yluogit

0.2.0 (2019-08-12)
------------------

- Add `optimize.PolicyLearner` based on Athey and Wager 2017 :cite:`athey2017efficient`
- Add the `CausalTreeRegressor` estimator based on Athey and Imbens 2016 :cite:`athey2016recursive` (experimental)
- Add missing imports in `features.py` to enable label encoding with grouping of rare values in `LabelEncoder()`
- Fix a bug that caused the mismatch between training and prediction features in `inference.meta.tlearner.predict()`

0.1.0 (unreleased)
------------------

- Initial release with the Uplift Random Forest, and S/T/X/R-learners.
