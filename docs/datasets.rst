==================
Benchmark Datasets
==================

CausalML ships loaders for three standard causal-inference benchmarks. Each is
downloaded from its original source the first time it is used and cached on disk;
none is redistributed with the package.

.. code-block:: python

    from causalml.dataset import fetch_lalonde, fetch_ihdp, fetch_twins

    lalonde = fetch_lalonde()                       # Bunch
    X, y, treatment = fetch_ihdp(replication=0, return_X_y_t=True)

Files are cached in ``~/causalml-data``, overridable with the ``CAUSALML_DATA``
environment variable or the ``data_home`` argument. Every download is verified
against a SHA256 digest, so a file that has been replaced or truncated raises
instead of being parsed as data. ``download_if_missing=False`` refuses to reach
the network, and ``causalml.dataset.clear_data_dir()`` empties the cache.

Which metric applies to which dataset
=====================================

What can be measured is a property of how the data was made, not a choice:

.. list-table::
   :header-rows: 1

   * - Dataset
     - Ground truth
     - Metrics that apply
   * - ``synthetic_data``, ``make_uplift_classification``
     - simulated per-unit ``tau``
     - :func:`~causalml.metrics.pehe`, :func:`~causalml.metrics.ate_error`
   * - IHDP
     - simulated ``mu0`` / ``mu1``
     - :func:`~causalml.metrics.pehe`, :func:`~causalml.metrics.ate_error`
   * - Twins
     - both twins observed
     - :func:`~causalml.metrics.pehe`, :func:`~causalml.metrics.ate_error`
   * - LaLonde
     - experimental ATE only
     - :func:`~causalml.metrics.ate_error` against that estimate
   * - Criteo, Hillstrom
     - none
     - AUUC, Qini, RATE

PEHE is only computable where the counterfactual was constructed. A ranking of
estimators on simulated outcomes is a statement about that simulation.

Datasets with a loader
======================

LaLonde (National Supported Work)
---------------------------------

A randomized job-training experiment: 445 rows, 185 treated, outcome ``re78``
(1978 earnings in dollars). Its experimental difference in means, about $1,794,
is the yardstick observational estimators are judged against
:cite:`lalonde1986evaluating`. The sample is the Dehejia-Wahba one
:cite:`dehejia1999causal`.

Source: the `causaldata <https://pypi.org/project/causaldata/>`_ package (MIT),
read at a pinned revision.

IHDP
----

Covariates from the Infant Health and Development Program randomized trial with
outcomes simulated on response surface B :cite:`hill2011bayesian`. The file holds
100 replications of the same 747 units. Each replication simulates its own
outcomes **and draws its own 672 / 75 train-test split**, so row ``i`` is a
different unit in each one and the treated count varies with it.

Results on IHDP are reported as a mean and standard error across replications::

    scores = [pehe(fetch_ihdp(replication=r).tau, predict(r)) for r in range(100)]

A single replication is not comparable to a published IHDP number.

Source: the `clinicalml/cfrnet <https://github.com/clinicalml/cfrnet>`_ lineage
(MIT), files hosted at ``fredjo.com``.

Twins
-----

Same-sex twin births from the NBER linked birth / infant death records: 11,400
pairs, 30 covariates. Treatment is being the heavier twin and the outcome is
one-year mortality. Because both twins are observed, **both potential outcomes
are measured rather than simulated** — the only dataset here whose ground truth
is not a modelling assumption :cite:`louizos2017causal`.

Two things to know before using it. The raw outcome columns hold days survived
with ``9999`` standing for "survived the year", so mortality is
``outcome < 9999``; read as a number instead, the column averages about 8,000 and
means nothing. And revealing one twin per pair is what makes this observational —
``fetch_twins`` assigns at random, giving a trial with known counterfactuals,
while the confounded variants in the literature assign from a covariate and
differ between papers. Both potential outcomes are returned as ``y0`` and ``y1``
so a caller can construct their own and say which.

Source: NBER linked birth / infant death data (US federal, public domain), via
the van der Schaar lab mirror at a pinned revision.

Datasets without a loader
=========================

The remaining benchmarks named in the v1.0 roadmap are not shipped. The rule the
loaders follow is to ship one where the source carries an explicit permissive
license or is public-domain government data, fetch it at runtime, and never
vendor or mirror it. These do not meet that bar today:

.. list-table::
   :header-rows: 1

   * - Dataset
     - Source
     - Status
   * - ACIC (2016-2019 competitions)
     - `vdorie/aciccomp <https://github.com/vdorie/aciccomp>`_
     - No license file. Available through the R packages in that repository.
   * - Jobs
     - LaLonde lineage, distributed with the Shalit et al. code
     - No license file. The experimental subset overlaps ``fetch_lalonde``.
   * - Criteo-Uplift
     - `Criteo AI Lab <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>`_
     - Non-commercial license. ``scikit-uplift`` ships ``fetch_criteo``.
   * - Hillstrom
     - `MineThatData <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>`_
     - No license file. ``scikit-uplift`` ships ``fetch_hillstrom``.

No license is not the same as no permission, and these files have been
redistributed in the literature for years. It does mean CausalML does not
redistribute them. For Criteo and Hillstrom,
`scikit-uplift <https://github.com/maks-sh/scikit-uplift>`_ provides loaders.

Loading one yourself takes the same shape as the built-in loaders:

.. code-block:: python

    import pandas as pd

    df = pd.read_csv("your_download.csv")
    X = df[feature_columns].to_numpy()
    y = df["outcome"].to_numpy()
    treatment = df["treatment"].to_numpy()

    learner.fit(X=X, treatment=treatment, y=y)
