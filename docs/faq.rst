==========================
Frequently Asked Questions
==========================

Importing CausalML fails with an XGBoost error on macOS
-------------------------------------------------------

``xgboost`` and ``lightgbm`` need the OpenMP runtime (``libomp``), which macOS
does not ship. Since v0.16, the import error names the fix directly: install it
with ``brew install libomp`` or ``conda install -c conda-forge llvm-openmp``,
then retry. See :ref:`Installation <installation:Installation>`.

Downloading a benchmark dataset fails with ``CERTIFICATE_VERIFY_FAILED``
------------------------------------------------------------------------

The dataset loaders (:doc:`datasets`) download over HTTPS with Python's
standard library, which needs a CA certificate bundle. Some Python
installations -- notably the python.org macOS installers -- do not wire one up
by default. Run the ``Install Certificates.command`` that ships with the
python.org installer, or point Python at the ``certifi`` bundle:

.. code-block:: bash

    export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

My causal tree results changed after upgrading
----------------------------------------------

``CausalTreeRegressor`` and ``CausalRandomForestRegressor`` estimate leaves
honestly by default since #584: the tree structure is grown on one half of the
sample and leaf values are re-estimated on the other, which changes fitted
trees, leaf values and CATE estimates without any code edit. Pass
``honesty=False`` to reproduce the previous behavior, and see
:ref:`Honest estimation <methodology:Honest estimation>` for why the new
default is better on most data.

``UpliftRandomForestClassifier`` got slower after upgrading
-----------------------------------------------------------

Its default changed from ``n_jobs=-1`` (all cores) to ``n_jobs=None`` (one
worker) in #991, because each concurrent tree fit holds its own working set:
peak memory grew with the machine's core count -- 8.8x the input array at
``n_jobs=-1`` on 10 cores versus 1.4x single-threaded, in the benchmark that
motivated the change. Fitted models are identical either way. Pass
``n_jobs=-1`` explicitly to restore the previous speed if you have the memory.

My Qini or AUUC score is negative
---------------------------------

A negative score means the model's ranking performed *worse* than treating
units in random order on that data -- units it ranked as high-benefit gained
less than average. Before concluding the model is bad, compute the score on
held-out data with ``return_ci=True``: on small samples the confidence
interval is often wide enough that a negative point estimate is
indistinguishable from zero. The :doc:`validation` page gives the full
evaluation workflow, and Step 6 of the :doc:`tutorial` shows a real example.

My propensity scores pile up near 0 or 1
----------------------------------------

That signals an overlap (positivity) problem: some units essentially always or
never receive treatment given their covariates, so their counterfactual is not
represented in the data. Estimators that weight by inverse propensity become
unstable there. Common responses are trimming the non-overlapping region or
clipping the scores away from the boundaries -- and reconsidering whether the
treatment is really variable for those units. See
:ref:`Checking Overlap <validation:Checking Overlap>`.

The Twins dataset's outcome looks like earnings, not mortality
--------------------------------------------------------------

The Twins benchmark encodes survival as ``9999``, so mortality is
``outcome < 9999`` -- reading the column as a number produces a mean near
8,000 and every downstream statistic is garbage. The loader's docstring and
:doc:`datasets` record this; the loader's tests pin it.

My IHDP results differ across replications more than expected
-------------------------------------------------------------

Each IHDP replication draws its own train/test split of the same 747 units,
so rows are **not** aligned across replications -- averaging
predictions row-wise across replications compares different children. Evaluate
each replication independently and aggregate the metric, as the
:doc:`benchmark leaderboard <examples/benchmark_leaderboard>` does.

Do I need TensorFlow, PyTorch or JAX?
-------------------------------------

Only for the neural estimators: ``DragonNet`` needs the ``tf`` or ``jax``
extra and ``CEVAE`` needs ``torch`` or ``jax``. Everything else -- meta-learners,
trees, IV, evaluation -- runs without any of them. Install via
``pip install causalml[tf]`` etc.; see
:ref:`Installation <installation:Installation>`.
