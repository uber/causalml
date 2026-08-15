Examples
========

Working example notebooks are available in the `example folder <https://github.com/uber/causalml/tree/master/docs/examples>`_.

The notebooks are grouped by topic. Within each group they run from
introductory to advanced, so a group's first notebook is the place to start.

Getting started
---------------

The two canonical tours of the package: CATE estimation with the meta-learners,
and uplift modeling with the tree-based learners.

.. toctree::
    :maxdepth: 1

    examples/meta_learners_with_synthetic_data
    examples/uplift_trees_with_synthetic_data

Meta-learners
-------------

.. toctree::
    :maxdepth: 1

    examples/meta_learners_with_synthetic_data_multiple_treatment
    examples/dr_learner_with_synthetic_data

Uplift and causal trees
-----------------------

.. toctree::
    :maxdepth: 1

    examples/causal_trees_with_synthetic_data
    examples/causal_trees_with_synthetic_data_multiple_treatment_groups
    examples/causal_trees_interpretation
    examples/causal_tree_honesty_parity

Evaluation and validation
-------------------------

Validating a CATE model without ground truth: uplift and Qini curves, TMLE-based
evaluation, calibration, and sensitivity analysis.

.. toctree::
    :maxdepth: 1

    examples/validation_with_tmle
    examples/calibration
    examples/qini_curves_for_costly_treatment_arms
    examples/sensitivity_example_with_synthetic_data

Interpretation and feature selection
------------------------------------

.. toctree::
    :maxdepth: 1

    examples/feature_interpretations_example
    examples/uplift_tree_visualization
    examples/feature_selection

Neural models
-------------

DragonNet and CEVAE, each with a comparison of its available backends. These
need the optional ``tf``, ``torch`` or ``jax`` extras.

.. toctree::
    :maxdepth: 1

    examples/dragonnet_example
    examples/dragonnet_jax_vs_tf
    examples/cevae_example
    examples/cevae_jax_vs_torch

Instrumental variables
----------------------

.. toctree::
    :maxdepth: 1

    examples/iv_nlsym_synthetic_data

Policy and value optimization
-----------------------------

Turning effect estimates into treatment decisions: policy learning,
counterfactual unit selection and value optimization, and probabilities of
causation.

.. toctree::
    :maxdepth: 1

    examples/binary_policy_learner_example
    examples/counterfactual_unit_selection
    examples/counterfactual_value_optimization
    examples/necessary_and_sufficient

Datasets and benchmarks
-----------------------

The benchmark leaderboard regenerates every number published on the
:doc:`datasets` page; the simulation studies compare estimators on synthetic
and semi-synthetic data.

.. toctree::
    :maxdepth: 1

    examples/benchmark_leaderboard
    examples/benchmark_simulation_studies
    examples/benchmark_semi_synthetic_simulation_studies
    examples/logistic_regression_based_data_generation_for_uplift_classification
