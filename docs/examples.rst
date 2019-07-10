Examples
========

Propensity Score Estimation
---------------------------

.. code-block:: python

    from causalml.propensity import ElasticNetPropensityModel

    pm = ElasticNetPropensityModel(n_fold=5, random_state=42)
    ps = pm.fit_predict(X, y)

Propensity Score Matching
-------------------------

.. code-block:: python

    from causalml.match import NearestNeigoborMatch, create_table_one

    psm = NearestNeighborMatch(replace=False,
                               ratio=1,
                               random_state=42)
    matched = psm.match_by_group(data=df,
                                 treatment_col=treatment_col,
                                 score_col=score_col,
                                 groupby_col=groupby_col)

    create_table_one(data=matched,
                     treatment_col=treatment_col,
                     features=covariates)

Average Treatment Effect (ATE) Estimation
-----------------------------------------

.. code-block:: python

    from causalml.inference.meta import LRSLearner
    from causalml.inference.meta import XGBTLearner, MLPTLearner
    from causalml.inference.meta import BaseXLearner

    lr = LRSLearner()
    te, lb, ub = lr.estimate_ate(X, treatment, y)
    logger.info('Average Treatment Effect (Linear Regression): {:.2f} ({:.2f}, {:.2f})'.format(te, lb, ub))

    xg = XGBTLearner(random_state=42)
    te, lb, ub = xg.estimate_ate(X, treatment, y)
    logger.info('Average Treatment Effect (XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te, lb, ub))

    nn = MLPTLearner(hidden_layer_sizes=(10, 10),
                     learning_rate_init=.1,
                     early_stopping=True,
                     random_state=42)
    te, lb, ub = nn.estimate_ate(X, treatment, y)
    logger.info('Average Treatment Effect (Neural Network (MLP)): {:.2f} ({:.2f}, {:.2f})'.format(te, lb, ub))

    xl = BaseXLearner(learner=XGBRegressor(random_state=42))
    te, lb, ub = xl.estimate_ate(X, p, treatment, y)
    logger.info('Average Treatment Effect (XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(te, lb, ub))

Synthetic Data Generation Process
---------------------------------

Single Simulation
~~~~~~~~~~~~~~~~~

.. code-block:: python

  from causalml.dataset.synthetic import *
  from causalml.metrics.synthetic import *

  # Generate synthetic data for single simulation
  y, X, treatment, tau, b, e = synthetic_data(mode=1)
  y, X, treatment, tau, b, e = simulate_nuisance_and_easy_treatment()

  # Generate predictions for single simulation
  single_sim_preds = get_synthetic_preds(simulate_nuisance_and_easy_treatment, n=1000)

  # Generate multiple scatter plots to compare learner performance for a single simulation
  scatter_plot_single_sim(single_sim_preds)

  # Visualize distribution of learner predictions for a single simulation
  distr_plot_single_sim(single_sim_preds, kind='kde')

Multiple Simulations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  from causalml.dataset.synthetic import *
  from causalml.metrics.synthetic import *

  # Generalize performance summary over k simulations
  num_simulations = 12
  preds_summary = get_synthetic_summary(simulate_nuisance_and_easy_treatment, n=1000, k=num_simulations)

  # Generate scatter plot of performance summary
  scatter_plot_summary(preds_summay, k=num_simulations)

  # Generate bar plot of performance summary
  bar_plot_summary(preds_summary, k=num_simulations)
