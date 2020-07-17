.. :changelog:

Changelog
=========

0.8.0 (2020-07-17)
------------------
CausalML surpassed `100,000 downloads <https://pepy.tech/project/causalml>`_! Thanks for the support.

Major Updates
~~~~~~~~~~~~~
- Add value optimization to `optimize` by @t-tte (`#183 <https://github.com/uber/causalml/pull/183>`_)
- Add counterfactual unit selection to `optimize` by @t-tte (`#184 <https://github.com/uber/causalml/pull/184>`_)
- Add sensitivity analysis to `metrics` by @ppstacy (`#195 <https://github.com/uber/causalml/pull/195>`_, `#212 <https://github.com/uber/causalml/pull/212>`_)
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
- Fix typos and update documents by @paulluo0106, @khof312, @jeongyoonlee (`#150 <https://github.com/uber/causalml/pull/150>`_, `#151 <https://github.com/uber/causalml/pull/151>`_, `#155 <https://github.com/uber/causalml/pull/155>`_, `#163 <https://github.com/uber/causalml/pull/163>`_)
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
- Fix incorrect edge connections, and add more information in the uplift tree plot by @paulluo0106
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
