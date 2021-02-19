.. :changelog:

Changelog
=========
0.10.0 (2021-02-18)
------------------
- CausalML surpassed `235,000 downloads <https://pepy.tech/project/causalml>`_!
- We have 4 new community contributors, Suraj (`@surajiyer <https://github.com/surajiyer>`_), Harsh (`@HarshCasper <https://github.com/HarshCasper>`_), Manoj (`@manojbalaji1 <https://github.com/manojbalaji1>`_) and Václav (`@vaclavbelak <https://github.com/vaclavbelak>`_). Thanks for the contribution!

Major Updates
~~~~~~~~~~~~~
- Add Policy learner, DR learner, DRIV learner by @huigangchen (`#292 <https://github.com/uber/causalml/pull/292>`_)
- Add wrapper for CEVAE, a deep latent-variable and variational autoencoder based model by @ppstacy(`#276 <https://github.com/uber/causalml/pull/276>`_)

Minor Updates
~~~~~~~~~~~~~
- Add propensity_learner to R-learner by @jeongyoonlee (`#297 <https://github.com/uber/causalml/pull/297>`_)
- Add BaseLearner class for other meta-learners to inherit from without duplicated code by @jeongyoonlee (`#295 <https://github.com/uber/causalml/pull/295>`_)
- Fix import error for sklearn>= 0.24 by @jeongyoonlee (`#283 <https://github.com/uber/causalml/pull/283>`_)
- Fix KeyError issue in Filter method for certain dataset by @surajiyer (`#281 <https://github.com/uber/causalml/pull/281>`_)
- Fix inconsistent cumlift score calculation of multiple models by @vaclavbelak (`#273 <https://github.com/uber/causalml/pull/273>`_)
- Fix duplicate values handling in feature selection method by @manojbalaji1 (`#271 <https://github.com/uber/causalml/pull/271>`_)
- Fix the color spectrum of  SHAP summary plot  for feature interpretations of meta-learners by @paullo0106 (`#269 <https://github.com/uber/causalml/pull/269>`_)
- Add IIA and value optimization related documentation by @t-tte (`#264 <https://github.com/uber/causalml/pull/264>`_)
- Fix StratifiedKFolk arguments for propensity score estimation by @paullo0106 (`#262 <https://github.com/uber/causalml/pull/262>`_)
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
