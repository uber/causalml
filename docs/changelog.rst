.. :changelog:

Changelog
=========

0.13.0 (Sep 2022)
-----------------
- CausalML surpassed `1MM downloads <https://pepy.tech/project/causalml>`_ on PyPI and `3,200 stars <https://github.com/uber/causalml/stargazers>`_ on GitHub. Thanks for choosing CausalML and supporting us on GitHub.
- We have 7 new contributors @saiwing-yeung, @lixuan12315, @aldenrogers, @vincewu51, @AlkanSte, @enzoliao, and @alexander-pv. Thanks for your contributions!
- @alexander-pv revamped `CausalTreeRegressor` and added `CausalRandomForestRegressor` with more seamless integration with `scikit-learn`'s Cython tree module. He also added integration with `shap` for causal tree/ random forest interpretation. Please check out the `example notebook <https://github.com/uber/causalml/blob/master/examples/causal_trees_interpretation.ipynb>`_.
- We dropped the support for Python 3.6 and removed its test workflow.

Updates
~~~~~~~~~~~~~
- Fix typo `(% -> $)` by @saiwing-yeung in https://github.com/uber/causalml/pull/488
- Add function for calculating PNS bounds by @t-tte in https://github.com/uber/causalml/pull/482
- Fix hard coding bug by @t-tte in https://github.com/uber/causalml/pull/492
- Update README of `conda` install and instruction of maintain in conda-forge by @ppstacy in https://github.com/uber/causalml/pull/485
- Update `examples.rst` by @lixuan12315 in https://github.com/uber/causalml/pull/496
- Fix incorrect `effect_learner_objective` in `XGBRRegressor` by @jeongyoonlee in https://github.com/uber/causalml/pull/504
- Fix Filter F doesn't work with latest `statsmodels`' F test f-value format by @paullo0106 in https://github.com/uber/causalml/pull/505
- Exclude tests in `setup.py` by @aldenrogers in https://github.com/uber/causalml/pull/508
- Enabling higher orders feature importance for F filter and LR filter by @zhenyuz0500 in https://github.com/uber/causalml/pull/509
- Ate pretrain 0506 by @vincewu51 in https://github.com/uber/causalml/pull/511
- Update `methodology.rst` by @AlkanSte in https://github.com/uber/causalml/pull/518
- Fix the bug of incorrect result in qini for multiple models by @enzoliao in https://github.com/uber/causalml/pull/520
- Test `get_qini()` by @enzoliao in https://github.com/uber/causalml/pull/523
- Fixed typo in `uplift_trees_with_synthetic_data.ipynb` by @jroessler in https://github.com/uber/causalml/pull/531
- Remove Python 3.6 test from workflows by @jeongyoonlee in https://github.com/uber/causalml/pull/535
- Causal trees update by @alexander-pv in https://github.com/uber/causalml/pull/522
- Causal trees interpretation example by @alexander-pv in https://github.com/uber/causalml/pull/536


0.12.3 (Feb 2022)
-----------------
This patch is to release a version without the constraint for Shap to be abled to use for Conda.

Updates
~~~~~~~~~~~~~
- `#483 <https://github.com/uber/causalml/pull/483>`_ by @ppstacy: Modify the requirement version of Shap


0.12.2 (Feb 2022)
-----------------
This patch includes three updates by @tonkolviktor and @heiderich as follows. We also start using `black <https://black.readthedocs.io/en/stable/integrations/index.html>`_, a Python formatter. Please check out the updated `contribution guideline <https://github.com/uber/causalml/blob/master/CONTRIBUTING.md>`_ to learn how to use it.

Updates
~~~~~~~~~~~~~
- `#473 <https://github.com/uber/causalml/pull/477>`_ by @tonkolviktor: Open up the scipy dependency version
- `#476 <https://github.com/uber/causalml/pull/476>`_ by @heiderich: Use preferred backend for joblib instead of hard-coding it
- `#477 <https://github.com/uber/causalml/pull/477>`_ by @heiderich: Allow parallel prediction for UpliftRandomForestClassifier and make the joblib's preferred backend configurable


0.12.1 (Feb 2022)
-----------------
This patch includes two bug fixes for UpliftRandomForestClassifier as follows:

Updates
~~~~~~~~~~~~~
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
~~~~~~~~~~~~~
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
------------------
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
------------------
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
