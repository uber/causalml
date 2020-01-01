.. :changelog:

Changelog
=========

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
