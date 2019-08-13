.. :changelog:

Changelog
=========

0.2.0 (2019-08-12)
------------------

- Add `optimize.PolicyLearner` based on Athey and Wager 2017 :cite:`athey2017efficient`
- Add the `CausalTreeRegressor` estimator based on Athey and Imbens 2016 :cite:`athey2016recursive` (experimental)
- Add missing imports in `features.py` to enable label encoding with grouping of rare values in `LabelEncoder()`
- Fix a bug that caused the mismatch between training and prediction features in `inference.meta.tlearner.predict()`

0.1.0 (unreleased)
------------------

- Initial release with the Uplift Random Forest, and S/T/X/R-learners.
