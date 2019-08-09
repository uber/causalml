<div align="center">
  <a href="https://github.com/uber/causalml"> <img width="380px" height="140px" src="docs/_static/img/causalml_logo.png"></a>
</div>

------------------------------------------------------

[![Build Status](https://travis-ci.com/uber/causalml.svg?token=t7jFKh1sKGtbqHWp2sGn&branch=master)](https://travis-ci.com/uber/causalml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3015/badge)](https://bestpractices.coreinfrastructure.org/projects/3015)


# Disclaimer
This project is stable and being incubated for long-term support. It may contain new experimental code, for which APIs are subject to change.

# Causal ML: A Python Package for Uplift Modeling and Causal Inference with ML

**Causal ML** is a Python package that provides a suite of uplift modeling and causal inference methods using machine learning algorithms based on recent
research. It provides a standard interface that allows user to estimate the Conditional Average Treatment Effect (CATE) or Individual Treatment
 Effect (ITE) from experimental or observational data. Essentially, it estimates the causal impact of intervention `T` on outcome `Y` for users
 with observed features `X`, without strong assumptions on the model form. Typical use cases include

* **Campaign targeting optimization**: An important lever to increase ROI in an advertising campaign is to target the ad to the set of customers who will have a favorable response in a given KPI such as engagement or sales. CATE identifies these customers by estimating the effect of the KPI from ad exposure at the individual level from A/B experiment or historical observational data.

* **Personalized engagement**: A company has multiple options to interact with its customers such as different product choices in up-sell or messaging channels for communications. One can use CATE to estimate the heterogeneous treatment effect for each customer and treatment option combination for an optimal personalized recommendation system.

The package currently supports the following methods

* **Tree-based algorithms**
    * Uplift tree/random forests on KL divergence, Euclidean Distance, and Chi-Square
    * Uplift tree/random forests on Contextual Treatment Selection
* **Meta-learner algorithms**
    * S-learner
    * T-learner
    * X-learner
    * R-learner


# Installation

## Prerequisites

Install dependencies:
```
$ pip install -r requirements.txt
```

Install from pip:

```
$ pip install causalml
```

Install from source:

```
$ git clone https://github.com/uber-common/causalml.git
$ cd causalml
$ python setup.py build_ext --inplace
$ python setup.py install
```


# Quick Start

## Average Treatment Effect Estimation with S, T, and X Learners

```python
from causalml.inference import LinearRegressionSLearner
from causalml.inference import XGBTLearner, MLPTLearner
from causalml.inference import BaseXLearner
from causalml.dataset import synthetic_data

y, X, treatment, _ = synthetic_data(mode=1, n=1000, p=5, sigma=1.0)

lr = LinearRegressionSLearner()
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

```


# Contributing

We welcome community contributors to the project. Before you start, please read our [code of conduct](./CODE_OF_CONDUCT.md) and check out [contributing guidelines](./CONTRIBUTING.md) first.


# Versioning

We document versions and changes in our [changelog](./docs/changelog.rst).


# License

This project is licensed under the Apache 2.0 License - see the [LICENSE](./LICENSE) file for details.


# References

## Papers

* Nicholas J Radcliffe and Patrick D Surry. Real-world uplift modelling with significance based uplift trees. White Paper TR-2011-1, Stochastic Solutions, 2011.
* Yan Zhao, Xiao Fang, and David Simchi-Levi. Uplift modeling with multiple treatments and general response types. Proceedings of the 2017
SIAM International Conference on Data Mining, SIAM, 2017.
* Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners for estimating heterogeneous treatment effects using machine learning.
Proceedings of the National Academy of Sciences, 2019.
* Xinkun Nie and Stefan Wager. Quasi-Oracle Estimation of Heterogeneous Treatment Effects. Atlantic Causal Inference Conference, 2018.

## Related projects

* [uplift](https://cran.r-project.org/web/packages/uplift/index.html): uplift models in R
* [grf](https://cran.r-project.org/web/packages/grf/index.html): generalized random forests that include heterogeneous treatment effect estimation in R
* [rlearner](https://github.com/xnie/rlearner): A R package that implements R-Learner
* [DoWhy](https://github.com/Microsoft/dowhy):  Causal inference in Python based on Judea Pearl's do-calculus
* [EconML](https://github.com/microsoft/EconML): A Python package that implements heterogeneous treatment effect estimators from econometrics and machine learning methods
