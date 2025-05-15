<div align="center">
  <a href="https://github.com/uber/causalml"><img width="380px" height="140px" src="https://raw.githubusercontent.com/uber/causalml/master/docs/_static/img/logo/causalml_logo.png"></a>
</div>

------------------------------------------------------

[![PyPI Version](https://badge.fury.io/py/causalml.svg)](https://pypi.org/project/causalml/)
[![Build Status](https://github.com/uber/causalml/actions/workflows/python-test.yaml/badge.svg)](https://github.com/uber/causalml/actions/workflows/python-test.yaml)
[![Documentation Status](https://readthedocs.org/projects/causalml/badge/?version=latest)](http://causalml.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/causalml)](https://pepy.tech/project/causalml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3015/badge)](https://bestpractices.coreinfrastructure.org/projects/3015)


# Disclaimer
This project is stable and being incubated for long-term support. It may contain new experimental code, for which APIs are subject to change.


# Causal ML: A Python Package for Uplift Modeling and Causal Inference with ML

**Causal ML** is a Python package that provides a suite of uplift modeling and causal inference methods using machine learning algorithms based on recent
research [[1]](#Literature). It provides a standard interface that allows user to estimate the Conditional Average Treatment Effect (CATE) or Individual Treatment
 Effect (ITE) from experimental or observational data. Essentially, it estimates the causal impact of intervention `T` on outcome `Y` for users
 with observed features `X`, without strong assumptions on the model form. Typical use cases include

* **Campaign targeting optimization**: An important lever to increase ROI in an advertising campaign is to target the ad to the set of customers who will have a favorable response in a given KPI such as engagement or sales. CATE identifies these customers by estimating the effect of the KPI from ad exposure at the individual level from A/B experiment or historical observational data.

* **Personalized engagement**: A company has multiple options to interact with its customers such as different product choices in up-sell or messaging channels for communications. One can use CATE to estimate the heterogeneous treatment effect for each customer and treatment option combination for an optimal personalized recommendation system.


# Documentation

Documentation is available at:

https://causalml.readthedocs.io/en/latest/about.html


# Installation

Installation instructions are available at:

https://causalml.readthedocs.io/en/latest/installation.html


# Quickstart

Quickstarts with code-snippets are available at:

https://causalml.readthedocs.io/en/latest/quickstart.html


# Example Notebooks

Example notebooks are available at:

https://causalml.readthedocs.io/en/latest/examples.html


# Contributing

We welcome community contributors to the project. Before you start, please read our [code of conduct](https://github.com/uber/causalml/blob/master/CODE_OF_CONDUCT.md) and check out [contributing guidelines](./CONTRIBUTING.md) first.


# Versioning

We document versions and changes in our [changelog](https://github.com/uber/causalml/blob/master/docs/changelog.rst).


# License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/uber/causalml/blob/master/LICENSE) file for details.


# References

## Documentation
* [Causal ML API documentation](https://causalml.readthedocs.io/en/latest/about.html)

## Workshops, Talks, and Publications
* (Workshop) [3rd Workshop on Causal Inference and Machine Learning in Practice](https://causal-machine-learning.github.io/kdd2025-workshop/) at KDD 2025
* (Workshop) [2nd Workshop on Causal Inference and Machine Learning in Practice](https://causal-machine-learning.github.io/kdd2024-workshop/) at KDD 2024
* (Workshop) [Causal Inference and Machine Learning in Practice: Use cases for Product, Brand, Policy and Beyond](https://causal-machine-learning.github.io/kdd2023-workshop/) at KDD 2023
* (Talk) Introduction to CausalML at [Causal Data Science Meeting 2021](https://www.causalscience.org/meeting/program/day-2/)
* (Talk) Introduction to CausalML at [2021 Conference on Digital Experimentation @ MIT (CODE@MIT)](https://ide.mit.edu/events/2021-conference-on-digital-experimentation-mit-codemit/)
* (Tutorial) [Causal Inference and Machine Learning in Practice with EconML and CausalML: Industrial Use Cases at Microsoft, TripAdvisor, Uber](https://causal-machine-learning.github.io/kdd2021-tutorial/) at KDD 2021
* (Publication) [CausalML: Python package for causal machine learning](https://arxiv.org/abs/2002.11631)
* (Publication) [Uplift Modeling for Multiple Treatments with Cost Optimization](https://ieeexplore.ieee.org/document/8964199) at [2019 IEEE International Conference on Data Science and Advanced Analytics (DSAA)](http://203.170.84.89/~idawis33/dsaa2019/preliminary-program/)
* (Publication) [Feature Selection Methods for Uplift Modeling](https://arxiv.org/abs/2005.03447)

## Citation
To cite CausalML in publications, you can refer to the following sources:

Whitepaper:
[CausalML: Python Package for Causal Machine Learning](https://arxiv.org/abs/2002.11631)

Bibtex:
> @misc{chen2020causalml,
>    title={CausalML: Python Package for Causal Machine Learning},
>    author={Huigang Chen and Totte Harinen and Jeong-Yoon Lee and Mike Yung and Zhenyu Zhao},
>    year={2020},
>    eprint={2002.11631},
>    archivePrefix={arXiv},
>    primaryClass={cs.CY}
>}


## Literature

1. Chen, Huigang, Totte Harinen, Jeong-Yoon Lee, Mike Yung, and Zhenyu Zhao. "Causalml: Python package for causal machine learning." arXiv preprint arXiv:2002.11631 (2020).
2. Radcliffe, Nicholas J., and Patrick D. Surry. "Real-world uplift modelling with significance-based uplift trees." White Paper TR-2011-1, Stochastic Solutions (2011): 1-33.
3. Zhao, Yan, Xiao Fang, and David Simchi-Levi. "Uplift modeling with multiple treatments and general response types." Proceedings of the 2017 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2017.
4. Hansotia, Behram, and Brad Rukstales. "Incremental value modeling." Journal of Interactive Marketing 16.3 (2002): 35-46.
5. Jannik Rößler, Richard Guse, and Detlef Schoder. "The Best of Two Worlds: Using Recent Advances from Uplift Modeling and Heterogeneous Treatment Effects to Optimize Targeting Policies". International Conference on Information Systems (2022)
6. Su, Xiaogang, et al. "Subgroup analysis via recursive partitioning." Journal of Machine Learning Research 10.2 (2009).
7. Su, Xiaogang, et al. "Facilitating score and causal inference trees for large observational studies." Journal of Machine Learning Research 13 (2012): 2955.
8. Athey, Susan, and Guido Imbens. "Recursive partitioning for heterogeneous causal effects." Proceedings of the National Academy of Sciences 113.27 (2016): 7353-7360.
9. Künzel, Sören R., et al. "Metalearners for estimating heterogeneous treatment effects using machine learning." Proceedings of the national academy of sciences 116.10 (2019): 4156-4165.
10. Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous treatment effects." arXiv preprint arXiv:1712.04912 (2017).
11. Bang, Heejung, and James M. Robins. "Doubly robust estimation in missing data and causal inference models." Biometrics 61.4 (2005): 962-973.
12. Van Der Laan, Mark J., and Daniel Rubin. "Targeted maximum likelihood learning." The international journal of biostatistics 2.1 (2006).
13. Kennedy, Edward H. "Optimal doubly robust estimation of heterogeneous causal effects." arXiv preprint arXiv:2004.14497 (2020).
14. Louizos, Christos, et al. "Causal effect inference with deep latent-variable models." arXiv preprint arXiv:1705.08821 (2017).
15. Shi, Claudia, David M. Blei, and Victor Veitch. "Adapting neural networks for the estimation of treatment effects." 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), 2019.
16. Zhao, Zhenyu, Yumin Zhang, Totte Harinen, and Mike Yung. "Feature Selection Methods for Uplift Modeling." arXiv preprint arXiv:2005.03447 (2020).
17. Zhao, Zhenyu, and Totte Harinen. "Uplift modeling for multiple treatments with cost optimization." In 2019 IEEE International Conference on Data Science and Advanced Analytics (DSAA), pp. 422-431. IEEE, 2019.


## Related projects

* [uplift](https://cran.r-project.org/web/packages/uplift/index.html): uplift models in R
* [grf](https://cran.r-project.org/web/packages/grf/index.html): generalized random forests that include heterogeneous treatment effect estimation in R
* [rlearner](https://github.com/xnie/rlearner): A R package that implements R-Learner
* [DoWhy](https://github.com/Microsoft/dowhy):  Causal inference in Python based on Judea Pearl's do-calculus
* [EconML](https://github.com/microsoft/EconML): A Python package that implements heterogeneous treatment effect estimators from econometrics and machine learning methods
