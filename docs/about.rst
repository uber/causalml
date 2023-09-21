About Causal ML
===========================

``Causal ML`` is a Python package that provides a suite of uplift modeling and causal inference methods using machine learning algorithms based on recent research.
It provides a standard interface that allows user to estimate the **Conditional Average Treatment Effect** (CATE) or **Individual Treatment Effect** (ITE) from experimental or observational data.
Essentially, it estimates the causal impact of intervention **T** on outcome **Y** for users with observed features **X**, without strong assumptions on the model form.

Typical use cases include:

- **Campaign Targeting Optimization**: An important lever to increase ROI in an advertising campaign is to target the ad to the set of customers who will have a favorable response in a given KPI such as engagement or sales. CATE identifies these customers by estimating the effect of the KPI from ad exposure at the individual level from A/B experiment or historical observational data.
- **Personalized Engagement**: Company has multiple options to interact with its customers such as different product choices in up-sell or messaging channels for communications. One can use CATE to estimate the heterogeneous treatment effect for each customer and treatment option combination for an optimal personalized recommendation system.

The package currently supports the following methods:

- Tree-based algorithms
    - :ref:`Uplift Random Forests <Uplift Tree>` on KL divergence, Euclidean Distance, and Chi-Square
    - :ref:`Uplift Random Forests <Uplift Tree>` on Contextual Treatment Selection
    - :ref:`Uplift Random Forests <DDP>` on delta-delta-p (:math:`\Delta\Delta P`) criterion (only for binary trees and two-class problems)
    - :ref:`Uplift Random Forests <IDDP>` on IDDP (only for binary trees and two-class problems)
    - :ref:`Interaction Tree <IT>` (only for binary trees and two-class problems)
    - :ref:`Causal Inference Tree <CIT>` (only for binary trees and two-class problems)
- Meta-learner algorithms
    - :ref:`S-learner`
    - :ref:`T-learner`
    - :ref:`X-learner`
    - :ref:`R-learner`
    - :ref:`Doubly Robust (DR) learner`
- Instrumental variables algorithms
    - :ref:`2-Stage Least Squares (2SLS)`
    - :ref:`Doubly Robust Instrumental Variable (DRIV) learner`
- Neural network based algorithms
    - CEVAE
    - DragonNet
- Treatment optimization algorithms
    - :ref:`Counterfactual Unit Selection`
    - :ref:`Counterfactual Value Estimator`
