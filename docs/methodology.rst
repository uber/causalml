===========
Methodology
===========

Meta-Learner Algorithms
-----------------------

A meta-algorithm (or meta-learner) is a framework to estimate the Conditional Average Treatment Effect (CATE) using any machine learning estimators (called base learners) :cite:`kunzel2019metalearners`.

A meta-algorithm uses either a single base learner while having the treatment indicator as a feature (e.g. S-learner), or multiple base learners separately for each of the treatment and control groups (e.g. T-learner, X-learner and R-learner).

Confidence intervals of average treatment effect estimates are calculated based on the lower bound formular (7) from :cite:`imbens2009recent`.

S-Learner
~~~~~~~~~

S-learner estimates the treatment effect using a single machine learning model as follows:

**Stage 1**

Estimate the average outcomes :math:`\mu(x)` with covariates :math:`X` and an indicator variable for treatment effect :math:`W`:

.. math::
  \mu(x) = E[Y \mid X=x, W=w]

using a machine learning model.

**Stage 2**

Define the CATE estimate as:

.. math::
   \hat\tau(x) = \hat\mu(x, W=1) - \hat\mu(x, W=0)

Including the propensity score in the model can reduce bias from regularization induced confounding :cite:`hahn2017bayesian`.

When the control and treatment groups are very different in covariates, a single linear model is not sufficient to encode the different relevant dimensions and smoothness of features for the control and treatment groups :cite:`alaa2018limits`.

T-Learner
~~~~~~~~~

T-learner :cite:`kunzel2019metalearners` consists of two stages as follows:

**Stage 1**

Estimate the average outcomes :math:`\mu_0(x)` and :math:`\mu_1(x)`:

.. math::
   \mu_0(x) = E[Y(0)|X=x] \\
   \mu_1(x) = E[Y(1)|X=x]

using machine learning models.

**Stage 2**

Define the CATE estimate as:

.. math::
   \hat\tau(x) = \hat\mu_1(x) - \hat\mu_0(x)

X-Learner
~~~~~~~~~

X-learner :cite:`kunzel2019metalearners` is an extension of T-learner, and consists of three stages as follows:

**Stage 1**

Estimate the average outcomes :math:`\mu_0(x)` and :math:`\mu_1(x)`:

.. math::
   \mu_0(x) = E[Y(0)|X=x] \\
   \mu_1(x) = E[Y(1)|X=x]

using machine learning models.

**Stage 2**

Impute the user level treatment effects, :math:`D^1_i` and :math:`D^0_j` for user :math:`i` in the treatment group based on :math:`\mu_0(x)`, and user :math:`j` in the control groups based on :math:`\mu_1(x)`:

.. math::
   D^1_i = Y^1_i - \hat\mu_0(X^1_i) \\
   D^0_i = \hat\mu_1(X^0_i) - Y^0_i

then estimate :math:`\tau_1(x) = E[D^1|X=x]`, and :math:`\tau_0(x) = E[D^0|X=x]` using machine learning models.

**Stage 3**

Define the CATE estimate by a weighted average of :math:`\tau_1(x)` and :math:`\tau_0(x)`:

.. math::
   \tau(x) = g(x)\tau_0(x) + (1 - g(x))\tau_1(x)

where :math:`g \in [0, 1]`. We can use propensity scores for :math:`g(x)`.

R-Learner
~~~~~~~~~

R-learner :cite:`nie2017quasi` uses the cross-validation out-of-fold estimates of outcomes :math:`\hat{m}^{(-i)}(x_i)` and propensity scores :math:`\hat{e}^{(-i)}(x_i)`. It consists of two stages as follows:

**Stage 1**

Fit :math:`\hat{m}(x)` and :math:`\hat{e}(x)` with machine learning models using cross-validation.

**Stage 2**

Estimate treatment effects by minimising the R-loss, :math:`\hat{L}_n(\tau(x))`:

.. math::
   \hat{L}_n(\tau(x)) = \frac{1}{n} \sum^n_{i=1}\big(\big(Y_i - \hat{m}^{(-i)}(X_i)\big) - \big(W_i - \hat{e}^{(-i)}(X_i)\big)\tau(X_i)\big)^2

where :math:`\hat{e}^{(-i)}(X_i)`, etc. denote the out-of-fold held-out predictions made without using the :math:`i`-th training sample.


Tree-Based Algorithms
---------------------

Uplift Tree
~~~~~~~~~~~

The Uplift Tree approach consists of a set of methods that use a tree-based algorithm where the splitting criterion is based on differences in uplift. :cite:`Rzepakowski2012-br` proposed three different ways to quantify the gain in divergence as the result of splitting :cite:`Gutierrez2016-co`:

.. math::
   D_{gain} = D_{after_{split}} (P^T, P^C) - D_{before_{split}}(P^T, P^C)

where :math:`D` measures the divergence and :math:`P^T` and :math:`P^C` refer to the probability distribution of the outcome of interest in the treatment and control groups, respectively. Three different ways to quantify the divergence, KL, ED and Chi, are implemented in the package.

KL
~~~
The Kullback-Leibler (KL) divergence is given by:

.. math::
   KL(P : Q) = \sum_{k=left, right}p_klog\frac{p_k}{q_k}

where :math:`p` is the sample mean in the treatment group, :math:`q` is the sample mean in the control group and :math:`k` indicates the leaf in which :math:`p` and :math:`q` are computed :cite:`Gutierrez2016-co`

ED
~~~
The Euclidean Distance is given by:

.. math::
   ED(P : Q) = \sum_{k=left, right}(p_k - q_k)^2

where the notation is the same as above.

Chi
~~~
Finally, the :math:`\chi^2`-divergence is given by:

.. math::
   \chi^2(P : Q) = \sum_{k=left, right}\frac{(p_k - q_k)^2}{q_k}

where the notation is again the same as above.

DDP
~~~

Another Uplift Tree algorithm that is implemented is the delta-delta-p (:math:`\Delta\Delta P`) approach by :cite:`hansotia2002ddp`, where the sample splitting criterion is defined as follows:

.. math::
    \Delta\Delta P=|(P^T(y|a_0)-P^C(y|a_0) - (P^T(y|a_1)-P^C(y|a_1)))|

where :math:`a_0` and :math:`a_1` are the outcomes of a Split A, :math:`y` is the selected class, and :math:`P^T` and :math:`P^C` are the response rates of treatment and control group, respectively. In other words, we first calculate the difference in the response rate in each branch (:math:`\Delta P_{left}` and :math:`\Delta P_{right}`), and subsequently, calculate their differences (:math:`\Delta\Delta P = |\Delta P_{left} - \Delta P_{right}|`).



CTS
~~~

The final Uplift Tree algorithm that is implemented is the Contextual Treatment Selection (CTS) approach by :cite:`Zhao2017-kg`, where the sample splitting criterion is defined as follows:

.. math::
   \hat{\Delta}_{\mu}(s) = \hat{p}(\phi_l \mid \phi) \times \max_{t=0, ..., K}\hat{y}_t(\phi_l) + \hat{p}(\phi_r \mid \phi) \times \max_{t=0, ..., K}\hat{y}_t(\phi_r) -  \max_{t=0, ..., K}\hat{y}_t(\phi)

where :math:`\phi_l` and :math:`\phi_r` refer to the feature subspaces in the left leaf and the right leaves respectively, :math:`\hat{p}(\phi_j \mid \phi)` denotes the estimated conditional probability of a subject's being in :math:`\phi_j` given :math:`\phi`, and :math:`\hat{y}_t(\phi_j)` is the conditional expected response under treatment :math:`t`.

Value optimization methods
--------------------------

The package supports methods for assigning treatment groups when treatments are costly. To understand the problem, it is helpful to divide populations into the following four categories:

* **Compliers**. Those who will have a favourable outcome if and only if they are treated.
* **Always-takers**. Those who will have a favourable outcome whether or not they are treated.
* **Never-takers**. Those who will never have a favourable outcome whether or not they are treated.
* **Defiers**. Those who will have a favourable outcome if and only if they are not treated.

For a more detailed discussion see e.g. :cite:`angrist2008mostly`.

Counterfactual Unit Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:cite:`ijcai2019-248` propose a method for selecting units for treatments using counterfactual logic. Suppose the following benefits for selecting units belonging to the different categories above:

* Compliers: :math:`\beta`
* Always-takers: :math:`\gamma`
* Never-takers: :math:`\theta`
* Defiers: :math:`\delta`

If :math:`X` denotes the set of individual's features, the unit selection problem can be formulated as follows:

.. math::
   argmax_X \beta P(\text{complier} \mid X) + \gamma P(\text{always-taker} \mid X) + \theta P(\text{never-taker} \mid X) + \delta P(\text{defier} \mid X)

The problem can be reformulated using counterfactual logic. Suppose :math:`W = w` indicates that an individual is treated and :math:`W = w'` indicates he or she is untreated. Similarly, let :math:`F = f` denote a favourable outcome for the individual and :math:`F = f'` an unfavourable outcome. Then the optimization problem becomes:

.. math::
   argmax_X \beta P(f_w, f'_{w'} \mid X) + \gamma P(f_w, f_{w'} \mid X) + \theta P(f'_w, f'_{w'} \mid X) + \delta P(f_{w'}, f'_{w} \mid X)

Note that the above simply follows from the definitions of the relevant users segments. :cite:`ijcai2019-248` then use counterfactual logic (:cite:`pearl2009causality`) to solve the above optimization problem under certain conditions.

N.B. The current implementation in the package is highly experimental.

Counterfactual Value Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The counterfactual value estimation method implemented in the package predicts the outcome for a unit under different treatment conditions using a standard machine learning model. The expected value of assigning a unit into a particular treatment is then given by

.. math::
   \mathbb{E}[(v - cc_w)Y_w - ic_w]

where :math:`Y_w` is the probability of a favourable event (such as conversion) under a given treatment :math:`w`, :math:`v` is the value of the favourable event, :math:`cc_w` is the cost of the treatment triggered in case of a favourable event, and :math:`ic_w` is the cost associated with the treatment whether or not the outcome is favourable. This method builds upon the ideas discussed in :cite:`zhao2019uplift`.

Selected traditional methods
----------------------------

The package supports selected traditional causal inference methods. These are usually used to conduct causal inference with observational (non-experimental) data. In these types of studies, the observed difference between the treatment and the control is in general not equal to the difference between "potential outcomes" :math:`\mathbb{E}[Y(1) - Y(0)]`. Thus, the methods below try to deal with this problem in different ways.


Matching
~~~~~~~~
The general idea in matching is to find treated and non-treated units that are as similar as possible in terms of their relevant characteristics. As such, matching methods can be seen as part of the family of causal inference approaches that try to mimic randomized controlled trials.

While there are a number of different ways to match treated and non-treated units, the most common method is to use the propensity score:

.. math::
   e_i(X_i) = P(W_i = 1 \mid X_i)

Treated and non-treated units are then matched in terms of :math:`e(X)` using some criterion of distance, such as :math:`k:1` nearest neighbours. Because matching is usually between the treated population and the control, this method estimates the average treatment effect on the treated (ATT):

.. math::
   \mathbb{E}[Y(1) \mid W = 1] - \mathbb{E}[Y(0) \mid W = 1]

See :cite:`stuart2010matching` for a discussion of the strengths and weaknesses of the different matching methods.

Inverse probability of treatment weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The inverse probability of treatment weighting (IPTW) approach uses the propensity score :math:`e` to weigh the treated and non-treated populations by the inverse of the probability of the actual treatment :math:`W`. For a binary treatment :math:`W \in \{1, 0\}`:

.. math::
   \frac{W}{e} + \frac{1 - W}{1 - e}

In this way, the IPTW approach can be seen as creating an artificial population in which the treated and non-treated units are similar in terms of their observed features :math:`X`.

One of the possible benefits of IPTW compared to matching is that less data may be discarded due to lack of overlap between treated and non-treated units. A known problem with the approach is that extreme propensity scores can generate highly variable estimators. Different methods have been proposed for trimming and normalizing the IPT weights (:cite:`https://doi.org/10.1111/1468-0262.00442`). An overview of the IPTW approach can be found in :cite:`https://doi.org/10.1002/sim.6607`.

Instrumental variables
~~~~~~~~~~~~~~~~~~~~~~

The instrumental variables approach attempts to estimate the effect of :math:`W` on :math:`Y` with the help of a third variable :math:`Z` that is correlated with :math:`W` but is uncorrelated with the error term for :math:`Y`. In other words, the instrument :math:`Z` is only related with :math:`Y` through the directed path that goes through :math:`W`. If these conditions are satisfied, the effect of :math:`W` on :math:`Y` can be estimated using the sample analog of:

.. math::
   \frac{Cov(Y_i, Z_i)}{Cov(W_i, Z_i)}

The most common method for instrumental variables estimation is the two-stage least squares (2SLS). In this approach, the cause variable :math:`W` is first regressed on the instrument :math:`Z`. Then, in the second stage, the outcome of interest :math:`Y` is regressed on the predicted value from the first-stage model. Intuitively, the effect of :math:`W` on :math:`Y` is estimated by using only the proportion of variation in :math:`W` due to variation in :math:`Z`. See :cite:`10.1257/jep.15.4.69` for a detailed discussion of the method.
