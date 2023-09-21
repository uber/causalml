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

Estimate the average outcomes :math:`\mu(x)` with covariates :math:`X` and an indicator variable for treatment :math:`W`:

.. math::
  \mu(x,w) = E[Y \mid X=x, W=w]

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

Doubly Robust (DR) learner
~~~~~~~~~~~~~~~~~~~~~~~~~~

DR-learner :cite:`kennedy2020optimal` estimates the CATE via cross-fitting a doubly-robust score function in two stages as follows. We start by randomly split the data :math:`\{Y, X, W\}` into 3 partitions :math:`\{Y^i, X^i, W^i\}, i=\{1,2,3\}`.

**Stage 1**

Fit a propensity score model :math:`\hat{e}(x)` with machine learning using :math:`\{X^1, W^1\}`, and fit outcome regression models :math:`\hat{m}_0(x)` and :math:`\hat{m}_1(x)` for treated and untreated users with machine learning using :math:`\{Y^2, X^2, W^2\}`.

**Stage 2**

Use machine learning to fit the CATE model, :math:`\hat{\tau}(X)` from the pseudo-outcome

.. math::
   \phi = \frac{W-\hat{e}(X)}{\hat{e}(X)(1-\hat{e}(X))}\left(Y-\hat{m}_W(X)\right)+\hat{m}_1(X)-\hat{m}_0(X)

with :math:`\{Y^3, X^3, W^3\}`

**Stage 3**

Repeat Stage 1 and Stage 2 again twice. First use :math:`\{Y^2, X^2, W^2\}`, :math:`\{Y^3, X^3, W^3\}`, and :math:`\{Y^1, X^1, W^1\}` for the propensity score model, the outcome models, and the CATE model. Then use :math:`\{Y^3, X^3, W^3\}`, :math:`\{Y^2, X^2, W^2\}`, and :math:`\{Y^1, X^1, W^1\}` for the propensity score model, the outcome models, and the CATE model. The final CATE model is the average of the 3 CATE models.

Doubly Robust Instrumental Variable (DRIV) learner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We combine the idea from DR-learner :cite:`kennedy2020optimal` with the doubly robust score function for LATE described in :cite:`10.1111/ectj.12097` to estimate the conditional LATE. Towards that end, we start by randomly split the data :math:`\{Y, X, W, Z\}` into 3 partitions :math:`\{Y^i, X^i, W^i, Z^i\}, i=\{1,2,3\}`.

**Stage 1**

Fit propensity score models :math:`\hat{e}_0(x)` and :math:`\hat{e}_1(x)` for assigned and unassigned users using :math:`\{X^1, W^1, Z^1\}`, and fit outcome regression models :math:`\hat{m}_0(x)` and :math:`\hat{m}_1(x)` for assigned and unassigned users with machine learning using :math:`\{Y^2, X^2, Z^2\}`. Assignment probabiliy, :math:`p_Z`, can either be user provided or come from a simple model, since in most use cases assignment is random by design.

**Stage 2**

Use machine learning to fit the conditional :ref:`LATE` model, :math:`\hat{\tau}(X)` by minimizing the following loss function

.. math::
   L(\hat{\tau}(X)) = \hat{E} &\left[\left(\hat{m}_1(X)-\hat{m}_0(X)+\frac{Z(Y-\hat{m}_1(X))}{p_Z}-\frac{(1-Z)(Y-\hat{m}_0(X))}{1-p_Z} \right.\right.\\
   &\left.\left.\quad -\Big(\hat{e}_1(X)-\hat{e}_0(X)+\frac{Z(W-\hat{e}_1(X))}{p_Z}-\frac{(1-Z)(W-\hat{e}_0(X))}{1-p_Z}\Big) \hat{\tau}(X) \right)^2\right]

with :math:`\{Y^3, X^3, W^3\}`

**Stage 3**

Similar to the DR-Learner Repeat Stage 1 and Stage 2 again twice with different permutations of partitions for estimation. The final conditional LATE model is the average of the 3 conditional LATE models.

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

IDDP
~~~~

Build upon the :math:`\Delta\Delta P` approach, the IDDP approach by :cite:`rossler2022the` is implemented, where the sample splitting
criterion is defined as follows:

.. math::
    IDDP = \frac{\Delta\Delta P^*}{I(\phi, \phi_l, \phi_r)}

where :math:`\Delta\Delta P^*` is defined as :math:`\Delta\Delta P - |E[Y(1) - Y(0)]| X \epsilon \phi|` and
:math:`I(\phi, \phi_l, \phi_r)` is defined as:

.. math::
    I(\phi, \phi_l, \phi_r) = H(\frac{n_t(\phi)} {n(\phi)}, \frac{n_c(\phi)}{n(\phi)}) * 2 \frac{1+\Delta\Delta P^*}{3} + \frac{n_t(\phi)}{n(\phi)} H(\frac{n_t(\phi_l)}{n(\phi)}, \frac{n_t(\phi_r)}{n(\phi)}) \\
    + \frac{n_c(\phi)}{n(\phi)} * H(\frac{n_c(\phi_l)}{n(\phi)}, \frac{n_c(\phi_r)}{n(\phi)}) + \frac{1}{2}

where the entropy H is defined as :math:`H(p,q)=(-p*log_2(p)) + (-q*log_2(q))` and where :math:`\phi` is a subset of the feature space
associated with the current decision node, and :math:`\phi_l` and :math:`\phi_r` are the left and right child nodes, respectively.
:math:`n_t(\phi)` is the number of treatment samples, :math:`n_c(\phi)` the number of control samples, and :math:`n(\phi)` the number
of all samples in the current (parent) node.

IT
~~

Further, the package implements the Interaction Tree (IT) proposed by :cite:`su2009subgroup`, where the sample splitting criterion
maximizes the G statistic among all permissible splits:

.. math::
    G(s^*) = max G(s)

where :math:`G(s)=t^2(s)` and :math:`t(s)` is defined as:

.. math::
    t(s) = \frac{(y^L_1 - y^L_0) - (y^R_1 - y^R_0)}{\sigma * (1/n_1 + 1/n_2 + 1/n_3 + 1/n_4)}

where :math:`\sigma=\sum_{i=4}^4w_is_i^2` is a pooled estimator of the constant variance, and :math:`w_i=(n_i-1)/\sum_{j=1}^4(n_j-1)`.
Further, :math:`y^L_1`, :math:`s^2_1`, and :math:`n_1` are the the sample mean, the sample variance, and the sample size
for the treatment group in the left child node ,respectively. Similar notation applies to the other quantities.

Note that this implementation deviates from the original implementation in that (1) the pruning techniques and (2) the validation method
for determining the best tree size are different.

CIT
~~~

Also, the package implements the Causal Inference Tree (CIT) by :cite:`su2012facilitating`, where the sample splitting
criterion calculates the likelihood ratio test statistic:

.. math::
    LRT(s) = -n_{\tau L}/2 * ln(n_{\tau L} SSE_{\tau L}) -n_{\tau R}/2 * ln(n_{\tau R} SSE_{\tau R}) + \\
    n_{\tau L1} ln n_{\tau L1} + n_{\tau L0} ln n_{\tau L0} + n_{\tau R1} ln n_{\tau R1} + n_{\tau R0} ln n_{\tau R0}

where :math:`n_{\tau}`, :math:`n_{\tau 0}`, and :math:`n_{\tau 1}` are the total number of observations in node :math:`\tau`,
the number of observations in node :math:`\tau` that are assigned to the control group, and the number of observations in node :math:`\tau`
that are assigned to the treatment group, respectively. :math:`SSE_{\tau}` is defined as:

.. math::
    SSE_{\tau} = \sum_{i \epsilon \tau: t_i=1}(y_i - \hat{y_{t1}})^2 + \sum_{i \epsilon \tau: t_i=0}(y_i - \hat{y_{t0}})^2

and :math:`\hat{y_{t0}}` and :math:`\hat{y_{t1}}` are the sample average responses of the control and treatment groups in node
:math:`\tau`, respectively.

Note that this implementation deviates from the original implementation in that (1) the pruning techniques and (2) the validation method
for determining the best tree size are different.

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

Probabilities of causation
--------------------------

A cause is said to be *necessary* for an outcome if the outcome would not have occurred in the absence of the cause. A cause is said to be *sufficient* for an outcome if the outcome would have occurred in the presence of the cause. A cause is said to be *necessary and sufficient* if both of the above two conditions hold. :cite:`tian2000probabilities` show that we can calculate bounds for the probability that a cause is of each of the above three types.

To understand how the bounds for the probabilities of causation are calculated, we need special notation to represent counterfactual quantities. Let :math:`y_t` represent the proposition “:math:`y` would occur if the treatment group was set to ‘treatment’”, :math:`y^{\prime}_c` represent the proposition “:math:`y` would not occur if the treatment group was set to ‘control’”, and similarly for the remaining two combinations of the (by assumption) binary outcome and treatment variables.

Then the probability that the treatment is *sufficient* for :math:`y` to occur can be defined as

.. math::

    PS = P(y_t \mid c, y^{\prime})

This is the probability that the :math:`y` would occur if the treatment was set to :math:`t` when in fact the treatment was set to control and the outcome did not occur.

The probability that the treatment is *necessary* for :math:`y` to occur can be defined as

.. math::
    PN = P(y^{\prime}_c \mid t, y)

This is the probability that :math:`y` would not occur if the treatment was set to control, while in actuality both :math:`y` occurs and the treatment takes place.

Finally, the probability that the treatment is both necessary and sufficient is defined as 

.. math::
    PNS = P(y_t, y^{\prime}_c)

and states that :math:`y` would occur if the treatment took place; and :math:`y` would not occur if the treatment did not take place. PNS is related with PN and PS as follows:

.. math::
    PNS = P(t, y)PN + P(c, y^{\prime})PS

In bounding the above three quantities, we utilize observational data in addition to experimental data. The observational data is characterized in terms of the joint probabilities:

.. math::
    P_{TY} = {P(t, y),  P(c, y), P(t, y^{\prime}), P(c, y^{\prime})}

Given this, :cite:`tian2000probabilities` use the program developed in :cite:`balke1995probabilistic` to obtain sharp bounds of the above three quantities. The main idea in this program is to turn the bounding task into a linear programming problem (for a modern implementation of their approach see `here <https://cran.r-project.org/web/packages/causaloptim/vignettes/vertexenum-speed.html>`_).

Using the linear programming approach and given certain constraints together with observational data, :cite:`tian2000probabilities` find that the shar lower bound for PNS is given by

.. math::
    max\{0, P(y_t) - P(y_c), P(y) - P(y_c), P(y_t) - P(y)\}

and the sharp upper bound is given by

.. math::
    min\{P(y_t), P(y^{\prime}_c), P(t, y) + P(c, y^{\prime}), P(y_t) - P(y_c) + P(t, y^{\prime}) + P(c, y)\}

They use a similar routine to find the bounds for PS and PN. The `get_pns_bounds()` function calculates the bounds for each of the three probabilities of causation using the results in :cite:`tian2000probabilities`.

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

2-Stage Least Squares (2SLS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the basic requirements for identifying the treatment effect of :math:`W` on :math:`Y` is that :math:`W` is orthogonal to the potential outcome of :math:`Y`, conditional on the covariates :math:`X`. This may be violated if both :math:`W` and :math:`Y` are affected by an unobserved variable, the error term after removing the true effect of :math:`W` from :math:`Y`, that is not in :math:`X`. In this case, the instrumental variables approach attempts to estimate the effect of :math:`W` on :math:`Y` with the help of a third variable :math:`Z` that is correlated with :math:`W` but is uncorrelated with the error term. In other words, the instrument :math:`Z` is only related with :math:`Y` through the directed path that goes through :math:`W`. If these conditions are satisfied, in the case without covariates, the effect of :math:`W` on :math:`Y` can be estimated using the sample analog of:

.. math::
   \frac{Cov(Y_i, Z_i)}{Cov(W_i, Z_i)}

The most common method for instrumental variables estimation is the two-stage least squares (2SLS). In this approach, the cause variable :math:`W` is first regressed on the instrument :math:`Z`. Then, in the second stage, the outcome of interest :math:`Y` is regressed on the predicted value from the first-stage model. Intuitively, the effect of :math:`W` on :math:`Y` is estimated by using only the proportion of variation in :math:`W` due to variation in :math:`Z`. Specifically, assume that we have the linear model

.. math::
   Y = W \alpha + X \beta + u = \Xi \gamma + u

Here for convenience we let :math:`\Xi=[W, X]` and :math:`\gamma=[\alpha', \beta']'`. Assume that we have instrumental variables :math:`Z` whose number of columns is at least the number of columns of :math:`W`, let :math:`\Omega=[Z, X]`, 2SLS estimator is as follows

.. math::
   \hat{\gamma}_{2SLS} = \left[\Xi'\Omega (\Omega'\Omega)^{-1} \Omega' \Xi\right]^{-1}\left[\Xi'\Omega'(\Omega'\Omega)^{-1}\Omega'Y\right].

See :cite:`10.1257/jep.15.4.69` for a detailed discussion of the method.

LATE
~~~~

In many situations the treatment :math:`W` may depend on subject's own choice and cannot be administered directly in an experimental setting. However one can randomly assign users into treatment/control groups so that users in the treatment group can be nudged to take the treatment. This is the case of noncompliance, where users may fail to comply with their assignment status, :math:`Z`, as to whether to take treatment or not. Similar to the section of Value optimization methods, in general there are 3 types of users in this situation,

* **Compliers** Those who will take the treatment if and only if they are assigned to the treatment group.
* **Always-Taker** Those who will take the treatment regardless which group they are assigned to.
* **Never-Taker** Those who wil not take the treatment regardless which group they are assigned to.

However one assumes that there is no Defier for identification purposes, i.e. those who will only take the treatment if they are assigned to the control group.

In this case one can measure the treatment effect of Compliers,

.. math::
   \hat{\tau}_{Complier}=\frac{E[Y|Z=1]-E[Y|Z=0]}{E[W|Z=1]-E[W|Z=0]}

This is Local Average Treatment Effect (LATE). The estimator is also equivalent to 2SLS if we take the assignment status, :math:`Z`, as an instrument.


Targeted maximum likelihood estimation (TMLE) for ATE
-----------------------------------------------------

Targeted maximum likelihood estimation (TMLE) :cite:`tmle` provides a doubly robust semiparametric method that "targets" directly on the average treatment effect with the aid from machine learning algorithms. Compared to other methods including outcome regression and inverse probability of treatment weighting, TMLE usually gives better performance especially when dealing with skewed treatment and outliers.

Given binary treatment :math:`W`, covariates :math:`X`, and outcome :math:`Y`, the TMLE for ATE is performed in the following steps

**Step 1**

Use cross fit to estimate the propensity score :math:`\hat{e}(x)`, the predicted outcome for treated :math:`\hat{m}_1(x)`, and predicted outcome for control :math:`\hat{m}_0(x)` with machine learning.

**Step 2**

Scale :math:`Y` into :math:`\tilde{Y}=\frac{Y-\min Y}{\max Y - \min Y}` so that :math:`\tilde{Y} \in [0,1]`. Use the same scale function to transform :math:`\hat{m}_i(x)` into :math:`\tilde{m}_i(x)`, :math:`i=0,1`. Clip the scaled functions so that their values stay in the unit interval.

**Step 3**

Let :math:`Q=\log(\tilde{m}_W(X)/(1-\tilde{m}_W(X)))`. Maximize the following pseudo log-likelihood function

.. math::
   \max_{h_0, h_1} -\frac{1}{N} \sum_i & \left[ \tilde{Y}_i \log \left(1+\exp(-Q_i-h_0 \frac{1-W}{1-\hat{e}(X_i)}-h_1 \frac{W}{\hat{e}(X_i)} \right) \right. \\
   &\quad\left.+(1-\tilde{Y}_i)\log\left(1+\exp(Q_i+h_0\frac{1-W}{1-\hat{e}(X_i)}+h_1\frac{W}{\hat{e}(X_i)}\right)\right]

**Step 4**

Let

.. math::
   \tilde{Q}_0 &= \frac{1}{1+\exp\left(-Q-h_0 \frac{1}{1-\hat{e}(X)}\right)},\\
   \tilde{Q}_1 &= \frac{1}{1+\exp\left(-Q-h_1 \frac{1}{\hat{e}(X)}\right)}.

The ATE estimate is the sample average of the differences of :math:`\tilde{Q}_1` and :math:`\tilde{Q}_0` after rescale to the original range.
