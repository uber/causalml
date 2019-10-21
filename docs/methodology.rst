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

| **Stage 1**
| Estimate the average outcomes :math:`\mu(x)` with covariates :math:`X` and an indicator variable for treatment effect :math:`W`:
|
|   :math:`\mu(x) = E[Y|X=x,W=w]`
|
| using a machine learning model.
|
| **Stage 2**
| Define the CATE estimate as:
|
|   :math:`\hat\tau(x) = \hat\mu(x, W=1) - \hat\mu(x, W=0)`
|
| Including the propensity score in the model can reduce bias from regularization induced confounding :cite:`hahn2017bayesian`.
|
| When the control and treatment groups are very different in covariates, a single linear model is not sufficient to encode the different relevant dimensions and smoothness of features for the control and treatment groups :cite:`alaa2018limits`.
|

T-Learner
~~~~~~~~~

T-learner :cite:`kunzel2019metalearners` consists of two stages as follows:

| **Stage 1**
| Estimate the average outcomes :math:`\mu_0(x)` and :math:`\mu_1(x)`:
|
|   :math:`\mu_0(x) = E[Y(0)|X=x]` and
|   :math:`\mu_1(x) = E[Y(1)|X=x]`
|
| using machine learning models.
|
| **Stage 2**
| Define the CATE estimate as:
|
|   :math:`\hat\tau(x) = \hat\mu_1(x) - \hat\mu_0(x)`
|

X-Learner
~~~~~~~~~

X-learner :cite:`kunzel2019metalearners` is an extension of T-learner, and consists of three stages as follows:

| **Stage 1**
| Estimate the average outcomes :math:`\mu_0(x)` and :math:`\mu_1(x)`:
|
|   :math:`\mu_0(x) = E[Y(0)|X=x]` and
|   :math:`\mu_1(x) = E[Y(1)|X=x]`
|
| using machine learning models.
|
| **Stage 2**
| Impute the user level treatment effects, :math:`D^1_i` and :math:`D^0_j` for user :math:`i` in the treatment group based on :math:`\mu_0(x)`, and user :math:`j` in the control groups based on :math:`\mu_1(x)`:
|
|   :math:`D^1_i = Y^1_i - \hat\mu_0(X^1_i)`, and
|   :math:`D^0_i = \hat\mu_1(X^0_i)` - Y^0_i
|
| then estimate :math:`\tau_1(x) = E[D^1|X=x]`, and :math:`\tau_0(x) = E[D^0|X=x]` using machine learning models.
|
| **Stage 3**
| Define the CATE estimate by a weighted average of :math:`\tau_1(x)` and :math:`\tau_0(x)`:
|
|   :math:`\tau(x) = g(x)\tau_0(x) + (1 - g(x))\tau_1(x)`
|
| where :math:`g \in [0, 1]`. We can use propensity scores for :math:`g(x)`.
|

R-Learner
~~~~~~~~~

R-learner :cite:`nie2017quasi` uses the cross-validation out-of-fold estimates of outcomes :math:`\hat{m}^{(-i)}(x_i)` and propensity scores :math:`\hat{e}^{(-i)}(x_i)`. It consists of two stages as follows:

| **Stage 1**
| Fit :math:`\hat{m}(x)` and :math:`\hat{e}(x)` with machine learning models using cross-validation.
|
| **Stage 2**
| Estimate treatment effects by minimising the R-loss, :math:`\hat{L}_n(\tau(x))`:
|
|   :math:`\hat{L}_n(\tau(x)) = \frac{1}{n} \sum^n_{i=1}\big(\big(Y_i - \hat{m}^{(-i)}(X_i)\big) - \big(W_i - \hat{e}^{(-i)}(X_i)\big)\tau(X_i)\big)^2`
|
| where :math:`e^{(-i)}(X_i)`, etc. denote the out-of-fold held-out predictions made without using the :math:`i`-th training sample.


Tree-Based Algorithms
---------------------

Uplift Tree
~~~~~~~~~~~

The Uplift Tree approach consists of a set of methods that use a tree-based algorithm where the splitting criterion is based on differences in uplift. :cite:`Rzepakowski2012-br` proposed three different ways to quantify the gain in divergence as the result of splitting :cite:`Gutierrez2016-co`:

   :math:`D_{gain} = D_{after_split} (P^T, P^C) - D_{before_split}(P^T, P^C)`

where :math:`D` measures the divergence and :math:`P^T` and :math:`P^C` refer to the probability distribution of the outcome of interest in the treatment and control groups, respectively. Three different ways to quantify the divergence, KL, ED and Chi, are implemented in the package.

KL
~~~
The Kullback-Leibler (KL) divergence is given by:

   :math:`KL(P : Q) = \sum_{k=left, right}p_klog\frac{p_k}{q_k}`

where :math:`p` is the sample mean in the treatmet group, :math:`q` is the sample mean in the control group and :math:`k` indicates the leaf in which :math:`p` and :math:`q` are computed :cite:`Gutierrez2016-co`

ED
~~~
The Euclidean Distance is given by:

   :math:`ED(P : Q) = \sum_{k=left, right}(p_k - q_k)^2`

where the notation is the same as above.

Chi
~~~
Finally, the :math:`\chi^2`-divergence is given by:

   :math:`\chi^2(P : Q) = \sum_{k=left, right}\frac{(p_k - q_k)^2}{q_k}`

where the notation is again the same as above.

CTS
~~~

The final Uplift Tree algorithm that is implemented is the Contextual Treatment Selection (CTS) approach by :cite:`Zhao2017-kg`, where the sample splitting criterion is defined as follows:

   :math:`\hat{\Delta}_{\mu}(s) = \hat{p}(\phi_l \mid \phi) \times \max_{t=0, ..., K}\hat{y}_t(\phi_l) + \hat{p}(\phi_r \mid \phi) \times \max_{t=0, ..., K}\hat{y}_t(\phi_r) -  \max_{t=0, ..., K}\hat{y}_t(\phi)`

where :math:`\phi_l` and :math:`\phi_r` refer to the feature subspaces in the left leaf and the right leaves respectively, :math:`\hat{p}(\phi_j \mid \phi)` denotes the estimated conditional probability of a subject's being in :math:`\phi_j` given :math:`\phi`, and :math:`\hat{y}_t(\phi_j)` is the conditional expected response under treatment :math:`t`.
