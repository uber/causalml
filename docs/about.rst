About CausalML
===========================

``CausalML`` is a Python package that provides a suite of uplift modeling and causal inference methods using machine learning algorithms based on recent research.
It provides a standard interface that allows user to estimate the **Conditional Average Treatment Effect** (CATE), also known as **Individual Treatment Effect** (ITE), from experimental or observational data.
Essentially, it estimates the causal impact of intervention **W** on outcome **Y** for users with observed features **X**, without strong assumptions on the model form.

GitHub Repo
-----------

https://github.com/uber/causalml

Mission
-------

From the CausalML `Charter <https://github.com/uber/causalml/blob/master/CHARTER.md>`_:

    CausalML is committed to democratizing causal machine learning through accessible, innovative, and well-documented open-source tools that empower data scientists, researchers, and organizations. At our core, we embrace inclusivity and foster a vibrant community where members exchange ideas, share knowledge, and collaboratively shape a future where CausalML drives advancements across diverse domains.

Contributing
------------
`Contributing.md <https://github.com/uber/causalml/blob/master/CONTRIBUTING.md>`_

Governance
----------
* `Charter <https://github.com/uber/causalml/blob/master/CHARTER.md>`_
* `Contributors <https://github.com/uber/causalml/graphs/contributors>`_
* `Maintainers <https://github.com/uber/causalml/blob/master/MAINTAINERS.md>`_

Intro to Causal Machine Learning
================================

What is Causal Machine Learning?
--------------------------------

Causal machine learning is a branch of machine learning that focuses on understanding the cause and effect relationships in data. It goes beyond just predicting outcomes based on patterns in the data, and tries to understand how changing one variable can affect an outcome.
Suppose we are trying to predict a student’s test score based on how many hours they study and how much sleep they get. Traditional machine learning models would find patterns in the data, like students who study more or sleep more tend to get higher scores.
But what if you want to know what would happen if a student studied an extra hour each day? Or slept an extra hour each night? Modeling these potential outcomes or counterfactuals is where causal machine learning comes in. It tries to understand cause-and-effect relationships - how much changing one variable (like study hours or sleep hours) will affect the outcome (the test score).
This is useful in many fields, including economics, healthcare, and policy making, where understanding the impact of interventions is crucial.
While traditional machine learning is great for prediction, causal machine learning helps us understand the difference in outcomes due to interventions.



Difference from Traditional Machine Learning
--------------------------------------------

Traditional machine learning and causal machine learning are both powerful tools, but they serve different purposes and answer different types of questions.
Traditional Machine Learning is primarily concerned with prediction. Given a set of input features, it learns a function from the data that can predict an outcome. It’s great at finding patterns and correlations in large datasets, but it doesn’t tell us about the cause-and-effect relationships between variables. It answers questions like “Given a patient’s symptoms, what disease are they likely to have?”
On the other hand, Causal Machine Learning is concerned with understanding the cause-and-effect relationships between variables. It goes beyond prediction and tries to answer questions about intervention: “What will happen if we change this variable?” For example, in a medical context, it could help answer questions like “What will happen if a patient takes this medication?”
In essence, while traditional machine learning can tell us “what is”, causal machine learning can help us understand “what if”. This makes causal machine learning particularly useful in fields where we need to make decisions based on data, such as policy making, economics, and healthcare.


Measuring Causal Effects
------------------------

**Randomized Control Trials (RCT)** are the gold standard for causal effect measurements.  Subjects are randomly exposed to a treatment and the Average Treatment Effect (ATE) is measured as the difference between the mean effects in the treatment and control groups.  Random assignment removes the effect of any confounders on the treatment.

If an RCT is available and the treatment effects are heterogeneous across covariates, measuring the conditional average treatment effect(CATE) can be of interest.  The CATE is an estimate of the treatment effect conditioned on all available experiment covariates and confounders.  We call these Heterogeneous Treatment Effects (HTEs).


Example Use Cases
-----------------

- **Campaign Targeting Optimization**: An important lever to increase ROI in an advertising campaign is to target the ad to the set of customers who will have a favorable response in a given KPI such as engagement or sales. CATE identifies these customers by estimating the effect of the KPI from ad exposure at the individual level from A/B experiment or historical observational data.

- **Personalized Engagement**: A company might have multiple options to interact with its customers such as different product choices in up-sell or different messaging channels for communications. One can use CATE to estimate the heterogeneous treatment effect for each customer and treatment option combination for an optimal personalized engagement experience.

