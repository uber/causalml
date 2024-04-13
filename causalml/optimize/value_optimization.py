import numpy as np


class CounterfactualValueEstimator:
    """
    Args
    ----
    treatment : array, shape = (num_samples, )
        An array of treatment group indicator values.

    control_name : string
        The name of the control condition as a string. Must be contained in the treatment array.

    treatment_names : list, length = cate.shape[1]
        A list of treatment group names. NB: The order of the items in the
        list must correspond to the order in which the conditional average
        treatment effect estimates are in cate_array.

    y_proba : array, shape = (num_samples, )
        The predicted probability of conversion using the Y ~ X model across
        the total sample.

    cate : array, shape = (num_samples, len(set(treatment)))
        Conditional average treatment effect estimations from any model.

    value : array, shape = (num_samples, )
        Value of converting each unit.

    conversion_cost : shape = (num_samples, len(set(treatment)))
        The cost of a treatment that is triggered if a unit converts after having been in the treatment, such as a
        promotion code.

    impression_cost : shape = (num_samples, len(set(treatment)))
       The cost of a treatment that is the same for each unit whether or not they convert, such as a cost associated
       with a promotion channel.


    Notes
    -----
    Because we get the conditional average treatment effects from
    cate-learners relative to the control condition, we subtract the
    cate for the unit in their actual treatment group from y_proba for that
    unit, in order to recover the control outcome. We then add the cates
    to the control outcome to obtain y_proba under each condition. These
    outcomes are counterfactual because just one of them is actually
    observed.
    """

    def __init__(
        self,
        treatment,
        control_name,
        treatment_names,
        y_proba,
        cate,
        value,
        conversion_cost,
        impression_cost,
        *args,
        **kwargs,
    ):
        self.treatment = treatment
        self.control_name = control_name
        self.treatment_names = treatment_names
        self.y_proba = y_proba
        self.cate = cate
        self.value = value
        self.conversion_cost = conversion_cost
        self.impression_cost = impression_cost

    def predict_best(self):
        """
        Predict the best treatment group based on the highest counterfactual
        value for a treatment.
        """
        self._get_counterfactuals()
        self._get_counterfactual_values()
        return self.best_treatment

    def predict_counterfactuals(self):
        """
        Predict the counterfactual values for each treatment group.
        """
        self._get_counterfactuals()
        self._get_counterfactual_values()
        return self.expected_values

    def _get_counterfactuals(self):
        """
        Get an array of counterfactual outcomes based on control outcome and
        the array of conditional average treatment effects.
        """
        conditions = self.treatment_names.copy()
        conditions.insert(0, self.control_name)
        cates_with_control = np.c_[np.zeros(self.cate.shape[0]), self.cate]
        cates_flat = cates_with_control.flatten()

        cates_filt = [
            actual_group == poss_group
            for actual_group in self.treatment
            for poss_group in conditions
        ]

        control_outcome = self.y_proba - cates_flat[cates_filt]
        self.counterfactuals = cates_with_control + control_outcome[:, None]

    def _get_counterfactual_values(self):
        """
        Calculate the expected value of assigning a unit to each of the
        treatment conditions given the value of conversion and the conversion
        and impression costs associated with the treatment.
        """

        self.expected_values = (
            self.value[:, None] - self.conversion_cost
        ) * self.counterfactuals - self.impression_cost

        self.best_treatment = np.argmax(self.expected_values, axis=1)
