import numpy as np


def get_treatment_costs(treatment, control_name, cc_dict, ic_dict):
    """
    Set the conversion and impression costs based on a dict of parameters.

    Calculate the actual cost of targeting a user with the actual treatment
    group using the above parameters.

    Params
    ------
    treatment : array, shape = (num_samples, )
        Treatment array.

    control_name, str
        Control group name as string.

    cc_dict : dict
        Dict containing the conversion cost for each treatment.

    ic_dict
        Dict containing the impression cost for each treatment.

    Returns
    -------
    conversion_cost : ndarray, shape = (num_samples, num_treatments)
        An array of conversion costs for each treatment.

    impression_cost : ndarray, shape = (num_samples, num_treatments)
        An array of impression costs for each treatment.

    conditions : list, len = len(set(treatment))
        A list of experimental conditions.
    """

    # Set the conversion costs of the treatments
    conversion_cost = np.zeros((len(treatment), len(cc_dict.keys())))
    for idx, dict_key in enumerate(cc_dict.keys()):
        conversion_cost[:, idx] = cc_dict.get(dict_key)

    # Set the impression costs of the treatments
    impression_cost = np.zeros((len(treatment), len(ic_dict.keys())))
    for idx, dict_key in enumerate(ic_dict.keys()):
        impression_cost[:, idx] = ic_dict.get(dict_key)

    # Get a sorted list of conditions
    conditions = list(set(treatment))
    conditions.remove(control_name)
    conditions_sorted = sorted(conditions)
    conditions_sorted.insert(0, control_name)

    return conversion_cost, impression_cost, conditions_sorted


def get_actual_value(
    treatment,
    observed_outcome,
    conversion_value,
    conditions,
    conversion_cost,
    impression_cost,
):
    """
    Set the conversion and impression costs based on a dict of parameters.

    Calculate the actual value of targeting a user with the actual treatment group
    using the above parameters.

    Params
    ------
    treatment : array, shape = (num_samples, )
        Treatment array.

    observed_outcome : array, shape = (num_samples, )
        Observed outcome array, aka y.

    conversion_value : array, shape = (num_samples, )
        The value of converting a given user.

    conditions : list, len = len(set(treatment))
        List of treatment conditions.

    conversion_cost : array, shape = (num_samples, num_treatment)
        Array of conversion costs for each unit in each treatment.

    impression_cost : array, shape = (num_samples, num_treatment)
        Array of impression costs for each unit in each treatment.

    Returns
    -------
    actual_value : array, shape = (num_samples, )
        Array of actual values of havng a user in their actual treatment group.

    conversion_value : array, shape = (num_samples, )
        Array of payoffs from converting a user.
    """

    cost_filter = [
        actual_group == possible_group
        for actual_group in treatment
        for possible_group in conditions
    ]

    conversion_cost_flat = conversion_cost.flatten()
    actual_cc = conversion_cost_flat[cost_filter]
    impression_cost_flat = impression_cost.flatten()
    actual_ic = impression_cost_flat[cost_filter]

    # Calculate the actual value of having a user in their actual treatment
    actual_value = (conversion_value - actual_cc) * observed_outcome - actual_ic

    return actual_value


def get_uplift_best(cate, conditions):
    """
    Takes the CATE prediction from a learner, adds the control
    outcome array and finds the name of the argmax conditon.

    Params
    ------
    cate : array, shape = (num_samples, )
        The conditional average treatment effect prediction.

    conditions : list, len = len(set(treatment))

    Returns
    -------
    uplift_recomm_name : array, shape = (num_samples, )
        The experimental group recommended by the learner.
    """
    cate_with_control = np.c_[np.zeros(cate.shape[0]), cate]
    uplift_best_idx = np.argmax(cate_with_control, axis=1)
    uplift_best_name = [conditions[idx] for idx in uplift_best_idx]

    return uplift_best_name
