import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_uplift_classification(
    n_samples=1000,
    treatment_name=["control", "treatment1", "treatment2", "treatment3"],
    y_name="conversion",
    n_classification_features=10,
    n_classification_informative=5,
    n_classification_redundant=0,
    n_classification_repeated=0,
    n_uplift_increase_dict={"treatment1": 2, "treatment2": 2, "treatment3": 2},
    n_uplift_decrease_dict={"treatment1": 0, "treatment2": 0, "treatment3": 0},
    delta_uplift_increase_dict={
        "treatment1": 0.02,
        "treatment2": 0.05,
        "treatment3": 0.1,
    },
    delta_uplift_decrease_dict={
        "treatment1": 0.0,
        "treatment2": 0.0,
        "treatment3": 0.0,
    },
    n_uplift_increase_mix_informative_dict={
        "treatment1": 1,
        "treatment2": 1,
        "treatment3": 1,
    },
    n_uplift_decrease_mix_informative_dict={
        "treatment1": 0,
        "treatment2": 0,
        "treatment3": 0,
    },
    positive_class_proportion=0.5,
    random_seed=20190101,
):
    """Generate a synthetic dataset for classification uplift modeling problem.

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        The number of samples to be generated for each treatment group.
    treatment_name: list, optional (default = ['control','treatment1','treatment2','treatment3'])
        The list of treatment names.
    y_name: string, optional (default = 'conversion')
        The name of the outcome variable to be used as a column in the output dataframe.
    n_classification_features: int, optional (default = 10)
        Total number of features for base classification
    n_classification_informative: int, optional (default = 5)
        Total number of informative features for base classification
    n_classification_redundant: int, optional (default = 0)
        Total number of redundant features for base classification
    n_classification_repeated: int, optional (default = 0)
        Total number of repeated features for base classification
    n_uplift_increase_dict: dictionary, optional (default: {'treatment1': 2, 'treatment2': 2, 'treatment3': 2})
        Number of features for generating positive treatment effects for corresponding treatment group.
        Dictionary of {treatment_key: number_of_features_for_increase_uplift}.
    n_uplift_decrease_dict: dictionary, optional (default: {'treatment1': 0, 'treatment2': 0, 'treatment3': 0})
        Number of features for generating negative treatment effects for corresponding treatment group.
        Dictionary of {treatment_key: number_of_features_for_increase_uplift}.
    delta_uplift_increase_dict: dictionary, optional (default: {'treatment1': .02, 'treatment2': .05, 'treatment3': .1})
        Positive treatment effect created by the positive uplift features on the base classification label.
        Dictionary of {treatment_key: increase_delta}.
    delta_uplift_decrease_dict: dictionary, optional (default: {'treatment1': 0., 'treatment2': 0., 'treatment3': 0.})
        Negative treatment effect created by the negative uplift features on the base classification label.
        Dictionary of {treatment_key: increase_delta}.
    n_uplift_increase_mix_informative_dict: dictionary, optional (default: {'treatment1': 1, 'treatment2': 1, 'treatment3': 1})
        Number of positive mix features for each treatment. The positive mix feature is defined as a linear combination
        of a randomly selected informative classification feature and a randomly selected positive uplift feature.
        The linear combination is made by two coefficients sampled from a uniform distribution between -1 and 1.
    n_uplift_decrease_mix_informative_dict: dictionary, optional (default: {'treatment1': 0, 'treatment2': 0, 'treatment3': 0})
        Number of negative mix features for each treatment. The negative mix feature is defined as a linear combination
        of a randomly selected informative classification feature and a randomly selected negative uplift feature. The
        linear combination is made by two coefficients sampled from a uniform distribution between -1 and 1.
    positive_class_proportion: float, optional (default = 0.5)
        The proportion of positive label (1) in the control group.
    random_seed : int, optional (default = 20190101)
        The random seed to be used in the data generation process.

    Returns
    -------
    df_res : DataFrame
        A data frame containing the treatment label, features, and outcome variable.
    x_name : list
        The list of feature names generated.

    Notes
    -----
    The algorithm for generating the base classification dataset is adapted from the make_classification method in the
    sklearn package, that uses the algorithm in Guyon [1] designed to generate the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
            selection benchmark", 2003.
    """
    # set seed
    np.random.seed(seed=random_seed)

    # create data frame
    df_res = pd.DataFrame()

    # generate treatment key
    n_all = n_samples * len(treatment_name)
    treatment_list = []
    for ti in treatment_name:
        treatment_list += [ti] * n_samples
    treatment_list = np.random.permutation(treatment_list)
    df_res["treatment_group_key"] = treatment_list

    # generate features and labels
    X1, Y1 = make_classification(
        n_samples=n_all,
        n_features=n_classification_features,
        n_informative=n_classification_informative,
        n_redundant=n_classification_redundant,
        n_repeated=n_classification_repeated,
        n_clusters_per_class=1,
        weights=[1 - positive_class_proportion, positive_class_proportion],
    )

    x_name = []
    x_informative_name = []
    for xi in range(n_classification_informative):
        x_name_i = "x" + str(len(x_name) + 1) + "_informative"
        x_name.append(x_name_i)
        x_informative_name.append(x_name_i)
        df_res[x_name_i] = X1[:, xi]
    for xi in range(n_classification_redundant):
        x_name_i = "x" + str(len(x_name) + 1) + "_redundant"
        x_name.append(x_name_i)
        df_res[x_name_i] = X1[:, n_classification_informative + xi]
    for xi in range(n_classification_repeated):
        x_name_i = "x" + str(len(x_name) + 1) + "_repeated"
        x_name.append(x_name_i)
        df_res[x_name_i] = X1[
            :, n_classification_informative + n_classification_redundant + xi
        ]

    for xi in range(
        n_classification_features
        - n_classification_informative
        - n_classification_redundant
        - n_classification_repeated
    ):
        x_name_i = "x" + str(len(x_name) + 1) + "_irrelevant"
        x_name.append(x_name_i)
        df_res[x_name_i] = np.random.normal(0, 1, n_all)

    # default treatment effects
    Y = Y1.copy()
    Y_increase = np.zeros_like(Y1)
    Y_decrease = np.zeros_like(Y1)

    # generate uplift (positive)
    for treatment_key_i in treatment_name:
        treatment_index = df_res.index[
            df_res["treatment_group_key"] == treatment_key_i
        ].tolist()
        if (
            treatment_key_i in n_uplift_increase_dict
            and n_uplift_increase_dict[treatment_key_i] > 0
        ):
            x_uplift_increase_name = []
            adjust_class_proportion = (delta_uplift_increase_dict[treatment_key_i]) / (
                1 - positive_class_proportion
            )
            X_increase, Y_increase = make_classification(
                n_samples=n_all,
                n_features=n_uplift_increase_dict[treatment_key_i],
                n_informative=n_uplift_increase_dict[treatment_key_i],
                n_redundant=0,
                n_clusters_per_class=1,
                weights=[1 - adjust_class_proportion, adjust_class_proportion],
            )
            for xi in range(n_uplift_increase_dict[treatment_key_i]):
                x_name_i = "x" + str(len(x_name) + 1) + "_uplift_increase"
                x_name.append(x_name_i)
                x_uplift_increase_name.append(x_name_i)
                df_res[x_name_i] = X_increase[:, xi]
            Y[treatment_index] = Y[treatment_index] + Y_increase[treatment_index]
            if n_uplift_increase_mix_informative_dict[treatment_key_i] > 0:
                for xi in range(
                    n_uplift_increase_mix_informative_dict[treatment_key_i]
                ):
                    x_name_i = "x" + str(len(x_name) + 1) + "_increase_mix"
                    x_name.append(x_name_i)
                    df_res[x_name_i] = (
                        np.random.uniform(-1, 1)
                        * df_res[np.random.choice(x_informative_name)]
                        + np.random.uniform(-1, 1)
                        * df_res[np.random.choice(x_uplift_increase_name)]
                    )

    # generate uplift (negative)
    for treatment_key_i in treatment_name:
        treatment_index = df_res.index[
            df_res["treatment_group_key"] == treatment_key_i
        ].tolist()
        if (
            treatment_key_i in n_uplift_decrease_dict
            and n_uplift_decrease_dict[treatment_key_i] > 0
        ):
            x_uplift_decrease_name = []
            adjust_class_proportion = (delta_uplift_decrease_dict[treatment_key_i]) / (
                1 - positive_class_proportion
            )
            X_decrease, Y_decrease = make_classification(
                n_samples=n_all,
                n_features=n_uplift_decrease_dict[treatment_key_i],
                n_informative=n_uplift_decrease_dict[treatment_key_i],
                n_redundant=0,
                n_clusters_per_class=1,
                weights=[1 - adjust_class_proportion, adjust_class_proportion],
            )
            for xi in range(n_uplift_decrease_dict[treatment_key_i]):
                x_name_i = "x" + str(len(x_name) + 1) + "_uplift_decrease"
                x_name.append(x_name_i)
                x_uplift_decrease_name.append(x_name_i)
                df_res[x_name_i] = X_decrease[:, xi]
            Y[treatment_index] = Y[treatment_index] - Y_decrease[treatment_index]
            if n_uplift_decrease_mix_informative_dict[treatment_key_i] > 0:
                for xi in range(
                    n_uplift_decrease_mix_informative_dict[treatment_key_i]
                ):
                    x_name_i = "x" + str(len(x_name) + 1) + "_decrease_mix"
                    x_name.append(x_name_i)
                    df_res[x_name_i] = (
                        np.random.uniform(-1, 1)
                        * df_res[np.random.choice(x_informative_name)]
                        + np.random.uniform(-1, 1)
                        * df_res[np.random.choice(x_uplift_decrease_name)]
                    )

    # truncate Y
    Y = np.clip(Y, 0, 1)

    df_res[y_name] = Y
    df_res["treatment_effect"] = Y - Y1
    return df_res, x_name
