import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from scipy.special import expit, logit


# ------ Define a list of functions for feature transformation
def _f_linear(x):
    """
    Linear transformation (actually identical transformation)
    """
    return np.array(x)


def _f_quadratic(x):
    """
    Quadratic transformation
    """
    return np.array(x) * np.array(x)


def _f_cubic(x):
    """
    Quadratic transformation
    """
    return np.array(x) * np.array(x) * np.array(x)


def _f_relu(x):
    """
    Relu transformation
    """
    x = np.array(x)
    return np.maximum(x, 0)


def _f_sin(x):
    """
    Sine transformation
    """
    return np.sin(np.array(x) * np.pi)


def _f_cos(x):
    """
    Cosine transformation
    """
    return np.cos(np.array(x) * np.pi)


# ------ Generating non-linear splines as feature transformation functions
def _generate_splines(
    n_functions=10,
    n_initial_points=10,
    s=0.01,
    x_min=-3,
    x_max=3,
    y_min=0,
    y_max=1,
    random_seed=2019,
):
    """
    Generate a list of spline functions for feature
    transformation.

    Parameters
    ----------
    n_functions : int, optional
        Number of spline functions to be created.
    n_initial_points: int, optional
        Number of initial random points to be placed on a 2D plot to fit a spline.
    s:  float or None, optional
        Positive smoothing factor used to choose the number of knots (arg in scipy.interpolate.UnivariateSpline).
    x_min: int or float, optional
        The minimum value of the X range.
    x_max:  int or float, optional
        The maximum value of the X range.
    y_min: int or float, optional
        The minimum value of the Y range.
    y_max: int or float, optional
        The maxium value of the Y range.
    random_seed: int, optional
        Random seed.

    Returns
    -------
    spls: list
        List of spline functions.
    """
    np.random.seed(random_seed)
    spls = []
    for i in range(n_functions):
        x = np.linspace(x_min, x_max, n_initial_points)
        y = np.random.uniform(y_min, y_max, n_initial_points)
        spl = UnivariateSpline(x, y, s=s)
        spls.append(spl)
    return spls


def _standardize(x):
    """
    Standardize a vector to be mean 0 and std 1.
    """
    return (np.array(x) - np.mean(x)) / np.std(x)


def _fixed_transformation(fs, x, f_index=0):
    """
    Transform and standardize a vector by a transformation function.
    If the given index is within the function list f_index < len(fs), then use fs[f_index] as the transformation
    function. Otherwise, randomly choose a function from the function list.

    Parameters
    ----------
    fs : list
        A collection of functions for transformation.
    x : list
        Feature values to be transformed.
    f_index : int, optional
        The function index to be used to select a transformation function.
    """
    try:
        y = fs[f_index](x)
    except IndexError:
        y = fs[np.asscalar(np.random.choice(len(fs), 1))](x)
    y = _standardize(y)
    return y


def _random_transformation(fs, x):
    """
    Transform and standardize a vector by a function randomly chosen from
    the function collection.

    Parameters
    ----------
    fs : list
        A collection of functions (splines) for transformation.
    x : list
        Feature values to be transformed.
    """
    fi = np.random.choice(range(len(fs)), 1)
    y = fs[fi[0]](x)
    y = _standardize(y)
    return y


def _softmax(z, p, xb):
    """
    Softmax function. This function is used to reversely solve the constant root value in the linear part to make the
    softmax function output mean to be a given value.

    Parameters
    ----------
    z : float
        Constant value in the linear part.
    p : float
        The target output mean value.
    xb : list
        An array, with each element as the sum of product of coefficient and feature value
    """
    sm_arr = expit(z + np.array(xb))
    res = p - np.mean(sm_arr)
    return res


# ------ Data generation function (V2) using logistic regression as underlying model
def make_uplift_classification_logistic(
    n_samples=10000,
    treatment_name=["control", "treatment1", "treatment2", "treatment3"],
    y_name="conversion",
    n_classification_features=10,
    n_classification_informative=5,
    n_classification_redundant=0,
    n_classification_repeated=0,
    n_uplift_dict={"treatment1": 2, "treatment2": 2, "treatment3": 3},
    n_mix_informative_uplift_dict={"treatment1": 1, "treatment2": 1, "treatment3": 0},
    delta_uplift_dict={"treatment1": 0.02, "treatment2": 0.05, "treatment3": -0.05},
    positive_class_proportion=0.1,
    random_seed=20200101,
    feature_association_list=["linear", "quadratic", "cubic", "relu", "sin", "cos"],
    random_select_association=True,
    error_std=0.05,
):
    """Generate a synthetic dataset for classification uplift modeling problem.

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        The number of samples to be generated for each treatment group.
    treatment_name: list, optional (default = ['control','treatment1','treatment2','treatment3'])
        The list of treatment names. The first element must be 'control' as control group, and the rest are treated as
        treatment groups.
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
    n_uplift_dict: dictionary, optional (default: {'treatment1': 2, 'treatment2': 2, 'treatment3': 3})
        Number of features for generating heterogeneous treatment effects for corresponding treatment group.
        Dictionary of {treatment_key: number_of_features_for_uplift}.
    n_mix_informative_uplift_dict: dictionary, optional (default: {'treatment1': 1, 'treatment2': 1, 'treatment3': 1})
        Number of mix features for each treatment. The mix feature is defined as a linear combination
        of a randomly selected informative classification feature and a randomly selected uplift feature.
        The mixture is made by a weighted sum (p*feature1 + (1-p)*feature2), where the weight p is drawn from a uniform
        distribution between 0 and 1.
    delta_uplift_dict: dictionary, optional (default: {'treatment1': .02, 'treatment2': .05, 'treatment3': -.05})
        Treatment effect (delta), can be positive or negative.
        Dictionary of {treatment_key: delta}.
    positive_class_proportion: float, optional (default = 0.1)
        The proportion of positive label (1) in the control group, or the mean of outcome variable for control group.
    random_seed : int, optional (default = 20200101)
        The random seed to be used in the data generation process.
    feature_association_list : list, optional (default = ['linear','quadratic','cubic','relu','sin','cos'])
        List of uplift feature association patterns to the treatment effect. For example, if the feature pattern is
        'quadratic', then the treatment effect will increase or decrease quadratically with the feature.
        The values in the list must be one of ('linear','quadratic','cubic','relu','sin','cos'). However, the same
        value can appear multiple times in the list.
    random_select_association : boolean, optional (default = True)
        How the feature patterns are selected from the feature_association_list to be applied in the data generation
        process. If random_select_association = True, then for every uplift feature, a random feature association
        pattern is selected from the list. If random_select_association = False, then the feature association pattern
        is selected from the list in turns to be applied to each feature one by one.
    error_std : float, optional (default = 0.05)
        Standard deviation to be used in the error term of the logistic regression. The error is drawn from a normal
        distribution with mean 0 and standard deviation specified in this argument.

    Returns
    -------
    df1 : DataFrame
        A data frame containing the treatment label, features, and outcome variable.
    x_name : list
        The list of feature names generated.
    """

    # Set means for each experiment group
    mean_dict = {}
    mean_dict[treatment_name[0]] = positive_class_proportion
    for treatment_key_i in treatment_name[1:]:
        mean_dict[treatment_key_i] = positive_class_proportion
        if treatment_key_i in delta_uplift_dict:
            mean_dict[treatment_key_i] += delta_uplift_dict[treatment_key_i]

    # create data frame
    df1 = pd.DataFrame()
    n = n_samples

    # set seed
    np.random.seed(seed=random_seed)

    # define feature association function list ------------------------------------------------#
    feature_association_pattern_dict = {
        "linear": _f_linear,
        "quadratic": _f_quadratic,
        "cubic": _f_cubic,
        "relu": _f_relu,
        "sin": _f_sin,
        "cos": _f_cos,
    }
    f_list = []
    for fi in feature_association_list:
        f_list.append(feature_association_pattern_dict[fi])

    # generate treatment key ------------------------------------------------#
    treatment_list = []
    for ti in treatment_name:
        treatment_list += [ti] * n
    treatment_list = np.random.permutation(treatment_list)
    df1["treatment_group_key"] = treatment_list

    # feature name list
    x_name = []

    x_informative_name = []
    x_informative_transformed = []

    # generate informative features -----------------------------------------#
    for xi in range(n_classification_informative):
        # observed feature
        x = np.random.normal(0, 1, df1.shape[0])
        x_name_i = "x" + str(len(x_name) + 1) + "_informative"
        x_name.append(x_name_i)
        x_informative_name.append(x_name_i)
        df1[x_name_i] = x
        # transformed feature that takes effect in the model
        x_name_i = x_name_i + "_transformed"
        df1[x_name_i] = _fixed_transformation(f_list, x, xi)
        x_informative_transformed.append(x_name_i)

    # generate redundant features (linear) ----------------------------------#
    # linearly combine informative ones
    for xi in range(n_classification_redundant):
        nx = (
            np.random.choice(n_classification_informative, size=1, replace=False)[0] + 1
        )
        bx = np.random.normal(0, 1, size=nx)
        fx = np.random.choice(
            n_classification_informative, size=nx, replace=False, p=None
        )
        x_name_i = "x" + str(len(x_name) + 1) + "_redundant_linear"
        for xxi in range(nx):
            x_name_i += "_x" + str(fx[xxi] + 1)
        x_name.append(x_name_i)
        x = np.zeros(df1.shape[0])
        for xxi in range(nx):
            x += bx[xxi] * df1[x_name[fx[xxi]]]
        x = _standardize(x)
        df1[x_name_i] = x

    # generate repeated features --------------------------------------------#
    # randomly select from informative ones
    for xi in range(n_classification_repeated):
        # [N] sklearn.datasets.make_classification may also draw repeated
        # features from redundant ones
        fx = np.random.choice(
            n_classification_informative, size=1, replace=False, p=None
        )
        x_name_i = "x" + str(len(x_name) + 1) + "_repeated" + "_x" + str(fx[0] + 1)
        x_name.append(x_name_i)
        df1[x_name_i] = df1[x_name[fx[0]]]

    # generate irrelevant features ------------------------------------------#
    for xi in range(
        n_classification_features
        - n_classification_informative
        - n_classification_redundant
        - n_classification_repeated
    ):
        x_name_i = "x" + str(len(x_name) + 1) + "_irrelevant"
        x_name.append(x_name_i)
        df1[x_name_i] = np.random.normal(0, 1, df1.shape[0])

    # Generate uplift features ------------------------------------------------#
    x_name_uplift_transformed_dict = dict()
    for treatment_key_i in treatment_name:
        treatment_index = df1.index[
            df1["treatment_group_key"] == treatment_key_i
        ].tolist()
        if treatment_key_i in n_uplift_dict and n_uplift_dict[treatment_key_i] > 0:
            x_name_uplift_transformed = []
            x_name_uplift = []
            for xi in range(n_uplift_dict[treatment_key_i]):
                # observed feature
                x = np.random.normal(0, 1, df1.shape[0])
                x_name_i = "x" + str(len(x_name) + 1) + "_uplift"
                x_name.append(x_name_i)
                x_name_uplift.append(x_name_i)
                df1[x_name_i] = x
                # transformed feature that takes effect in the model
                x_name_i = x_name_i + "_transformed"
                if random_select_association:
                    df1[x_name_i] = _fixed_transformation(
                        f_list, x, random.randint(0, len(f_list) - 1)
                    )
                else:
                    df1[x_name_i] = _fixed_transformation(f_list, x, xi % len(f_list))
                x_name_uplift_transformed.append(x_name_i)
            x_name_uplift_transformed_dict[treatment_key_i] = x_name_uplift_transformed

    # generate mixed informative and uplift features
    for treatment_key_i in treatment_name:
        if (
            treatment_key_i in n_mix_informative_uplift_dict
            and n_mix_informative_uplift_dict[treatment_key_i] > 0
        ):
            for xi in range(n_mix_informative_uplift_dict[treatment_key_i]):
                x_name_i = "x" + str(len(x_name) + 1) + "_mix"
                x_name.append(x_name_i)
                p_weight = np.random.uniform(0, 1)
                df1[x_name_i] = (
                    p_weight * df1[np.random.choice(x_informative_name)]
                    + (1 - p_weight) * df1[np.random.choice(x_name_uplift)]
                )

    # generate conversion probability ------------------------------------------------#
    # baseline conversion
    coef_classify = []
    for ci in range(n_classification_informative):
        rcoef = [0]
        while np.abs(rcoef) < 0.1:
            rcoef = np.random.randn(1) * np.sqrt(1.0 / n_classification_informative)
        coef_classify.append(rcoef[0])
    x_classify = df1[x_informative_transformed].values
    p1 = positive_class_proportion
    a10 = logit(p1)
    err = np.random.normal(0, error_std, df1.shape[0])
    xb_array = (x_classify * coef_classify).sum(axis=1) + err
    # solve for the constant value so that the output metric mean equal to the function input positive_class_proportion
    a1 = fsolve(_softmax, a10, args=(p1, xb_array))[0]
    df1["conversion_prob_linear"] = a1 + xb_array
    df1["control_conversion_prob_linear"] = df1["conversion_prob_linear"].values

    # uplift conversion
    for treatment_key_i in treatment_name:
        if (
            treatment_key_i in delta_uplift_dict
            and np.abs(delta_uplift_dict[treatment_key_i]) > 0.0
        ):
            treatment_index = df1.index[
                df1["treatment_group_key"] == treatment_key_i
            ].tolist()
            # coefficient
            coef_uplift = []
            for ci in range(n_uplift_dict[treatment_key_i]):
                coef_uplift.append(0.5)
            x_uplift = df1.loc[
                :, x_name_uplift_transformed_dict[treatment_key_i]
            ].values
            p2 = mean_dict[treatment_key_i]
            a20 = np.log(p2 / (1.0 - p2)) - a1
            xb_array = df1["conversion_prob_linear"].values + (
                x_uplift * coef_uplift
            ).sum(axis=1)
            xb_array_treatment = xb_array[treatment_index]
            a2 = fsolve(_softmax, a20, args=(p2, xb_array_treatment))[0]
            df1["%s_conversion_prob_linear" % (treatment_key_i)] = a2 + xb_array
            df1.loc[treatment_index, "conversion_prob_linear"] = df1.loc[
                treatment_index, "%s_conversion_prob_linear" % (treatment_key_i)
            ].values
        else:
            df1["%s_conversion_prob_linear" % (treatment_key_i)] = df1[
                "conversion_prob_linear"
            ].values

    # generate conversion probability and true treatment effect ---------------------------------#
    df1["conversion_prob"] = 1 / (1 + np.exp(-df1["conversion_prob_linear"].values))
    df1["control_conversion_prob"] = 1 / (
        1 + np.exp(-df1["control_conversion_prob_linear"].values)
    )
    for treatment_key_i in treatment_name:
        df1["%s_conversion_prob" % (treatment_key_i)] = 1 / (
            1 + np.exp(-df1["%s_conversion_prob_linear" % (treatment_key_i)].values)
        )
        df1["%s_true_effect" % (treatment_key_i)] = (
            df1["%s_conversion_prob" % (treatment_key_i)].values
            - df1["control_conversion_prob"].values
        )

    # generate Y ------------------------------------------------------------#
    df1["conversion_prob"] = np.clip(df1["conversion_prob"].values, 0, 1)
    df1[y_name] = np.random.binomial(1, df1["conversion_prob"].values)

    return df1, x_name


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
    n_uplift_increase_mix_informative_dict: dictionary, optional
        Number of positive mix features for each treatment. The positive mix feature is defined as a linear combination
        of a randomly selected informative classification feature and a randomly selected positive uplift feature.
        The linear combination is made by two coefficients sampled from a uniform distribution between -1 and 1.
        default: {'treatment1': 1, 'treatment2': 1, 'treatment3': 1}
    n_uplift_decrease_mix_informative_dict: dictionary, optional
        Number of negative mix features for each treatment. The negative mix feature is defined as a linear combination
        of a randomly selected informative classification feature and a randomly selected negative uplift feature. The
        linear combination is made by two coefficients sampled from a uniform distribution between -1 and 1.
        default: {'treatment1': 0, 'treatment2': 0, 'treatment3': 0}
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
