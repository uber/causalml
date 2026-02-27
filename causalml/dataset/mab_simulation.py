import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from scipy.special import expit
from scipy.interpolate import UnivariateSpline
import random
from scipy.optimize import fsolve
from scipy.special import logit

# ------ Define a list of functions for feature transformation
def _f_linear(x):
    """Linear transformation (identical transformation)."""
    return np.array(x)

def _f_quadratic(x):
    """Quadratic transformation."""
    return np.array(x) * np.array(x)

def _f_cubic(x):
    """Cubic transformation."""
    return np.array(x) * np.array(x) * np.array(x)

def _f_relu(x):
    """ReLU transformation."""
    x = np.array(x)
    return np.maximum(x, 0)

def _f_sin(x):
    """Sine transformation."""
    return np.sin(np.array(x) * np.pi)

def _f_cos(x):
    """Cosine transformation."""
    return np.cos(np.array(x) * np.pi)

def _standardize(x):
    """Standardize a vector to be mean 0 and std 1."""
    return (np.array(x) - np.mean(x)) / np.std(x)

def _fixed_transformation(fs, x, f_index=0):
    """Transform and standardize a vector by a transformation function.
    
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
        y = fs[np.random.choice(len(fs), 1)[0]](x)
    y = _standardize(y)
    return y

def _random_transformation(fs, x):
    """Transform and standardize a vector by a function randomly chosen from the function collection.

    Parameters
    ----------
    fs : list
        A collection of functions for transformation.
    x : list
        Feature values to be transformed.
    """
    fi = np.random.choice(range(len(fs)), 1)
    y = fs[fi[0]](x)
    y = _standardize(y)
    return y

def _softmax(z, p, xb):
    sm_arr = expit(z + np.array(xb))
    res = p - np.mean(sm_arr)
    return res

def make_mab_data(
    n_samples: int = 10000,
    n_arms: int = 4,
    n_features: int = 10,
    n_informative: int = 5,
    n_redundant: int = 0,
    n_repeated: int = 0,
    arm_effects: Dict[str, float] = None,
    positive_class_proportion: float = 0.1,
    random_seed: int = 20200101,
    feature_association_list: List[str] = ["linear", "quadratic", "cubic", "relu", "sin", "cos"],
    random_select_association: bool = True,
    error_std: float = 0.05,
    n_arm_features: Dict[str, int] = None,
    n_mixed_features: Dict[str, int] = None,
    custom_coef_arm: bool = False,
    coef_arm_dict: Dict[str, List[float]] = None,
    custom_coef_informative: bool = False,
    coef_informative_list: List[float] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Generate synthetic data for multi-armed bandit experiments.
    
    This function generates data that can be used directly with both classical and contextual MAB algorithms.
    For classical MAB, only the 'arm' and 'reward' columns are needed.
    For contextual MAB, the feature columns are also used.
    
    Parameters
    ----------
    n_samples : int, optional (default=10000)
        Number of samples to generate.
    n_arms : int, optional (default=4)
        Number of arms/treatments.
    n_features : int, optional (default=10)
        Total number of features.
    n_informative : int, optional (default=5)
        Number of informative features.
    n_redundant : int, optional (default=0)
        Number of redundant features.
    n_repeated : int, optional (default=0)
        Number of repeated features.
    arm_effects : dict, optional (default=None)
        Dictionary of arm effects. If None, random effects will be generated.
    positive_class_proportion : float, optional (default=0.1)
        Proportion of positive outcomes in the control group.
    random_seed : int, optional (default=20200101)
        Random seed for reproducibility.
    feature_association_list : list, optional (default=["linear", "quadratic", "cubic", "relu", "sin", "cos"])
        List of feature transformation functions to use.
    random_select_association : bool, optional (default=True)
        Whether to randomly select feature associations.
    error_std : float, optional (default=0.05)
        Standard deviation of the error term.
    n_arm_features : dict, optional (default=None)
        Dictionary specifying number of arm-specific features for each arm. If None, defaults to 2 features per arm.
    n_mixed_features : dict, optional (default=None)
        Dictionary specifying number of mixed features for each arm. If None, defaults to 1 feature per arm.
    custom_coef_arm : bool, optional (default=False)
        Whether to use custom coefficients for arm-specific features.
    coef_arm_dict : dict, optional (default=None)
        Dictionary of custom coefficients for arm-specific features.
    custom_coef_informative : bool, optional (default=False)
        Whether to use custom coefficients for informative features.
    coef_informative_list : list, optional (default=None)
        List of custom coefficients for informative features.
        
    Returns
    -------
    df : pd.DataFrame
        Generated dataset with the following columns:
        - arm: The arm/treatment assigned to each sample
        - reward: The binary reward (0 or 1) for each sample
        - reward_prob: The probability of reward for each sample
        - feature_*: Generated features (informative, redundant, repeated, and irrelevant)
        - feature_*_transformed: Transformed versions of informative features
    x_name : list
        List of feature names
    """
    # Create arm names
    arm_names = [f"arm_{i}" for i in range(n_arms)]
    
    # Set default arm effects if not provided
    if arm_effects is None:
        arm_effects = {arm: np.random.uniform(-0.1, 0.1) for arm in arm_names}
    
    # Set default arm-specific features if not provided
    if n_arm_features is None:
        n_arm_features = {arm: 2 for arm in arm_names[1:]}
    
    # Set default mixed features if not provided
    if n_mixed_features is None:
        n_mixed_features = {arm: 1 for arm in arm_names[1:]}

    # Set means for each experiment group
    mean_dict = {}
    mean_dict[arm_names[0]] = positive_class_proportion
    for arm in arm_names[1:]:
        mean_dict[arm] = positive_class_proportion + arm_effects[arm]

    df = pd.DataFrame()
    np.random.seed(seed=random_seed)

    feature_association_pattern_dict = {
        "linear": _f_linear,
        "quadratic": _f_quadratic,
        "cubic": _f_cubic,
        "relu": _f_relu,
        "sin": _f_sin,
        "cos": _f_cos,
    }
    f_list = [feature_association_pattern_dict[fi] for fi in feature_association_list]

    # generate treatment assignments
    treatment_list = []
    for arm in arm_names:
        treatment_list += [arm] * n_samples
    treatment_list = np.random.permutation(treatment_list)
    df["arm"] = treatment_list

    x_name = []
    x_informative_name = []
    x_informative_transformed = []

    # informative features
    for xi in range(n_informative):
        x = np.random.normal(0, 1, df.shape[0])
        x_name_i = f"feature_{len(x_name)+1}_informative"
        x_name.append(x_name_i)
        x_informative_name.append(x_name_i)
        df[x_name_i] = x
        x_name_i = x_name_i + "_transformed"
        df[x_name_i] = _fixed_transformation(f_list, x, xi)
        x_informative_transformed.append(x_name_i)

    # redundant features
    for xi in range(n_redundant):
        nx = np.random.choice(n_informative, size=1, replace=False)[0] + 1
        bx = np.random.normal(0, 1, size=nx)
        fx = np.random.choice(n_informative, size=nx, replace=False, p=None)
        x_name_i = f"feature_{len(x_name)+1}_redundant_linear"
        for xxi in range(nx):
            x_name_i += f"_x{fx[xxi]+1}"
        x_name.append(x_name_i)
        x = np.zeros(df.shape[0])
        for xxi in range(nx):
            x += bx[xxi] * df[x_name[fx[xxi]]]
        x = _standardize(x)
        df[x_name_i] = x

    # repeated features
    for xi in range(n_repeated):
        fx = np.random.choice(n_informative, size=1, replace=False, p=None)
        x_name_i = f"feature_{len(x_name)+1}_repeated_x{fx[0]+1}"
        x_name.append(x_name_i)
        df[x_name_i] = df[x_name[fx[0]]]

    # irrelevant features
    for xi in range(n_features - n_informative - n_redundant - n_repeated):
        x_name_i = f"feature_{len(x_name)+1}_irrelevant"
        x_name.append(x_name_i)
        df[x_name_i] = np.random.normal(0, 1, df.shape[0])

    # arm-specific features
    x_arm_transformed_dict = {arm: [] for arm in arm_names}
    for arm in arm_names:
        if arm in n_arm_features and n_arm_features[arm] > 0:
            for ci in range(n_arm_features[arm]):
                x_name_i = f"feature_{arm}_{ci}"
                x_name.append(x_name_i)
                df[x_name_i] = np.random.normal(0, 1, df.shape[0])
                x_arm_transformed_dict[arm].append(x_name_i)

    # mixed informative and arm-specific features
    for arm in arm_names:
        if arm in n_mixed_features and n_mixed_features[arm] > 0:
            for xi in range(n_mixed_features[arm]):
                x_name_i = f"feature_{len(x_name)+1}_mix"
                x_name.append(x_name_i)
                p_weight = np.random.uniform(0, 1)
                df[x_name_i] = (
                    p_weight * df[np.random.choice(x_informative_name)]
                    + (1 - p_weight) * df[np.random.choice(x_arm_transformed_dict[arm])]
                )

    # baseline reward probability
    coef_classify = []
    for ci in range(n_informative):
        rcoef = [0]
        while np.abs(rcoef) < 0.1:
            if custom_coef_informative and coef_informative_list is not None:
                rcoef = coef_informative_list[ci] * (1 + 0.1 * np.random.randn(1)) * np.sqrt(1.0 / n_informative)
            else:
                rcoef = 0.5 * (1 + 0.1 * np.random.randn(1)) * np.sqrt(1.0 / n_informative)
        coef_classify.append(rcoef[0])
    x_classify = df[x_informative_transformed].values
    p1 = positive_class_proportion
    a10 = logit(p1)
    err = np.random.normal(0, error_std, df.shape[0])
    xb_array = (x_classify * coef_classify).sum(axis=1) + err
    a1 = fsolve(_softmax, a10, args=(p1, xb_array))[0]
    df["reward_prob_linear"] = a1 + xb_array
    df["control_reward_prob_linear"] = df["reward_prob_linear"].values

    # arm-specific reward probability
    for arm in arm_names:
        if arm != arm_names[0]:  # Skip control arm
            treatment_index = df.index[df["arm"] == arm].tolist()
            coef_arm = []
            for ci in range(n_arm_features[arm]):
                if custom_coef_arm and coef_arm_dict is not None and arm in coef_arm_dict:
                    coef_arm.append(coef_arm_dict[arm][ci])
                else:
                    coef_arm.append(0.5)
            x_arm = df.loc[:, x_arm_transformed_dict[arm]].values
            p2 = mean_dict[arm]
            a20 = np.log(p2 / (1.0 - p2)) - a1
            xb_array = df["reward_prob_linear"].values + (x_arm * coef_arm).sum(axis=1)
            xb_array_treatment = xb_array[treatment_index]
            a2 = fsolve(_softmax, a20, args=(p2, xb_array_treatment))[0]
            df[f"{arm}_reward_prob_linear"] = a2 + xb_array
            df.loc[treatment_index, "reward_prob_linear"] = df.loc[treatment_index, f"{arm}_reward_prob_linear"].values
        else:
            df[f"{arm}_reward_prob_linear"] = df["reward_prob_linear"].values

    # generate reward probability and true treatment effect
    df["reward_prob"] = 1 / (1 + np.exp(-df["reward_prob_linear"].values))
    df["control_reward_prob"] = 1 / (1 + np.exp(-df["control_reward_prob_linear"].values))
    for arm in arm_names:
        df[f"{arm}_reward_prob"] = 1 / (1 + np.exp(-df[f"{arm}_reward_prob_linear"].values))
        df[f"{arm}_true_effect"] = df[f"{arm}_reward_prob"].values - df["control_reward_prob"].values

    # generate reward
    df["reward_prob"] = np.clip(df["reward_prob"].values, 0, 1)
    df["reward"] = np.random.binomial(1, df["reward_prob"].values)

    return df, x_name

def make_classical_mab_data(
    n_samples: int = 10000,
    n_arms: int = 4,
    arm_effects: Dict[str, float] = None,
    positive_class_proportion: float = 0.1,
    random_seed: int = 20200101,
    error_std: float = 0.05
) -> pd.DataFrame:
    """Generate synthetic data for classical multi-armed bandit experiments.
    
    This is a simplified version of make_mab_data that only generates data
    needed for classical MAB algorithms (arm and reward).
    
    Parameters
    ----------
    n_samples : int, optional (default=10000)
        Number of samples to generate.
    n_arms : int, optional (default=4)
        Number of arms/treatments.
    arm_effects : dict, optional (default=None)
        Dictionary of arm effects. If None, random effects will be generated.
    positive_class_proportion : float, optional (default=0.1)
        Proportion of positive outcomes in the control group.
    random_seed : int, optional (default=20200101)
        Random seed for reproducibility.
    error_std : float, optional (default=0.05)
        Standard deviation of the error term.
        
    Returns
    -------
    df : pd.DataFrame
        Generated dataset with the following columns:
        - arm: The arm/treatment assigned to each sample
        - reward: The binary reward (0 or 1) for each sample
        - reward_prob: The probability of reward for each sample
    """
    return make_mab_data(
        n_samples=n_samples,
        n_arms=n_arms,
        n_features=0,  # No features needed for classical MAB
        n_informative=0,
        n_redundant=0,
        n_repeated=0,
        arm_effects=arm_effects,
        positive_class_proportion=positive_class_proportion,
        random_seed=random_seed,
        error_std=error_std
    )

# ------ Spline generator

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
    np.random.seed(random_seed)
    spls = []
    for i in range(n_functions):
        x = np.linspace(x_min, x_max, n_initial_points)
        y = np.random.uniform(y_min, y_max, n_initial_points)
        spl = UnivariateSpline(x, y, s=s)
        spls.append(spl)
    return spls

# ------ New MAB data generation function (logistic model)
def make_mab_logistic(
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
    # Set means for each experiment group
    mean_dict = {}
    mean_dict[treatment_name[0]] = positive_class_proportion
    for treatment_key_i in treatment_name[1:]:
        mean_dict[treatment_key_i] = positive_class_proportion
        if treatment_key_i in delta_uplift_dict:
            mean_dict[treatment_key_i] += delta_uplift_dict[treatment_key_i]

    df1 = pd.DataFrame()
    n = n_samples
    np.random.seed(seed=random_seed)

    feature_association_pattern_dict = {
        "linear": _f_linear,
        "quadratic": _f_quadratic,
        "cubic": _f_cubic,
        "relu": _f_relu,
        "sin": _f_sin,
        "cos": _f_cos,
    }
    f_list = [feature_association_pattern_dict[fi] for fi in feature_association_list]

    # generate treatment key
    treatment_list = []
    for ti in treatment_name:
        treatment_list += [ti] * n
    treatment_list = np.random.permutation(treatment_list)
    df1["treatment_group_key"] = treatment_list

    x_name = []
    x_informative_name = []
    x_informative_transformed = []

    # informative features
    for xi in range(n_classification_informative):
        x = np.random.normal(0, 1, df1.shape[0])
        x_name_i = f"x{len(x_name)+1}_informative"
        x_name.append(x_name_i)
        x_informative_name.append(x_name_i)
        df1[x_name_i] = x
        x_name_i = x_name_i + "_transformed"
        df1[x_name_i] = _fixed_transformation(f_list, x, xi)
        x_informative_transformed.append(x_name_i)

    # redundant features
    for xi in range(n_classification_redundant):
        nx = np.random.choice(n_classification_informative, size=1, replace=False)[0] + 1
        bx = np.random.normal(0, 1, size=nx)
        fx = np.random.choice(n_classification_informative, size=nx, replace=False, p=None)
        x_name_i = f"x{len(x_name)+1}_redundant_linear"
        for xxi in range(nx):
            x_name_i += f"_x{fx[xxi]+1}"
        x_name.append(x_name_i)
        x = np.zeros(df1.shape[0])
        for xxi in range(nx):
            x += bx[xxi] * df1[x_name[fx[xxi]]]
        x = _standardize(x)
        df1[x_name_i] = x

    # repeated features
    for xi in range(n_classification_repeated):
        fx = np.random.choice(n_classification_informative, size=1, replace=False, p=None)
        x_name_i = f"x{len(x_name)+1}_repeated_x{fx[0]+1}"
        x_name.append(x_name_i)
        df1[x_name_i] = df1[x_name[fx[0]]]

    # irrelevant features
    for xi in range(
        n_classification_features
        - n_classification_informative
        - n_classification_redundant
        - n_classification_repeated
    ):
        x_name_i = f"x{len(x_name)+1}_irrelevant"
        x_name.append(x_name_i)
        df1[x_name_i] = np.random.normal(0, 1, df1.shape[0])

    # uplift features
    x_name_uplift_transformed_dict = dict()
    for treatment_key_i in treatment_name:
        treatment_index = df1.index[df1["treatment_group_key"] == treatment_key_i].tolist()
        if treatment_key_i in n_uplift_dict and n_uplift_dict[treatment_key_i] > 0:
            x_name_uplift_transformed = []
            x_name_uplift = []
            for xi in range(n_uplift_dict[treatment_key_i]):
                x = np.random.normal(0, 1, df1.shape[0])
                x_name_i = f"x{len(x_name)+1}_uplift"
                x_name.append(x_name_i)
                x_name_uplift.append(x_name_i)
                df1[x_name_i] = x
                x_name_i = x_name_i + "_transformed"
                if random_select_association:
                    df1[x_name_i] = _fixed_transformation(f_list, x, random.randint(0, len(f_list) - 1))
                else:
                    df1[x_name_i] = _fixed_transformation(f_list, x, xi % len(f_list))
                x_name_uplift_transformed.append(x_name_i)
            x_name_uplift_transformed_dict[treatment_key_i] = x_name_uplift_transformed

    # mixed informative and uplift features
    for treatment_key_i in treatment_name:
        if treatment_key_i in n_mix_informative_uplift_dict and n_mix_informative_uplift_dict[treatment_key_i] > 0:
            for xi in range(n_mix_informative_uplift_dict[treatment_key_i]):
                x_name_i = f"x{len(x_name)+1}_mix"
                x_name.append(x_name_i)
                p_weight = np.random.uniform(0, 1)
                df1[x_name_i] = (
                    p_weight * df1[np.random.choice(x_informative_name)]
                    + (1 - p_weight) * df1[np.random.choice(x_name_uplift)]
                )

    # baseline conversion probability
    coef_classify = []
    for ci in range(n_classification_informative):
        rcoef = [0]
        while np.abs(rcoef) < 0.1:
            rcoef = 1.0 * (1 + 0.1 * np.random.randn(1)) * np.sqrt(1.0 / n_classification_informative)
        coef_classify.append(rcoef[0])
    x_classify = df1[x_informative_transformed].values
    p1 = positive_class_proportion
    a10 = logit(p1)
    err = np.random.normal(0, error_std, df1.shape[0])
    xb_array = (x_classify * coef_classify).sum(axis=1) + err
    a1 = fsolve(_softmax, a10, args=(p1, xb_array))[0]
    df1["conversion_prob_linear"] = a1 + xb_array
    df1["control_conversion_prob_linear"] = df1["conversion_prob_linear"].values

    # uplift conversion
    for treatment_key_i in treatment_name:
        if treatment_key_i in delta_uplift_dict and np.abs(delta_uplift_dict[treatment_key_i]) > 0.0:
            treatment_index = df1.index[df1["treatment_group_key"] == treatment_key_i].tolist()
            coef_uplift = []
            for ci in range(n_uplift_dict[treatment_key_i]):
                coef_uplift.append(0.5)
            x_uplift = df1.loc[:, x_name_uplift_transformed_dict[treatment_key_i]].values
            p2 = mean_dict[treatment_key_i]
            a20 = np.log(p2 / (1.0 - p2)) - a1
            xb_array = df1["conversion_prob_linear"].values + (x_uplift * coef_uplift).sum(axis=1)
            xb_array_treatment = xb_array[treatment_index]
            a2 = fsolve(_softmax, a20, args=(p2, xb_array_treatment))[0]
            df1[f"{treatment_key_i}_conversion_prob_linear"] = a2 + xb_array
            df1.loc[treatment_index, "conversion_prob_linear"] = df1.loc[treatment_index, f"{treatment_key_i}_conversion_prob_linear"].values
        else:
            df1[f"{treatment_key_i}_conversion_prob_linear"] = df1["conversion_prob_linear"].values

    # generate conversion probability and true treatment effect
    df1["conversion_prob"] = 1 / (1 + np.exp(-df1["conversion_prob_linear"].values))
    df1["control_conversion_prob"] = 1 / (1 + np.exp(-df1["control_conversion_prob_linear"].values))
    for treatment_key_i in treatment_name:
        df1[f"{treatment_key_i}_conversion_prob"] = 1 / (1 + np.exp(-df1[f"{treatment_key_i}_conversion_prob_linear"].values))
        df1[f"{treatment_key_i}_true_effect"] = df1[f"{treatment_key_i}_conversion_prob"].values - df1["control_conversion_prob"].values

    # generate Y
    df1["conversion_prob"] = np.clip(df1["conversion_prob"].values, 0, 1)
    df1[y_name] = np.random.binomial(1, df1["conversion_prob"].values)

    return df1, x_name 