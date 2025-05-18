import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from scipy.special import expit
from scipy.interpolate import UnivariateSpline

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

def make_mab_data(
    n_samples: int = 10000,
    n_arms: int = 4,
    n_features: int = 6,
    n_informative: int = 4,
    n_redundant: int = 0,
    n_repeated: int = 0,
    arm_effects: Dict[str, float] = None,
    positive_class_proportion: float = 0.1,
    random_seed: int = 20200101,
    feature_association_list: List[str] = ["linear", "quadratic", "cubic", "relu", "sin", "cos"],
    random_select_association: bool = True,
    error_std: float = 0.05
) -> pd.DataFrame:
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
    n_features : int, optional (default=6)
        Total number of features.
    n_informative : int, optional (default=4)
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
        
    Returns
    -------
    df : pd.DataFrame
        Generated dataset with the following columns:
        - arm: The arm/treatment assigned to each sample
        - reward: The binary reward (0 or 1) for each sample
        - reward_prob: The probability of reward for each sample
        - feature_*: Generated features (informative, redundant, repeated, and irrelevant)
        - feature_*_transformed: Transformed versions of informative features
    """
    np.random.seed(seed=random_seed)
    
    # Create arm names
    arm_names = [f"arm_{i}" for i in range(n_arms)]
    
    # Set default arm effects if not provided
    if arm_effects is None:
        arm_effects = {arm: np.random.uniform(-0.1, 0.1) for arm in arm_names}
    
    # Create data frame
    df = pd.DataFrame()
    
    # Generate treatment assignments
    treatment_list = np.random.choice(arm_names, size=n_samples)
    df["arm"] = treatment_list
    
    # Define feature association functions
    feature_association_pattern_dict = {
        "linear": _f_linear,
        "quadratic": _f_quadratic,
        "cubic": _f_cubic,
        "relu": _f_relu,
        "sin": _f_sin,
        "cos": _f_cos
    }
    f_list = [feature_association_pattern_dict[fi] for fi in feature_association_list]
    
    # Generate features
    feature_names = []
    informative_features = []
    
    # Generate informative features
    for i in range(n_informative):
        x = np.random.normal(0, 1, n_samples)
        feature_name = f"feature_{i+1}_informative"
        feature_names.append(feature_name)
        informative_features.append(feature_name)
        df[feature_name] = x
        
        # Transform feature
        transformed_name = f"{feature_name}_transformed"
        if random_select_association:
            df[transformed_name] = _random_transformation(f_list, x)
        else:
            df[transformed_name] = _fixed_transformation(f_list, x, i % len(f_list))
    
    # Generate redundant features
    for i in range(n_redundant):
        source_idx = np.random.choice(len(informative_features))
        source_feature = informative_features[source_idx]
        feature_name = f"feature_{len(feature_names)+1}_redundant"
        feature_names.append(feature_name)
        df[feature_name] = df[source_feature] + np.random.normal(0, 0.1, n_samples)
    
    # Generate repeated features
    for i in range(n_repeated):
        source_idx = np.random.choice(len(informative_features))
        source_feature = informative_features[source_idx]
        feature_name = f"feature_{len(feature_names)+1}_repeated"
        feature_names.append(feature_name)
        df[feature_name] = df[source_feature]
    
    # Generate irrelevant features
    n_irrelevant = n_features - n_informative - n_redundant - n_repeated
    for i in range(n_irrelevant):
        feature_name = f"feature_{len(feature_names)+1}_irrelevant"
        feature_names.append(feature_name)
        df[feature_name] = np.random.normal(0, 1, n_samples)
    
    # Generate rewards
    base_prob = positive_class_proportion
    df["reward_prob"] = base_prob
    
    # Add arm effects
    for arm in arm_names:
        arm_idx = df["arm"] == arm
        df.loc[arm_idx, "reward_prob"] += arm_effects[arm]
    
    # Add feature effects
    feature_coefs = np.random.normal(0, 1, n_informative)
    for i, feature in enumerate(informative_features):
        df["reward_prob"] += feature_coefs[i] * df[f"{feature}_transformed"]
    
    # Add noise
    df["reward_prob"] += np.random.normal(0, error_std, n_samples)
    
    # Clip probabilities to [0, 1]
    df["reward_prob"] = np.clip(df["reward_prob"], 0, 1)
    
    # Generate binary rewards
    df["reward"] = np.random.binomial(1, df["reward_prob"])
    
    return df

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