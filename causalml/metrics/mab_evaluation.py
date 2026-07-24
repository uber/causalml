import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from scipy import stats
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MABMetrics:
    """Class for storing MAB algorithm evaluation metrics."""
    cumulative_reward: float
    average_reward: float
    regret: float
    arm_pulls: Dict[str, int]
    arm_rewards: Dict[str, float]
    arm_means: Dict[str, float]
    timestamp: str

def evaluate_mab(
    bandit,
    df: pd.DataFrame,
    reward_col: str = 'reward',
    arm_col: str = 'arm',
    true_means: Optional[Dict[str, float]] = None
) -> MABMetrics:
    """Evaluate a MAB algorithm on a dataset.
    
    Parameters
    ----------
    bandit : BaseBandit or BaseContextualBandit
        The bandit algorithm to evaluate.
    df : pd.DataFrame
        The dataset to evaluate on.
    reward_col : str, optional (default='reward')
        Name of the reward column.
    arm_col : str, optional (default='arm')
        Name of the arm column.
    true_means : dict, optional (default=None)
        Dictionary of true mean rewards for each arm.
        If provided, regret will be calculated.
        
    Returns
    -------
    metrics : MABMetrics
        Evaluation metrics including cumulative reward, average reward,
        regret (if true_means provided), and arm statistics.
    """
    # Initialize metrics
    cumulative_reward = 0
    arm_pulls = {arm: 0 for arm in df[arm_col].unique()}
    arm_rewards = {arm: 0.0 for arm in df[arm_col].unique()}
    
    # Run bandit algorithm
    for _, row in df.iterrows():
        # Select arm
        if hasattr(bandit, 'select_arm'):
            arm = bandit.select_arm()
        else:
            arm = bandit.select_arm(row)
            
        # Get reward
        reward = row[reward_col]
        
        # Update bandit
        if hasattr(bandit, 'update'):
            bandit.update(arm, reward)
        else:
            bandit.update(row, reward)
            
        # Update metrics
        cumulative_reward += reward
        arm_pulls[arm] += 1
        arm_rewards[arm] += reward
    
    # Calculate final metrics
    n_samples = len(df)
    average_reward = cumulative_reward / n_samples
    
    # Calculate arm means
    arm_means = {
        arm: arm_rewards[arm] / arm_pulls[arm] if arm_pulls[arm] > 0 else 0
        for arm in arm_pulls
    }
    
    # Calculate regret if true means provided
    regret = 0
    if true_means is not None:
        best_arm = max(true_means.items(), key=lambda x: x[1])[0]
        best_mean = true_means[best_arm]
        regret = best_mean * n_samples - cumulative_reward
    
    return MABMetrics(
        cumulative_reward=cumulative_reward,
        average_reward=average_reward,
        regret=regret,
        arm_pulls=arm_pulls,
        arm_rewards=arm_rewards,
        arm_means=arm_means,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

def compare_mab_algorithms(
    algorithms: Dict[str, Union['BaseBandit', 'BaseContextualBandit']],
    df: pd.DataFrame,
    reward_col: str = 'reward',
    arm_col: str = 'arm',
    true_means: Optional[Dict[str, float]] = None,
    n_runs: int = 1,
    plot: bool = True
) -> Dict[str, MABMetrics]:
    """Compare multiple MAB algorithms on the same dataset.
    
    Parameters
    ----------
    algorithms : dict
        Dictionary of algorithm names to bandit instances.
    df : pd.DataFrame
        The dataset to evaluate on.
    reward_col : str, optional (default='reward')
        Name of the reward column.
    arm_col : str, optional (default='arm')
        Name of the arm column.
    true_means : dict, optional (default=None)
        Dictionary of true mean rewards for each arm.
    n_runs : int, optional (default=1)
        Number of times to run each algorithm.
    plot : bool, optional (default=True)
        Whether to plot the comparison results.
        
    Returns
    -------
    results : dict
        Dictionary of algorithm names to MABMetrics.
    """
    results = {}
    
    # Run each algorithm
    for name, algorithm in algorithms.items():
        run_metrics = []
        for _ in range(n_runs):
            # Create a copy of the algorithm for this run
            alg_copy = algorithm.__class__(**algorithm.__dict__)
            metrics = evaluate_mab(
                alg_copy,
                df,
                reward_col=reward_col,
                arm_col=arm_col,
                true_means=true_means
            )
            run_metrics.append(metrics)
        
        # Average metrics across runs
        avg_metrics = MABMetrics(
            cumulative_reward=np.mean([m.cumulative_reward for m in run_metrics]),
            average_reward=np.mean([m.average_reward for m in run_metrics]),
            regret=np.mean([m.regret for m in run_metrics]),
            arm_pulls=run_metrics[0].arm_pulls,  # Same for all runs
            arm_rewards={arm: np.mean([m.arm_rewards[arm] for m in run_metrics])
                        for arm in run_metrics[0].arm_rewards},
            arm_means={arm: np.mean([m.arm_means[arm] for m in run_metrics])
                      for arm in run_metrics[0].arm_means},
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        results[name] = avg_metrics
    
    if plot:
        plot_mab_comparison(results, true_means)
    
    return results

def plot_mab_comparison(
    results: Dict[str, MABMetrics],
    true_means: Optional[Dict[str, float]] = None
):
    """Plot comparison of MAB algorithms.
    
    Parameters
    ----------
    results : dict
        Dictionary of algorithm names to MABMetrics.
    true_means : dict, optional (default=None)
        Dictionary of true mean rewards for each arm.
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MAB Algorithm Comparison', fontsize=16)
    
    # Plot 1: Average Reward
    rewards = [metrics.average_reward for metrics in results.values()]
    sns.barplot(x=list(results.keys()), y=rewards, ax=axes[0, 0])
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Regret
    regrets = [metrics.regret for metrics in results.values()]
    sns.barplot(x=list(results.keys()), y=regrets, ax=axes[0, 1])
    axes[0, 1].set_title('Cumulative Regret')
    axes[0, 1].set_ylabel('Regret')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Arm Pulls
    arm_pulls = pd.DataFrame({
        name: metrics.arm_pulls
        for name, metrics in results.items()
    }).T
    arm_pulls.plot(kind='bar', stacked=True, ax=axes[1, 0])
    axes[1, 0].set_title('Arm Pulls Distribution')
    axes[1, 0].set_ylabel('Number of Pulls')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Arm Means vs True Means
    if true_means is not None:
        arm_means = pd.DataFrame({
            name: metrics.arm_means
            for name, metrics in results.items()
        }).T
        true_means_df = pd.DataFrame([true_means] * len(results), index=results.keys())
        
        # Plot estimated means
        arm_means.plot(kind='bar', ax=axes[1, 1], alpha=0.7)
        # Plot true means
        for arm in true_means:
            axes[1, 1].axhline(
                y=true_means[arm],
                color='k',
                linestyle='--',
                alpha=0.3,
                label=f'True {arm}'
            )
        axes[1, 1].set_title('Arm Mean Estimates vs True Means')
        axes[1, 1].set_ylabel('Mean Reward')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curve(
    bandit,
    df: pd.DataFrame,
    reward_col: str = 'reward',
    arm_col: str = 'arm',
    window_size: int = 100,
    plot: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Plot the learning curve of a MAB algorithm.
    
    Parameters
    ----------
    bandit : BaseBandit or BaseContextualBandit
        The bandit algorithm to evaluate.
    df : pd.DataFrame
        The dataset to evaluate on.
    reward_col : str, optional (default='reward')
        Name of the reward column.
    arm_col : str, optional (default='arm')
        Name of the arm column.
    window_size : int, optional (default=100)
        Size of the moving average window.
    plot : bool, optional (default=True)
        Whether to plot the learning curve.
        
    Returns
    -------
    rewards : np.ndarray
        Array of rewards over time.
    avg_rewards : np.ndarray
        Array of moving average rewards.
    """
    # Initialize arrays
    n_samples = len(df)
    rewards = np.zeros(n_samples)
    
    # Run bandit algorithm
    for i, row in df.iterrows():
        # Select arm
        if hasattr(bandit, 'select_arm'):
            arm = bandit.select_arm()
        else:
            arm = bandit.select_arm(row)
            
        # Get reward
        reward = row[reward_col]
        
        # Update bandit
        if hasattr(bandit, 'update'):
            bandit.update(arm, reward)
        else:
            bandit.update(row, reward)
            
        # Store reward
        rewards[i] = reward
    
    # Calculate moving average
    avg_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, alpha=0.3, label='Rewards')
        plt.plot(range(window_size-1, n_samples), avg_rewards, label=f'{window_size}-step Moving Average')
        plt.title('Learning Curve')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return rewards, avg_rewards

def cumulative_reward(rewards: np.ndarray) -> np.ndarray:
    """Calculate cumulative reward over time.
    
    Parameters
    ----------
    rewards : np.ndarray
        Array of rewards over time.
        
    Returns
    -------
    cum_reward : np.ndarray
        Array of cumulative rewards.
    """
    return np.cumsum(rewards)

def cumulative_regret(rewards: np.ndarray, optimal_reward: float) -> np.ndarray:
    """Calculate cumulative regret over time.
    
    Parameters
    ----------
    rewards : np.ndarray
        Array of rewards over time.
    optimal_reward : float
        The optimal reward that could have been achieved.
        
    Returns
    -------
    cum_regret : np.ndarray
        Array of cumulative regrets.
    """
    return np.cumsum(optimal_reward - rewards)

def plot_cumulative_reward(results: dict, filename: str = None):
    """Plot cumulative reward for each algorithm."""
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        plt.plot(cumulative_reward(result['rewards']), label=name)
    plt.title('Cumulative Reward Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_cumulative_regret(results: dict, optimal_reward: float, filename: str = None):
    """Plot cumulative regret for each algorithm."""
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        plt.plot(cumulative_regret(result['rewards'], optimal_reward), label=name)
    plt.title('Cumulative Regret Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_arm_selection_frequency(results: dict, n_arms: int, filename: str = None):
    """Plot arm selection frequency for each algorithm."""
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        arm_freq = pd.Series(result['selected_arms']).value_counts(normalize=True)
        plt.bar(np.arange(len(arm_freq)) + 0.1 * list(results.keys()).index(name),
                arm_freq.values,
                width=0.1,
                label=name)
    plt.title('Arm Selection Frequency')
    plt.xlabel('Arm')
    plt.ylabel('Selection Frequency')
    plt.xticks(np.arange(n_arms) + 0.5, [f'Arm {i}' for i in range(n_arms)])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close() 