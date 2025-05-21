"""
Example: Multi-Armed Bandit Algorithm Comparison
==============================================

This example demonstrates how to use different Multi-Armed Bandit (MAB) algorithms
from the causalml library, including both classical and contextual MAB algorithms,
as well as their batch versions.

The example covers:
1. Data generation for MAB experiments
2. Running different MAB algorithms:
   - Classical MAB: EpsilonGreedy, UCB, ThompsonSampling
   - Batch Classical MAB: Batch versions of the above
   - Contextual MAB: LinUCB, CohortThompsonSampling
   - Batch Contextual MAB: Batch versions of the above
3. Evaluating and visualizing results:
   - Cumulative reward over time
   - Cumulative regret over time
   - Arm selection frequency

Usage
-----
Run this example with::

    python docs/examples/bandit/mab_comparison.py

The script will generate three visualization files:
- cumulative_reward.png: Shows how total reward accumulates over time
- cumulative_regret.png: Shows how regret accumulates over time
- arm_selection_frequency.png: Shows the final distribution of arm selections

Expected Results
--------------
The example uses a synthetic dataset with 3 arms having different true means (0.1, 0.5, 0.9).
You should observe:
1. All algorithms eventually identify arm_2 as the best arm (mean = 0.9)
2. Contextual MAB algorithms may converge faster due to using feature information
3. Batch algorithms show more stable but potentially slower convergence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from causalml.dataset import make_mab_data
from causalml.optimize import (
    EpsilonGreedy,
    UCB,
    ThompsonSampling,
    BatchBandit,
    LinUCB,
    BatchLinUCB,
    CohortThompsonSampling,
    BatchCohortThompsonSampling
)
from causalml.metrics import (
    cumulative_reward,
    cumulative_regret,
    plot_cumulative_reward,
    plot_cumulative_regret,
    plot_arm_selection_frequency
)


def run_mab_experiment():
    """Run a comprehensive MAB experiment comparing different algorithms.
    
    This function:
    1. Generates synthetic data for MAB experiments
    2. Initializes and runs different MAB algorithms
    3. Collects and stores results
    4. Generates visualizations of the results
    
    The experiment uses a dataset with:
    - 1000 samples
    - 3 arms with different true means (0.1, 0.5, 0.9)
    - 2 features for contextual MAB algorithms
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data
    print("Generating data...")
    n_samples = 300
    n_arms = 5
    n_features = 2
    arm_effects = {f"arm_{i}": eff for i, eff in enumerate([0.1, 0.2, 0.3, 0.5, 0.9])}
    df = make_mab_data(
        n_samples=n_samples,
        n_arms=n_arms,
        n_features=n_features,
        arm_effects=arm_effects,
        random_seed=42,
        error_std=0.2
    )
    
    # Dynamically select the first two feature columns for contextual MAB
    feature_cols = [col for col in df.columns if col.startswith("feature_")][:2]

    # Add cohort column
    df["cohort"] = np.random.choice([0, 1, 2], size=len(df))

    # --- Make reward contextual: reward_prob depends on features and arm ---
    feature_weights = np.random.uniform(-1, 1, len(feature_cols))
    df['feature_score'] = df[feature_cols].dot(feature_weights)
    df['reward_prob'] = df['reward_prob'] + 0.5 * df['feature_score']
    df['reward_prob'] = np.clip(df['reward_prob'], 0, 1)
    df['reward'] = np.random.binomial(1, df['reward_prob'])

    # Initialize algorithms
    print("Initializing algorithms...")
    algorithms = {
        # Classical MAB
        'EpsilonGreedy': EpsilonGreedy(df, reward='reward', arm='arm', epsilon=0.3),
        'UCB': UCB(df, reward='reward', arm='arm', alpha=1.0),
        'ThompsonSampling': ThompsonSampling(df, reward='reward', arm='arm'),
        
        # Batch Classical MAB
        'BatchEpsilonGreedy': BatchBandit(
            EpsilonGreedy(df, reward='reward', arm='arm', epsilon=0.3),
            batch_size=10
        ),
        'BatchUCB': BatchBandit(
            UCB(df, reward='reward', arm='arm', alpha=1.0),
            batch_size=10
        ),
        'BatchThompsonSampling': BatchBandit(
            ThompsonSampling(df, reward='reward', arm='arm'),
            batch_size=10
        ),
        
        # Contextual MAB
        'LinUCB': LinUCB(
            df,
            features=feature_cols,
            reward='reward',
            arm='arm',
            alpha=1.0
        ),
        'CohortThompsonSampling': CohortThompsonSampling(
            df,
            feature='cohort',
            reward='reward',
            arm='arm'
        ),
        
        # Batch Contextual MAB
        'BatchLinUCB': BatchLinUCB(
            df,
            features=feature_cols,
            reward='reward',
            arm='arm',
            alpha=1.0,
            batch_size=10
        ),
        'BatchCohortThompsonSampling': BatchCohortThompsonSampling(
            df,
            feature='cohort',
            reward='reward',
            arm='arm',
            batch_size=10
        )
    }
    
    # Run experiments
    print("Running experiments...")
    results = {}
    for name, algo in algorithms.items():
        print(f"Running {name}...")
        
        # Initialize metrics
        rewards = []
        selected_arms = []
        
        # Run algorithm
        if 'Batch' in name:
            # For batch algorithms
            for i in range(0, len(df), algo.batch_size):
                batch_df = df.iloc[i:i + algo.batch_size]
                if len(batch_df) == 0:
                    break
                if 'LinUCB' in name:
                    contexts = batch_df[algo.features].values
                    arms = algo.batch_select(contexts)
                    # Sample rewards for chosen arms
                    rewards_batch = []
                    for arm, context in zip(arms, contexts):
                        possible_rows = batch_df[batch_df['arm'] == arm]
                        if not possible_rows.empty:
                            row = possible_rows.sample(1).iloc[0]
                            rewards_batch.append(row['reward'])
                        else:
                            rewards_batch.append(0.0)
                    rewards.extend(rewards_batch)
                    algo.batch_update(arms, contexts, rewards_batch)
                elif 'Cohort' in name:
                    contexts = batch_df[algo.feature].values.reshape(-1, 1)
                    arms = algo.batch_select(contexts)
                    rewards_batch = []
                    for arm, context in zip(arms, contexts):
                        possible_rows = batch_df[(batch_df['arm'] == arm) & (batch_df[algo.feature] == context[0])]
                        if not possible_rows.empty:
                            row = possible_rows.sample(1).iloc[0]
                            rewards_batch.append(row['reward'])
                        else:
                            rewards_batch.append(0.0)
                    rewards.extend(rewards_batch)
                    algo.batch_update(arms, contexts, rewards_batch)
                else:
                    arms = algo.select_batch()
                    rewards_batch = []
                    for arm in arms:
                        possible_rows = batch_df[batch_df['arm'] == arm]
                        if not possible_rows.empty:
                            row = possible_rows.sample(1).iloc[0]
                            rewards_batch.append(row['reward'])
                        else:
                            rewards_batch.append(0.0)
                    rewards.extend(rewards_batch)
                    algo.update_batch(arms, rewards_batch)
                selected_arms.extend(arms)
        else:
            # For non-batch algorithms
            for _ in range(len(df)):
                if 'LinUCB' in name:
                    # Randomly sample a context row
                    row = df.sample(1).iloc[0]
                    context = row[algo.features].values
                    arm = algo.select_arm(context)
                    # Sample a reward for the chosen arm and context
                    possible_rows = df[(df['arm'] == arm)]
                    if not possible_rows.empty:
                        reward_row = possible_rows.sample(1).iloc[0]
                        reward = reward_row['reward']
                    else:
                        reward = 0.0
                    algo.update(arm, context, reward)
                elif 'Cohort' in name:
                    row = df.sample(1).iloc[0]
                    context = np.array([row[algo.feature]])
                    arm = algo.select_arm(context)
                    possible_rows = df[(df['arm'] == arm) & (df[algo.feature] == context[0])]
                    if not possible_rows.empty:
                        reward_row = possible_rows.sample(1).iloc[0]
                        reward = reward_row['reward']
                    else:
                        reward = 0.0
                    algo.update(arm, context, reward)
                else:
                    arm = algo.select_arm()
                    possible_rows = df[df['arm'] == arm]
                    if not possible_rows.empty:
                        reward_row = possible_rows.sample(1).iloc[0]
                        reward = reward_row['reward']
                    else:
                        reward = 0.0
                    algo.update(arm, reward)
                rewards.append(reward)
                selected_arms.append(arm)
        
        # Store results
        results[name] = {
            'rewards': np.array(rewards),
            'selected_arms': np.array(selected_arms)
        }
    
    # Evaluate and visualize results
    print("Evaluating results...")

    # Print summary table of results
    print("\nSummary Table:")
    print(f"{'Algorithm':<30} {'Avg Reward':>12} {'Cum Reward':>12} {'Regret':>12}")
    for name, result in results.items():
        avg_reward = np.mean(result['rewards'])
        cum_reward = np.sum(result['rewards'])
        regret = (0.9 * n_samples) - cum_reward  # 0.9 is the best arm mean
        print(f"{name:<30} {avg_reward:12.4f} {cum_reward:12.1f} {regret:12.1f}")

    # Plot cumulative rewards
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        cum_reward = cumulative_reward(result['rewards'])
        plt.plot(cum_reward, label=name)
    plt.title('Cumulative Reward Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Reward')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('cumulative_reward.png')
    plt.close()
    
    # Plot cumulative regret
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        cum_regret = cumulative_regret(result['rewards'], optimal_reward=0.9)  # Best arm has mean 0.9
        plt.plot(cum_regret, label=name)
    plt.title('Cumulative Regret Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('cumulative_regret.png')
    plt.close()
    
    # Plot arm selection frequency
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
    plt.savefig('arm_selection_frequency.png')
    plt.close()
    
    print("Results have been saved as PNG files.")


if __name__ == '__main__':
    run_mab_experiment() 