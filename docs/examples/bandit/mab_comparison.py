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
    n_samples = 1000
    n_arms = 3
    n_features = 2
    
    # Generate data with features for contextual MAB
    df = make_mab_data(
        n_samples=n_samples,
        n_arms=n_arms,
        n_features=n_features,
        arm_effects=[0.1, 0.5, 0.9],  # Different true means for each arm
        random_seed=42
    )
    
    # Initialize algorithms
    print("Initializing algorithms...")
    algorithms = {
        # Classical MAB
        'EpsilonGreedy': EpsilonGreedy(df, reward='reward', arm='arm', epsilon=0.1),
        'UCB': UCB(df, reward='reward', arm='arm', alpha=1.0),
        'ThompsonSampling': ThompsonSampling(df, reward='reward', arm='arm'),
        
        # Batch Classical MAB
        'BatchEpsilonGreedy': BatchBandit(
            EpsilonGreedy(df, reward='reward', arm='arm', epsilon=0.1),
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
            features=['feature_0', 'feature_1'],
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
            features=['feature_0', 'feature_1'],
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
                    # For LinUCB, we need to pass features
                    contexts = batch_df[algo.features].values
                    arms = algo.batch_select(contexts)
                    rewards.extend(batch_df['reward'].values)
                elif 'Cohort' in name:
                    # For Cohort Thompson Sampling, we need to pass cohort
                    contexts = batch_df[algo.feature].values
                    arms = algo.batch_select(contexts)
                    rewards.extend(batch_df['reward'].values)
                else:
                    # For classical MAB
                    arms = algo.select_batch()
                    rewards.extend(batch_df['reward'].values)
                
                selected_arms.extend(arms)
                
                # Update model
                if 'LinUCB' in name:
                    algo.batch_update(arms, contexts, batch_df['reward'].values)
                elif 'Cohort' in name:
                    algo.batch_update(arms, contexts, batch_df['reward'].values)
                else:
                    algo.update_batch(arms, batch_df['reward'].values)
        else:
            # For non-batch algorithms
            for _, row in df.iterrows():
                if 'LinUCB' in name:
                    context = row[algo.features].values
                    arm = algo.select_arm(context)
                    algo.update(arm, context, row['reward'])
                elif 'Cohort' in name:
                    context = row[algo.feature]
                    arm = algo.select_arm(context)
                    algo.update(arm, context, row['reward'])
                else:
                    arm = algo.select_arm()
                    algo.update(arm, row['reward'])
                
                rewards.append(row['reward'])
                selected_arms.append(arm)
        
        # Store results
        results[name] = {
            'rewards': np.array(rewards),
            'selected_arms': np.array(selected_arms)
        }
    
    # Evaluate and visualize results
    print("Evaluating results...")
    
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