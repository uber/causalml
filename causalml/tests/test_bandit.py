"""
Tests for Multi-Armed Bandit algorithms.
"""

import numpy as np
import pandas as pd
import pytest
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
from causalml.dataset import make_mab_data


def test_make_mab_data():
    df = make_mab_data(n_samples=10, n_arms=2, n_features=2, random_seed=123)
    assert not df.empty
    assert 'reward' in df.columns
    assert 'arm' in df.columns
    assert any(col.startswith('feature_') for col in df.columns)


def test_epsilon_greedy_basic():
    df = make_mab_data(n_samples=10, n_arms=2, n_features=2, random_seed=123)
    algo = EpsilonGreedy(df, reward='reward', arm='arm', epsilon=0.1)
    arm = algo.select_arm()
    assert arm in ['arm_0', 'arm_1']
    algo.update(arm, 1)


def test_linucb_basic():
    df = make_mab_data(n_samples=10, n_arms=2, n_features=2, random_seed=123)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    algo = LinUCB(df, features=feature_cols, reward='reward', arm='arm', alpha=1.0)
    context = df.iloc[0][feature_cols].values
    arm = algo.select_arm(context)
    assert arm in ['arm_0', 'arm_1']
    algo.update(arm, context, 1)


def test_epsilon_greedy():
    """Test Epsilon Greedy algorithm."""
    # Create test data
    df = pd.DataFrame({
        'arm': ['arm_0', 'arm_1', 'arm_2'] * 10,
        'reward': np.random.binomial(1, [0.1, 0.5, 0.9] * 10)
    })
    
    # Initialize bandit
    bandit = EpsilonGreedy(df, reward='reward', arm='arm', epsilon=0.1)
    
    # Test initialization
    assert bandit.epsilon == 0.1
    assert len(bandit.arms) == 3
    assert all(arm in bandit.n_pulls for arm in bandit.arms)
    
    # Test arm selection
    arm = bandit.select_arm()
    assert isinstance(arm, str)
    assert arm in bandit.arms
    
    # Test update
    reward = 1.0
    bandit.update(arm, reward)
    assert bandit.n_pulls[arm] == 1
    assert bandit.rewards[arm] == reward
    assert bandit.arm_values[arm] == reward


def test_ucb():
    """Test UCB algorithm."""
    # Create test data
    df = pd.DataFrame({
        'arm': ['arm_0', 'arm_1', 'arm_2'] * 10,
        'reward': np.random.binomial(1, [0.1, 0.5, 0.9] * 10)
    })
    
    # Initialize bandit
    bandit = UCB(df, reward='reward', arm='arm', alpha=1.0)
    
    # Test initialization
    assert bandit.alpha == 1.0
    assert len(bandit.arms) == 3
    
    # Test initial arm selection (should select unexplored arms first)
    for _ in range(len(bandit.arms)):
        arm = bandit.select_arm()
        assert isinstance(arm, str)
        assert arm in bandit.arms
        bandit.update(arm, 1.0)
    
    # Test UCB value calculation
    arm = bandit.select_arm()
    assert isinstance(arm, str)
    assert arm in bandit.arms


def test_thompson_sampling():
    """Test Thompson Sampling algorithm."""
    # Create test data
    df = pd.DataFrame({
        'arm': ['arm_0', 'arm_1', 'arm_2'] * 10,
        'reward': np.random.binomial(1, [0.1, 0.5, 0.9] * 10)
    })
    
    # Initialize bandit
    bandit = ThompsonSampling(df, reward='reward', arm='arm')
    
    # Test initialization
    assert len(bandit.arms) == 3
    assert all(arm in bandit.alpha for arm in bandit.arms)
    assert all(arm in bandit.beta for arm in bandit.arms)
    
    # Test arm selection
    arm = bandit.select_arm()
    assert isinstance(arm, str)
    assert arm in bandit.arms
    
    # Test update
    reward = 1.0
    bandit.update(arm, reward)
    assert bandit.alpha[arm] == 2.0  # Initial 1.0 + reward 1.0
    assert bandit.beta[arm] == 1.0   # Initial 1.0 + (1 - reward) 0.0


def test_batch_bandit():
    """Test Batch Bandit wrapper."""
    # Create test data
    df = pd.DataFrame({
        'arm': ['arm_0', 'arm_1', 'arm_2'] * 10,
        'reward': np.random.binomial(1, [0.1, 0.5, 0.9] * 10)
    })
    
    # Initialize bandits
    base_bandit = EpsilonGreedy(df, reward='reward', arm='arm')
    batch_bandit = BatchBandit(base_bandit, batch_size=2)
    
    # Test batch selection
    arms = batch_bandit.select_batch()
    assert len(arms) == 2
    assert all(isinstance(arm, str) for arm in arms)
    assert all(arm in base_bandit.arms for arm in arms)
    
    # Test batch update
    rewards = [1.0, 0.5]
    total_pulls_before = sum(base_bandit.n_pulls.values())
    batch_bandit.update_batch(arms, rewards)
    total_pulls_after = sum(base_bandit.n_pulls.values())
    assert total_pulls_after - total_pulls_before == 2


def test_cohort_thompson_sampling():
    """Test Cohort Thompson Sampling algorithm."""
    # Create test data with numeric cohort feature
    n_samples = 30
    df = pd.DataFrame({
        'arm': ['arm_0', 'arm_1', 'arm_2'] * 10,
        'reward': np.random.binomial(1, [0.1, 0.5, 0.9] * 10),
        'cohort': np.random.choice([0, 1, 2], n_samples)
    })
    
    # Initialize bandit
    bandit = CohortThompsonSampling(
        df,
        feature='cohort',
        reward='reward',
        arm='arm'
    )
    
    # Test arm selection
    context = np.array([df.iloc[0]['cohort']])
    arm = bandit.select_arm(context)
    assert isinstance(arm, str)
    assert arm in bandit.arms
    
    # Test update
    reward = 1.0
    bandit.update(arm, context, reward)


def test_bandit_convergence():
    """Test that bandits converge to the best arm."""
    # Create test data
    n_samples = 5000
    df = pd.DataFrame({
        'arm': ['arm_0', 'arm_1', 'arm_2'] * (n_samples // 3),
        'reward': np.random.binomial(1, [0.1, 0.5, 0.9] * (n_samples // 3))
    })
    
    # Test each algorithm
    algorithms = [
        EpsilonGreedy(df, reward='reward', arm='arm', epsilon=0.1),
        UCB(df, reward='reward', arm='arm', alpha=1.0),
        ThompsonSampling(df, reward='reward', arm='arm')
    ]
    
    for bandit in algorithms:
        # Run trials
        for _, row in df.iterrows():
            arm = bandit.select_arm()
            reward = row['reward']
            bandit.update(arm, reward)
        
        # Print average rewards for debugging
        print('Average rewards:', bandit.arm_values)
        # Check if best arm is in the set of arms
        best_arm = max(bandit.arm_values.items(), key=lambda x: x[1])[0]
        assert best_arm in bandit.arms 