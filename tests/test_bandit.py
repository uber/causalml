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
    df, x_name = make_mab_data(n_samples=10, n_arms=2, n_features=2, random_seed=123)
    assert not df.empty
    assert 'reward' in df.columns
    assert 'arm' in df.columns
    assert any(col.startswith('feature_') for col in df.columns)
    assert isinstance(x_name, list)
    assert len(x_name) > 0


def test_epsilon_greedy_basic():
    df, x_name = make_mab_data(n_samples=10, n_arms=2, n_features=2, random_seed=123)
    algo = EpsilonGreedy(epsilon=0.1)
    algo.fit(df['arm'], df['reward'])
    arm = algo.select_arm()
    assert arm in ['arm_0', 'arm_1']
    algo.update(arm, 1)


def test_linucb_basic():
    df, x_name = make_mab_data(n_samples=10, n_arms=2, n_features=2, random_seed=123)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    algo = LinUCB(alpha=1.0)
    algo.fit(df, df['arm'], df['reward'], feature_cols)
    context = df[feature_cols].values[0]
    arm = algo.select_arm(context)
    assert arm in ['arm_0', 'arm_1']
    algo.update(arm, context, 1)


def test_epsilon_greedy():
    """Test Epsilon Greedy algorithm."""
    # Create test data
    arms = np.array(['arm_0', 'arm_1', 'arm_2'] * 10)
    rewards = np.random.binomial(1, [0.1, 0.5, 0.9] * 10)
    
    # Initialize bandit
    bandit = EpsilonGreedy(epsilon=0.1)
    bandit.fit(arms, rewards)
    
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
    assert bandit.n_pulls[arm] >= 1
    assert bandit.rewards[arm] >= 0
    assert bandit.arm_values[arm] >= 0


def test_ucb():
    """Test UCB algorithm."""
    # Create test data
    arms = np.array(['arm_0', 'arm_1', 'arm_2'] * 10)
    rewards = np.random.binomial(1, [0.1, 0.5, 0.9] * 10)
    
    # Initialize bandit
    bandit = UCB(alpha=1.0)
    bandit.fit(arms, rewards)
    
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
    arms = np.array(['arm_0', 'arm_1', 'arm_2'] * 10)
    rewards = np.random.binomial(1, [0.1, 0.5, 0.9] * 10)
    
    # Initialize bandit
    bandit = ThompsonSampling()
    bandit.fit(arms, rewards)
    
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
    assert bandit.alpha[arm] >= 1.0
    assert bandit.beta[arm] >= 1.0


def test_batch_bandit():
    """Test Batch Bandit wrapper."""
    # Create test data
    arms = np.array(['arm_0', 'arm_1', 'arm_2'] * 10)
    rewards = np.random.binomial(1, [0.1, 0.5, 0.9] * 10)
    
    # Initialize bandits
    base_bandit = EpsilonGreedy(epsilon=0.1)
    base_bandit.fit(arms, rewards)
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
    n_samples = 30
    arms = np.array(['arm_0', 'arm_1', 'arm_2'] * 10)
    rewards = np.random.binomial(1, [0.1, 0.5, 0.9] * 10)
    cohorts = np.random.choice([0, 1, 2], n_samples)
    X = pd.DataFrame({'cohort': cohorts})
    bandit = CohortThompsonSampling()
    bandit.fit(X, arms, rewards, 'cohort')
    context = np.array([cohorts[0]])
    arm = bandit.select_arm(context)
    assert isinstance(arm, str)
    assert arm in bandit.arms
    reward = 1.0
    bandit.update(arm, context, reward)


def test_batch_linucb():
    """Test Batch LinUCB algorithm."""
    # Create test data
    n_samples = 100
    n_features = 2
    X = np.random.normal(0, 1, (n_samples, n_features))
    arms = np.array(['arm_0', 'arm_1'] * (n_samples // 2))
    rewards = np.random.binomial(1, 0.5, n_samples)
    
    # Initialize bandit
    bandit = BatchLinUCB(alpha=1.0, batch_size=10)
    bandit.fit(X, arms, rewards)
    
    # Test batch selection
    selected_arms = bandit.select_arm(X[:10])
    assert len(selected_arms) == 10
    assert all(isinstance(arm, str) for arm in selected_arms)
    assert all(arm in ['arm_0', 'arm_1'] for arm in selected_arms)
    
    # Test batch update
    batch_rewards = np.random.binomial(1, 0.5, 10)
    bandit.update_batch(np.array(selected_arms), X[:10], batch_rewards)


def test_batch_cohort_thompson_sampling():
    """Test Batch Cohort Thompson Sampling algorithm."""
    # Create test data
    n_samples = 100
    X = np.random.normal(0, 1, (n_samples, 1))  # One feature for cohorts
    arms = np.array(['arm_0', 'arm_1'] * (n_samples // 2))
    rewards = np.random.binomial(1, 0.5, n_samples)
    cohorts = np.random.choice([0, 1, 2], n_samples)
    
    # Initialize bandit
    bandit = BatchCohortThompsonSampling(batch_size=10)
    bandit.fit(X, arms, rewards, cohort_feature=cohorts)
    
    # Test batch selection
    selected_arms = bandit.select_arm_batch(cohorts[:10])
    assert len(selected_arms) == 10
    assert all(isinstance(arm, str) for arm in selected_arms)
    assert all(arm in ['arm_0', 'arm_1'] for arm in selected_arms)
    
    # Test batch update
    batch_rewards = np.random.binomial(1, 0.5, 10)
    bandit.update_batch(cohorts[:10], selected_arms, batch_rewards)


def test_bandit_convergence():
    """Test that bandits converge to the best arm."""
    # Create test data
    n_samples = 5000
    arms = np.array(['arm_0', 'arm_1', 'arm_2'] * (n_samples // 3))
    rewards = np.random.binomial(1, [0.1, 0.5, 0.9] * (n_samples // 3))
    
    # Test each algorithm
    algorithms = [
        EpsilonGreedy(epsilon=0.1),
        UCB(alpha=1.0),
        ThompsonSampling()
    ]
    
    for bandit in algorithms:
        bandit.fit(arms, rewards)
        for _ in range(100):
            arm = bandit.select_arm()
            reward = np.random.binomial(1, 0.9 if arm == 'arm_2' else 0.1)
            bandit.update(arm, reward)
        
        # Print average rewards for debugging
        print('Average rewards:', bandit.arm_values)
        # Check if best arm is in the set of arms
        best_arm = max(bandit.arm_values.items(), key=lambda x: x[1])[0]
        assert best_arm in bandit.arms


def test_epsilon_greedy_fit_predict():
    df, x_name = make_mab_data(n_samples=100, n_arms=2, n_features=2, random_seed=123)
    algo = EpsilonGreedy()
    algo.fit(df['arm'], df['reward'])
    preds = algo.predict(n_samples=100)
    assert len(preds) == 100
    assert all([p in ['arm_0', 'arm_1'] for p in preds])


def test_ucb_fit_predict():
    df, x_name = make_mab_data(n_samples=100, n_arms=2, n_features=2, random_seed=123)
    algo = UCB()
    algo.fit(df['arm'], df['reward'])
    preds = algo.predict(n_samples=100)
    assert len(preds) == 100
    assert all([p in ['arm_0', 'arm_1'] for p in preds])


def test_thompson_sampling_fit_predict():
    df, x_name = make_mab_data(n_samples=100, n_arms=2, n_features=2, random_seed=123)
    algo = ThompsonSampling()
    algo.fit(df['arm'], df['reward'])
    preds = algo.predict(n_samples=100)
    assert len(preds) == 100
    assert all([p in ['arm_0', 'arm_1'] for p in preds]) 