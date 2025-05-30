import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod

# ============= Classical Multi-Armed Bandit Algorithms =============

class BaseBandit(ABC):
    """Base class for all bandit algorithms."""
    def __init__(self, batch_size: int = 1):
        # Number of samples to process in each batch
        self.batch_size = batch_size
        # List of unique arm identifiers
        self.arms = None
        # Total number of unique arms
        self.n_arms = None
        # Dictionary tracking number of times each arm has been pulled
        self.n_pulls = None
        # Dictionary tracking cumulative rewards for each arm
        self.rewards = None
        # Dictionary tracking average reward (value) for each arm
        self.arm_values = None

    def fit(self, arm, reward):
        """
        Fit the bandit model to offline data.
        
        Args:
            arm: arm assignment (array-like)
            reward: observed reward (array-like)
            
        Returns:
            self: returns an instance of self.
        """
        # Reset stats
        self.arms = np.unique(arm)
        self.n_arms = len(self.arms)
        self.n_pulls = {a: 0 for a in self.arms}
        self.rewards = {a: 0.0 for a in self.arms}
        self.arm_values = {a: 0.0 for a in self.arms}
        for a, r in zip(arm, reward):
            self.update(a, r)
        return self

    def predict(self, n_samples: int = 1):
        """
        Predict the best arm for n_samples.
        
        Args:
            n_samples: number of predictions to make
            
        Returns:
            list: list of predicted arms
        """
        return [self.select_arm() for _ in range(n_samples)]

    @abstractmethod
    def select_arm(self, context=None) -> int:
        """Select an arm based on the current state or context."""
        pass

    def update(self, chosen_arm: int, reward: float, context=None) -> None:
        """Update the model with the observed reward."""
        self.n_pulls[chosen_arm] += 1
        self.rewards[chosen_arm] += reward
        self.arm_values[chosen_arm] = self.rewards[chosen_arm] / self.n_pulls[chosen_arm]

    def batch_update(self, chosen_arms: List[int], rewards: List[float], contexts=None) -> None:
        """Update the model with a batch of observations."""
        for i, (arm, reward) in enumerate(zip(chosen_arms, rewards)):
            context = contexts[i] if contexts is not None else None
            self.update(arm, reward, context)

    def get_arm_values(self) -> Dict[int, float]:
        """Get the current value estimates for all arms."""
        return self.arm_values

    def run(self) -> pd.DataFrame:
        """Run the bandit algorithm on the entire dataset."""
        selected_arms = []
        rewards = []
        for i in range(0, len(self.df), self.batch_size):
            batch = self.df.iloc[i:i+self.batch_size]
            chosen_arms = [self.select_arm() for _ in range(len(batch))]
            reward_batch = np.where(batch[self.arm].values == chosen_arms, batch[self.reward].values, 0)
            self.batch_update(chosen_arms, reward_batch)
            selected_arms.extend(chosen_arms)
            rewards.extend(reward_batch)
        self.df['chosen_arm'] = selected_arms
        self.df['observed_reward'] = rewards
        return self.df

class EpsilonGreedy(BaseBandit):
    """Epsilon Greedy bandit algorithm."""
    def __init__(self, epsilon: float = 0.1, batch_size: int = 1):
        super().__init__(batch_size)
        # Probability of exploration (random arm selection)
        self.epsilon = epsilon

    def select_arm(self, context=None) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.arms)
        else:
            return max(self.arm_values.items(), key=lambda x: x[1])[0]

class UCB(BaseBandit):
    """Upper Confidence Bound (UCB) bandit algorithm."""
    def __init__(self, alpha: float = 1.0, batch_size: int = 1):
        super().__init__(batch_size)
        # Exploration parameter controlling the width of the confidence bound
        self.alpha = alpha

    def select_arm(self, context=None) -> int:
        unexplored_arms = [arm for arm in self.arms if self.n_pulls[arm] == 0]
        if unexplored_arms:
            return unexplored_arms[0]
        total_pulls = sum(self.n_pulls.values())
        ucb_values = {
            arm: self.arm_values[arm] + self.alpha * np.sqrt(
                np.log(total_pulls) / self.n_pulls[arm]
            )
            for arm in self.arms
        }
        return max(ucb_values.items(), key=lambda x: x[1])[0]

class ThompsonSampling(BaseBandit):
    """Thompson Sampling bandit algorithm."""
    def __init__(self, batch_size: int = 1):
        super().__init__(batch_size)
        # Dictionary of alpha parameters for Beta distribution of each arm
        self.alpha = None
        # Dictionary of beta parameters for Beta distribution of each arm
        self.beta = None

    def fit(self, arm, reward):
        """
        Fit the bandit model to offline data.
        
        Args:
            arm: arm assignment (array-like)
            reward: observed reward (array-like)
            
        Returns:
            self: returns an instance of self.
        """
        self.arms = np.unique(arm)
        self.n_arms = len(self.arms)
        self.n_pulls = {a: 0 for a in self.arms}
        self.rewards = {a: 0.0 for a in self.arms}
        self.arm_values = {a: 0.0 for a in self.arms}
        self.alpha = {a: 1.0 for a in self.arms}
        self.beta = {a: 1.0 for a in self.arms}
        for a, r in zip(arm, reward):
            self.update(a, r)
            self.alpha[a] += r
            self.beta[a] += (1 - r)
        return self

    def select_arm(self, context=None) -> int:
        samples = {
            arm: np.random.beta(self.alpha[arm], self.beta[arm])
            for arm in self.arms
        }
        return max(samples.items(), key=lambda x: x[1])[0]

    def update(self, chosen_arm: int, reward: float, context=None) -> None:
        super().update(chosen_arm, reward)
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += (1 - reward)

class BatchBandit:
    """Wrapper class for batch processing with any bandit algorithm."""
    def __init__(self, bandit: BaseBandit, batch_size: int):
        # The underlying bandit algorithm to be used
        self.bandit = bandit
        # Number of samples to process in each batch
        self.batch_size = batch_size

    def select_batch(self) -> List[int]:
        return [self.bandit.select_arm() for _ in range(self.batch_size)]

    def update_batch(self, arms: List[int], rewards: List[float]) -> None:
        self.bandit.batch_update(arms, rewards)

    def get_arm_values(self) -> Dict[int, float]:
        return self.bandit.get_arm_values()

    def fit(self, arm, reward):
        self.bandit.fit(arm, reward)
        return self

    def predict(self, n_samples: int = 1):
        """Predict the best arm for n_samples using the underlying bandit."""
        return self.bandit.predict(n_samples)

# ============= Contextual Multi-Armed Bandit Algorithms =============

class BaseContextualBandit(ABC):
    """Base class for all contextual bandit algorithms."""
    def __init__(self, batch_size: int = 1):
        # Number of samples to process in each batch
        self.batch_size = batch_size
        # List of unique arm identifiers
        self.arms = None
        # List of feature names used for context
        self.features = None
        # Total number of unique arms
        self.n_arms = None
        # Dictionary tracking number of times each arm has been pulled
        self.n_pulls = None
        # Dictionary tracking cumulative rewards for each arm
        self.rewards = None
        # Dictionary tracking average reward (value) for each arm
        self.arm_values = None

    @abstractmethod
    def fit(self, X, arm, reward, features):
        pass

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        pass

    @abstractmethod
    def update(self, chosen_arm: int, context: np.ndarray, reward: float) -> None:
        pass

    def batch_select(self, context_batch: np.ndarray) -> List[int]:
        return [self.select_arm(context) for context in context_batch]

    def batch_update(self, chosen_arms: List[int], context_batch: np.ndarray, rewards: np.ndarray) -> None:
        for arm, context, reward in zip(chosen_arms, context_batch, rewards):
            self.update(arm, context, reward)

class LinUCB(BaseContextualBandit):
    """Linear Upper Confidence Bound (LinUCB) bandit algorithm."""
    def __init__(self, alpha: float = 1.0, batch_size: int = 1):
        super().__init__(batch_size)
        # Exploration parameter controlling the width of the confidence bound
        self.alpha = alpha
        # Dimension of the feature space
        self.d = None
        # Dictionary of A matrices (feature covariance matrices) for each arm
        self.A = None
        # Dictionary of b vectors (feature-reward correlation) for each arm
        self.b = None

    def fit(self, X, arm, reward, features):
        self.features = features
        self.arms = np.unique(arm)
        self.n_arms = len(self.arms)
        self.d = len(features)
        self.n_pulls = {a: 0 for a in self.arms}
        self.rewards = {a: 0.0 for a in self.arms}
        self.arm_values = {a: 0.0 for a in self.arms}
        self.A = {a: np.identity(self.d, dtype=np.float64) for a in self.arms}
        self.b = {a: np.zeros(self.d, dtype=np.float64) for a in self.arms}
        for x, a, r in zip(X[features].values, arm, reward):
            self.update(a, x, r)
        return self

    def select_arm(self, context: np.ndarray) -> int:
        ucb_values = {}
        for arm in self.arms:
            theta = np.dot(np.linalg.inv(self.A[arm]), self.b[arm])
            ucb = np.dot(context, theta) + self.alpha * np.sqrt(
                np.dot(np.dot(context, np.linalg.inv(self.A[arm])), context)
            )
            ucb_values[arm] = ucb
        return max(ucb_values.items(), key=lambda x: x[1])[0]

    def update(self, chosen_arm: int, context: np.ndarray, reward: float) -> None:
        context = context.astype(np.float64)
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += reward * context

class BatchLinUCB(LinUCB):
    """Batch Linear Upper Confidence Bound (BatchLinUCB) bandit algorithm."""
    def __init__(self, alpha: float = 1.0, batch_size: int = 32):
        super().__init__(alpha, batch_size)
        # Store data as numpy arrays
        self.X = None  # Context features
        self.arms = None  # Available arms
        self.d = None  # Feature dimension

    def fit(self, X, arm, reward):
        """
        Fit the bandit model to offline data.
        
        Args:
            X: Context features (numpy array)
            arm: Arm assignments (numpy array)
            reward: Observed rewards (numpy array)
            
        Returns:
            self: returns an instance of self.
        """
        # Convert inputs to numpy arrays if they aren't already
        self.X = np.asarray(X, dtype=np.float64)
        arm = np.asarray(arm)
        reward = np.asarray(reward)
        
        self.arms = np.unique(arm)
        self.d = self.X.shape[1]
        self.n_pulls = {a: 0 for a in self.arms}
        self.rewards = {a: 0.0 for a in self.arms}
        self.arm_values = {a: 0.0 for a in self.arms}
        
        # Initialize A and b for each arm
        self.A = {a: np.identity(self.d, dtype=np.float64) for a in self.arms}
        self.b = {a: np.zeros(self.d, dtype=np.float64) for a in self.arms}
        
        return self

    def select_arm(self, context_batch):
        """
        Select arms for a batch of contexts using vectorized operations.
        
        Args:
            context_batch: Array of context features for the batch
            
        Returns:
            list: List of selected arms
        """
        ucb_values = {}
        
        for arm in self.arms:
            theta = np.dot(np.linalg.inv(self.A[arm]), self.b[arm])
            # Compute UCB values for the entire batch using einsum
            ucb = np.dot(context_batch, theta) + self.alpha * np.sqrt(
                np.einsum('ij,jk,ik->i', context_batch, np.linalg.inv(self.A[arm]), context_batch)
            )
            ucb_values[arm] = ucb
        
        # Select the arm with the highest UCB value for each context in the batch
        chosen_arms = np.argmax(np.array(list(ucb_values.values())).T, axis=1)
        
        return [self.arms[i] for i in chosen_arms]

    def update_batch(self, chosen_arms, context_batch, rewards):
        """
        Update the model with a batch of observations using vectorized operations.
        
        Args:
            chosen_arms: Array of chosen arms
            context_batch: Array of context features
            rewards: Array of observed rewards
        """
        for arm in self.arms:
            indices = np.where(chosen_arms == arm)[0]
            if len(indices) == 0:
                continue
            
            X = context_batch[indices]
            R = rewards[indices]
            
            self.A[arm] += np.dot(X.T, X)
            self.b[arm] += np.dot(X.T, R)

    def run(self):
        """
        Run the bandit algorithm on the entire dataset.
        
        Returns:
            tuple: (selected_arms, observed_rewards)
        """
        selected_arms = []
        rewards = []
        
        # Process the data in batches
        for i in range(0, len(self.X), self.batch_size):
            batch_end = min(i + self.batch_size, len(self.X))
            context_batch = self.X[i:batch_end]
            chosen_arms = self.select_arm(context_batch)
            
            # Calculate rewards for the batch
            reward_batch = np.zeros(len(chosen_arms))
            for j, (chosen_arm, true_arm) in enumerate(zip(chosen_arms, self.arms[i:batch_end])):
                if chosen_arm == true_arm:
                    reward_batch[j] = self.rewards[true_arm]
            
            self.update_batch(chosen_arms, context_batch, reward_batch)
            
            selected_arms.extend(chosen_arms)
            rewards.extend(reward_batch)
        
        return np.array(selected_arms), np.array(rewards)

class CohortThompsonSampling(BaseContextualBandit):
    """Cohort Thompson Sampling bandit algorithm."""
    def __init__(self, batch_size: int = 1):
        super().__init__(batch_size)
        # Name of the feature used for cohorting
        self.cohort_feature = None
        # List of unique cohort values
        self.cohorts = None
        # Nested dictionary tracking successful pulls for each arm in each cohort
        self.successes = None
        # Nested dictionary tracking failed pulls for each arm in each cohort
        self.failures = None

    def fit(self, X, arm, reward, cohort_feature):
        self.cohort_feature = cohort_feature
        self.arms = np.unique(arm)
        self.cohorts = np.unique(X[cohort_feature])
        self.successes = {cohort: {a: 0 for a in self.arms} for cohort in self.cohorts}
        self.failures = {cohort: {a: 0 for a in self.arms} for cohort in self.cohorts}
        for i in range(len(X)):
            context = np.array([X.iloc[i][cohort_feature]])
            a = arm.iloc[i] if hasattr(arm, 'iloc') else arm[i]
            r = reward.iloc[i] if hasattr(reward, 'iloc') else reward[i]
            self.update(a, context, r)
        return self

    def select_arm(self, context: np.ndarray) -> int:
        cohort = context[0]
        sampled_theta = {}
        for arm in self.arms:
            a = self.successes[cohort][arm] + 1
            b = self.failures[cohort][arm] + 1
            sampled_theta[arm] = np.random.beta(a, b)
        return max(sampled_theta, key=sampled_theta.get)

    def update(self, chosen_arm: int, context: np.ndarray, reward: float) -> None:
        cohort = context[0]
        if reward > 0:
            self.successes[cohort][chosen_arm] += 1
        else:
            self.failures[cohort][chosen_arm] += 1

class BatchCohortThompsonSampling(CohortThompsonSampling):
    """Batch Cohort Thompson Sampling bandit algorithm."""
    def __init__(self, batch_size: int = 32):
        super().__init__(batch_size)
        # Store data as numpy arrays
        self.X = None  # Context features
        self.arm_assignments = None  # Actual arm assignments for each sample
        self.rewards_array = None    # Actual observed rewards for each sample
        self.cohorts = None  # Available cohort values

    def fit(self, X, arm, reward, cohort_feature):
        """
        Fit the bandit model to offline data.
        
        Args:
            X: Context features (numpy array)
            arm: Arm assignments (numpy array)
            reward: Observed rewards (numpy array)
            cohort_feature: Index of the feature to use for cohorting, or an array of cohort assignments
        
        Returns:
            self: returns an instance of self.
        """
        # Convert inputs to numpy arrays if they aren't already
        self.X = np.asarray(X, dtype=np.float64)
        self.arm_assignments = np.asarray(arm)
        self.rewards_array = np.asarray(reward)

        # If cohort_feature is an array, use it directly; if int, use as column index
        if isinstance(cohort_feature, (np.ndarray, list)):
            cohort_feature = np.asarray(cohort_feature)
            self.cohort_assignments = cohort_feature
        else:
            self.cohort_assignments = self.X[:, cohort_feature]
        self.cohorts = np.unique(self.cohort_assignments)
        self.arms = np.unique(self.arm_assignments)

        # Initialize success and failure counts for each arm in each cohort
        self.successes = {cohort: {a: 0 for a in self.arms} for cohort in self.cohorts}
        self.failures = {cohort: {a: 0 for a in self.arms} for cohort in self.cohorts}
        self.cohort_feature = cohort_feature
        return self

    def select_arm_batch(self, cohort_batch):
        """
        Select arms for a batch of cohorts.
        
        Args:
            cohort_batch: Array of cohort values for the batch
        
        Returns:
            list: List of selected arms
        """
        chosen_arms = []
        for cohort in cohort_batch:
            sampled_theta = {}
            for arm in self.arms:
                # Draw samples from the Beta distribution for each arm
                a = self.successes[cohort][arm] + 1
                b = self.failures[cohort][arm] + 1
                sampled_theta[arm] = np.random.beta(a, b)
            # Select the arm with the highest sample
            chosen_arm = max(sampled_theta, key=sampled_theta.get)
            chosen_arms.append(chosen_arm)
        return chosen_arms

    def update_batch(self, cohort_batch, chosen_arms, reward_batch):
        """
        Update the model with a batch of observations.
        
        Args:
            cohort_batch: Array of cohort values
            chosen_arms: Array of chosen arms
            reward_batch: Array of observed rewards
        """
        for cohort, arm, reward in zip(cohort_batch, chosen_arms, reward_batch):
            if reward > 0:
                self.successes[cohort][arm] += 1
            else:
                self.failures[cohort][arm] += 1

    def run(self):
        """
        Run the bandit algorithm on the entire dataset.
        
        Returns:
            tuple: (selected_arms, observed_rewards)
        """
        selected_arms = []
        rewards = []
        # Process the data in batches
        for i in range(0, len(self.X), self.batch_size):
            batch_end = min(i + self.batch_size, len(self.X))
            cohort_batch = self.cohort_assignments[i:batch_end]
            arm_batch = self.arm_assignments[i:batch_end]
            reward_batch_true = self.rewards_array[i:batch_end]
            # Select arms for the entire batch
            chosen_arms = self.select_arm_batch(cohort_batch)
            # Calculate rewards for the batch
            reward_batch = np.zeros(len(chosen_arms))
            for j, (chosen_arm, true_arm, true_reward) in enumerate(zip(chosen_arms, arm_batch, reward_batch_true)):
                if chosen_arm == true_arm:
                    reward_batch[j] = true_reward
            # Update the arms with the observed rewards
            self.update_batch(cohort_batch, chosen_arms, reward_batch)
            selected_arms.extend(chosen_arms)
            rewards.extend(reward_batch)
        return np.array(selected_arms), np.array(rewards) 