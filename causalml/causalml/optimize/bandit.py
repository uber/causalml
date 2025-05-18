import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod

# ============= Classical Multi-Armed Bandit Algorithms =============

class BaseBandit(ABC):
    """Base class for all bandit algorithms."""
    def __init__(self, df: pd.DataFrame, reward: str, arm: str, batch_size: int = 1):
        self.df = df
        self.reward = reward
        self.arm = arm
        self.batch_size = batch_size
        self.arms = df[arm].unique()
        self.n_arms = len(self.arms)
        self.n_pulls = {arm: 0 for arm in self.arms}
        self.rewards = {arm: 0.0 for arm in self.arms}
        self.arm_values = {arm: 0.0 for arm in self.arms}

    @abstractmethod
    def select_arm(self) -> int:
        """Select an arm based on the current state."""
        pass

    def update(self, chosen_arm: int, reward: float) -> None:
        """Update the model with the observed reward."""
        self.n_pulls[chosen_arm] += 1
        self.rewards[chosen_arm] += reward
        self.arm_values[chosen_arm] = self.rewards[chosen_arm] / self.n_pulls[chosen_arm]

    def batch_update(self, chosen_arms: List[int], rewards: List[float]) -> None:
        """Update the model with a batch of observations."""
        for arm, reward in zip(chosen_arms, rewards):
            self.update(arm, reward)

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
    def __init__(self, df: pd.DataFrame, reward: str, arm: str, epsilon: float = 0.1, batch_size: int = 1):
        super().__init__(df, reward, arm, batch_size)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.arms)
        else:
            return max(self.arm_values.items(), key=lambda x: x[1])[0]

class UCB(BaseBandit):
    """Upper Confidence Bound (UCB) bandit algorithm."""
    def __init__(self, df: pd.DataFrame, reward: str, arm: str, alpha: float = 1.0, batch_size: int = 1):
        super().__init__(df, reward, arm, batch_size)
        self.alpha = alpha

    def select_arm(self) -> int:
        # If any arm hasn't been pulled, select it
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
    def __init__(self, df: pd.DataFrame, reward: str, arm: str, batch_size: int = 1):
        super().__init__(df, reward, arm, batch_size)
        self.alpha = {arm: 1.0 for arm in self.arms}
        self.beta = {arm: 1.0 for arm in self.arms}

    def select_arm(self) -> int:
        samples = {
            arm: np.random.beta(self.alpha[arm], self.beta[arm])
            for arm in self.arms
        }
        return max(samples.items(), key=lambda x: x[1])[0]

    def update(self, chosen_arm: int, reward: float) -> None:
        super().update(chosen_arm, reward)
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += (1 - reward)

class BatchBandit:
    """Wrapper class for batch processing with any bandit algorithm."""
    def __init__(self, bandit: BaseBandit, batch_size: int):
        self.bandit = bandit
        self.batch_size = batch_size

    def select_batch(self) -> List[int]:
        return [self.bandit.select_arm() for _ in range(self.batch_size)]

    def update_batch(self, arms: List[int], rewards: List[float]) -> None:
        self.bandit.batch_update(arms, rewards)

    def get_arm_values(self) -> Dict[int, float]:
        return self.bandit.get_arm_values()

# ============= Contextual Multi-Armed Bandit Algorithms =============

class BaseContextualBandit(ABC):
    """Base class for all contextual bandit algorithms."""
    def __init__(self, df: pd.DataFrame, features: List[str], reward: str, arm: str, batch_size: int = 1):
        self.df = df
        self.features = features
        self.reward = reward
        self.arm = arm
        self.batch_size = batch_size
        self.arms = df[arm].unique()
        
        # Ensure the context features are numeric
        self.df[self.features] = self.df[self.features].astype(float)

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        """Select an arm based on the given context."""
        pass

    @abstractmethod
    def update(self, chosen_arm: int, context: np.ndarray, reward: float) -> None:
        """Update the model with the observed reward."""
        pass

    def batch_select(self, context_batch: np.ndarray) -> List[int]:
        """Select arms for a batch of contexts."""
        return [self.select_arm(context) for context in context_batch]

    def batch_update(self, chosen_arms: List[int], context_batch: np.ndarray, rewards: np.ndarray) -> None:
        """Update the model with a batch of observations."""
        for arm, context, reward in zip(chosen_arms, context_batch, rewards):
            self.update(arm, context, reward)

    def run(self) -> pd.DataFrame:
        """Run the bandit algorithm on the entire dataset."""
        selected_arms = []
        rewards = []
        
        for i in range(0, len(self.df), self.batch_size):
            batch = self.df.iloc[i:i+self.batch_size]
            context_batch = batch[self.features].values.astype(np.float64)
            chosen_arms = self.batch_select(context_batch)
            
            reward_batch = np.where(batch[self.arm].values == chosen_arms, batch[self.reward].values, 0)
            
            self.batch_update(chosen_arms, context_batch, reward_batch)
            
            selected_arms.extend(chosen_arms)
            rewards.extend(reward_batch)
        
        self.df['chosen_arm'] = selected_arms
        self.df['observed_reward'] = rewards
        
        return self.df

class LinUCB(BaseContextualBandit):
    """Linear Upper Confidence Bound (LinUCB) bandit algorithm."""
    def __init__(self, df: pd.DataFrame, features: List[str], reward: str, arm: str, alpha: float = 1.0, batch_size: int = 1):
        super().__init__(df, features, reward, arm, batch_size)
        self.alpha = alpha
        self.d = len(features)
        
        # Initialize A and b for each arm with float64 type
        self.A = {arm: np.identity(self.d, dtype=np.float64) for arm in self.arms}
        self.b = {arm: np.zeros(self.d, dtype=np.float64) for arm in self.arms}
         
    def select_arm(self, context: np.ndarray) -> int:
        ucb_values = {}
        
        for arm in self.arms:
            theta = np.dot(np.linalg.inv(self.A[arm]), self.b[arm])
            ucb = np.dot(context, theta) + self.alpha * np.sqrt(
                np.dot(np.dot(context, np.linalg.inv(self.A[arm])), context)
            )
            ucb_values[arm] = ucb
        
        return max(ucb_values, key=ucb_values.get)
    
    def update(self, chosen_arm: int, context: np.ndarray, reward: float) -> None:
        context = context.astype(np.float64)
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += reward * context

class BatchLinUCB(LinUCB):
    """Batch Linear Upper Confidence Bound (BatchLinUCB) bandit algorithm."""
    def __init__(self, df: pd.DataFrame, features: List[str], reward: str, arm: str, 
                 alpha: float = 1.0, batch_size: int = 32):
        super().__init__(df, features, reward, arm, alpha, batch_size)
    
    def batch_select(self, context_batch: np.ndarray) -> List[int]:
        ucb_values = {}
        
        for arm in self.arms:
            theta = np.dot(np.linalg.inv(self.A[arm]), self.b[arm])
            # Compute UCB values for the entire batch
            ucb = np.dot(context_batch, theta) + self.alpha * np.sqrt(
                np.einsum('ij,jk,ik->i', context_batch, np.linalg.inv(self.A[arm]), context_batch)
            )
            ucb_values[arm] = ucb
        
        # Select the arm with the highest UCB value for each context in the batch
        chosen_arms = np.argmax(np.array(list(ucb_values.values())).T, axis=1)
        
        return [self.arms[i] for i in chosen_arms]

class CohortThompsonSampling(BaseContextualBandit):
    """Cohort Thompson Sampling bandit algorithm."""
    def __init__(self, df: pd.DataFrame, feature: str, reward: str, arm: str, batch_size: int = 1):
        super().__init__(df, [feature], reward, arm, batch_size)
        self.feature = feature
        self.cohorts = df[feature].unique()
        
        # Initialize success and failure counts for each arm in each cohort
        self.successes = {cohort: {arm: 0 for arm in self.arms} for cohort in self.cohorts}
        self.failures = {cohort: {arm: 0 for arm in self.arms} for cohort in self.cohorts}

    def select_arm(self, context: np.ndarray) -> int:
        cohort = context[0]  # Since we only have one feature
        sampled_theta = {}
        for arm in self.arms:
            # Draw samples from the Beta distribution for each arm
            a = self.successes[cohort][arm] + 1
            b = self.failures[cohort][arm] + 1
            sampled_theta[arm] = np.random.beta(a, b)
        
        # Select the arm with the highest sample
        return max(sampled_theta, key=sampled_theta.get)

    def update(self, chosen_arm: int, context: np.ndarray, reward: float) -> None:
        cohort = context[0]  # Since we only have one feature
        # Update success and failure counts for the chosen arm
        if reward > 0:
            self.successes[cohort][chosen_arm] += 1
        else:
            self.failures[cohort][chosen_arm] += 1

class BatchCohortThompsonSampling(CohortThompsonSampling):
    """Batch Cohort Thompson Sampling bandit algorithm."""
    def __init__(self, df: pd.DataFrame, feature: str, reward: str, arm: str, batch_size: int = 32):
        super().__init__(df, feature, reward, arm, batch_size) 