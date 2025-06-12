"""
Reinforcement Learning Environment for Lead Prioritization
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class LeadPrioritizationEnv(gym.Env):
    """Custom environment for lead prioritization using reinforcement learning"""
    
    def __init__(self, features, relevance_scores):
        super().__init__()
        self.features = features
        self.relevance_scores = relevance_scores
        self.n_leads = len(relevance_scores)
        self.current_idx = 0

        # Define observation space - shape based on feature dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(features.shape[1],),
            dtype=np.float32
        )

        # Define action space - continuous value between 0 and 1 for prioritization
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        self.current_idx = 0
        return self.features[self.current_idx], {}

    def step(self, action):
        """Execute one step in the environment"""
        action_value = float(np.clip(action[0], 0.0, 1.0))
        relevance = float(self.relevance_scores[self.current_idx])
        done = self.current_idx == self.n_leads - 1

        # Reward calculation based on action and relevance alignment
        if action_value >= 0.8:  # High priority action
            if relevance >= 0.7:
                reward = 10 * relevance
            elif relevance >= 0.4:
                reward = 3 * relevance
            else:
                reward = -5
        elif 0.5 <= action_value < 0.8:  # Medium priority action
            if relevance >= 0.7:
                reward = 5 * relevance
            elif relevance >= 0.4:
                reward = 1 * relevance
            else:
                reward = -2
        else:  # Low priority action
            if relevance >= 0.7:
                reward = -7
            elif relevance >= 0.4:
                reward = -1
            else:
                reward = 0.5

        self.current_idx += 1

        # Get next observation
        if not done:
            obs = self.features[self.current_idx]
        else:
            obs = np.zeros_like(self.features[0])

        return obs, reward, done, False, {}

    def get_current_lead_info(self):
        """Get information about current lead being processed"""
        if self.current_idx < self.n_leads:
            return {
                'index': self.current_idx,
                'relevance': self.relevance_scores[self.current_idx],
                'features': self.features[self.current_idx]
            }
        return None