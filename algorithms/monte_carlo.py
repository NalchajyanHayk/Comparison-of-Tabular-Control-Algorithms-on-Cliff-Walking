from __future__ import annotations

import numpy as np

from algorithms.base import BaseControlAlgorithm
from policies import EpsilonGreedyPolicy


class MonteCarloControl(BaseControlAlgorithm):
    @property
    def name(self) -> str:
        return "Monte Carlo"

    def train(self):
        returns_sum = np.zeros_like(self.q_table, dtype=float)
        returns_count = np.zeros_like(self.q_table, dtype=float)
        episode_rewards = []

        for episode in range(self.config.num_episodes):
            state, _ = self.env.reset(seed=self.seed + episode)
            trajectory = []
            total_reward = 0.0

            for _ in range(self.config.max_steps_per_episode):
                action = EpsilonGreedyPolicy.select_action(
                    self.q_table, state, self.config.epsilon, self.rng
                )
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                trajectory.append((state, action, reward))
                total_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            G = 0.0
            for t in reversed(range(len(trajectory))):
                s, a, r = trajectory[t]
                G = self.config.gamma * G + r
                returns_sum[s, a] += G
                returns_count[s, a] += 1.0
                self.q_table[s, a] = returns_sum[s, a] / returns_count[s, a]

            episode_rewards.append(total_reward)

        return self.q_table.copy(), np.array(episode_rewards, dtype=float)