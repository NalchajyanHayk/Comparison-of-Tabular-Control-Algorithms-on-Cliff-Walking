from __future__ import annotations

import numpy as np

from algorithms.base import BaseControlAlgorithm
from policies import EpsilonGreedyPolicy


class SarsaControl(BaseControlAlgorithm):
    @property
    def name(self) -> str:
        return "SARSA"

    def train(self):
        episode_rewards = []

        for episode in range(self.config.num_episodes):
            state, _ = self.env.reset(seed=self.seed + episode)
            action = EpsilonGreedyPolicy.select_action(
                self.q_table, state, self.config.epsilon, self.rng
            )

            total_reward = 0.0

            for _ in range(self.config.max_steps_per_episode):
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward

                if terminated or truncated:
                    td_target = reward
                    self.q_table[state, action] += self.config.alpha * (
                        td_target - self.q_table[state, action]
                    )
                    break

                next_action = EpsilonGreedyPolicy.select_action(
                    self.q_table, next_state, self.config.epsilon, self.rng
                )
                td_target = reward + self.config.gamma * self.q_table[next_state, next_action]
                self.q_table[state, action] += self.config.alpha * (
                    td_target - self.q_table[state, action]
                )

                state, action = next_state, next_action

            episode_rewards.append(total_reward)

        return self.q_table.copy(), np.array(episode_rewards, dtype=float)