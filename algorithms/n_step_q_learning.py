from __future__ import annotations

import numpy as np

from algorithms.base import BaseControlAlgorithm
from policies import EpsilonGreedyPolicy


class NStepQLearningControl(BaseControlAlgorithm):
    @property
    def name(self) -> str:
        return f"{self.config.n_step}-step Q-Learning"

    def train(self):
        n = self.config.n_step
        episode_rewards = []

        for episode in range(self.config.num_episodes):
            state, _ = self.env.reset(seed=self.seed + episode)

            states = [state]
            actions = []
            rewards = [0.0]
            T = np.inf
            t = 0
            total_reward = 0.0
            step_counter = 0

            while True:
                if t < T and step_counter < self.config.max_steps_per_episode:
                    action = EpsilonGreedyPolicy.select_action(
                        self.q_table, states[t], self.config.epsilon, self.rng
                    )
                    actions.append(action)

                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    rewards.append(reward)
                    states.append(next_state)
                    total_reward += reward
                    step_counter += 1

                    if terminated or truncated or step_counter >= self.config.max_steps_per_episode:
                        T = t + 1

                tau = t - n + 1
                if tau >= 0:
                    G = 0.0
                    upper = min(tau + n, int(T) if T != np.inf else tau + n)
                    for i in range(tau + 1, upper + 1):
                        G += (self.config.gamma ** (i - tau - 1)) * rewards[i]

                    if tau + n < T:
                        G += (self.config.gamma ** n) * np.max(self.q_table[states[tau + n]])

                    s_tau = states[tau]
                    a_tau = actions[tau]
                    self.q_table[s_tau, a_tau] += self.config.alpha * (
                        G - self.q_table[s_tau, a_tau]
                    )

                if tau == T - 1:
                    break

                t += 1

            episode_rewards.append(total_reward)

        return self.q_table.copy(), np.array(episode_rewards, dtype=float)