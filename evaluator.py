from __future__ import annotations

import numpy as np

from config import ExperimentConfig
from env_manager import EnvironmentManager
from policies import EpsilonGreedyPolicy


class Evaluator:
    @staticmethod
    def evaluate_greedy_policy(q_table: np.ndarray, config: ExperimentConfig, seed: int) -> tuple[float, float]:
        env = EnvironmentManager.create_env(config.env_name, seed=seed)
        returns = []

        for episode in range(config.eval_episodes):
            state, _ = env.reset(seed=seed + episode)
            total_reward = 0.0

            for _ in range(config.max_steps_per_episode):
                action = EpsilonGreedyPolicy.greedy_action(q_table, state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            returns.append(total_reward)

        env.close()
        return float(np.mean(returns)), float(np.std(returns))