from __future__ import annotations

import gymnasium as gym


class EnvironmentManager:
    @staticmethod
    def create_env(env_name: str, seed: int | None = None):
        env = gym.make(env_name)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return env