from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import pandas as pd


class FileUtils:
    @staticmethod
    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def save_json(data: Dict, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


class MathUtils:
    @staticmethod
    def moving_average(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1 or len(values) < window:
            return values.copy()
        return np.convolve(values, np.ones(window) / window, mode="valid")


class DataFrameUtils:
    @staticmethod
    def save_training_rewards(reward_history: dict, output_path: str) -> None:
        rows = []
        for algorithm_name, runs in reward_history.items():
            for seed_idx, rewards in enumerate(runs):
                for episode_idx, reward in enumerate(rewards, start=1):
                    rows.append(
                        {
                            "algorithm": algorithm_name,
                            "seed_index": seed_idx,
                            "episode": episode_idx,
                            "episodic_reward": float(reward),
                        }
                    )
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)