from __future__ import annotations

import os
from collections import defaultdict

import pandas as pd

from algorithms import (
    ExpectedSarsaControl,
    MonteCarloControl,
    NStepQLearningControl,
    NStepSarsaControl,
    QLearningControl,
    SarsaControl,
)
from config import ExperimentConfig
from evaluator import Evaluator
from logger_config import LoggerFactory
from utils import DataFrameUtils, FileUtils


class ExperimentManager:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = LoggerFactory.create_logger(log_dir=config.output_dir)

    def _build_algorithms(self, seed: int):
        return [
            MonteCarloControl(self.config, seed),
            SarsaControl(self.config, seed),
            QLearningControl(self.config, seed),
            ExpectedSarsaControl(self.config, seed),
            NStepSarsaControl(self.config, seed),
            NStepQLearningControl(self.config, seed),
        ]

    def run(self):
        FileUtils.ensure_dir(self.config.output_dir)
        FileUtils.save_json(
            self.config.to_dict(),
            os.path.join(self.config.output_dir, "hyperparameters.json"),
        )

        reward_history = defaultdict(list)
        evaluation_rows = []

        self.logger.info("Starting experiment")
        self.logger.info("Configuration: %s", self.config.to_dict())

        for seed in self.config.seeds:
            self.logger.info("Running experiments for seed=%s", seed)
            algorithms = self._build_algorithms(seed)

            for algorithm in algorithms:
                self.logger.info("Training algorithm=%s | seed=%s", algorithm.name, seed)

                q_table, rewards = algorithm.train()
                reward_history[algorithm.name].append(rewards)

                eval_mean, eval_std = Evaluator.evaluate_greedy_policy(
                    q_table=q_table,
                    config=self.config,
                    seed=10000 + seed,
                )

                evaluation_rows.append(
                    {
                        "algorithm": algorithm.name,
                        "seed": seed,
                        "avg_train_reward_last_100": float(rewards[-100:].mean()),
                        "greedy_eval_mean": eval_mean,
                        "greedy_eval_std": eval_std,
                    }
                )

                self.logger.info(
                    "Finished %s | seed=%s | last100=%.3f | eval_mean=%.3f | eval_std=%.3f",
                    algorithm.name,
                    seed,
                    rewards[-100:].mean(),
                    eval_mean,
                    eval_std,
                )

                algorithm.close()

        eval_df = pd.DataFrame(evaluation_rows)

        summary_df = (
            eval_df.groupby("algorithm", as_index=False)
            .agg(
                avg_train_reward_last_100_mean=("avg_train_reward_last_100", "mean"),
                avg_train_reward_last_100_std=("avg_train_reward_last_100", "std"),
                greedy_eval_mean_mean=("greedy_eval_mean", "mean"),
                greedy_eval_mean_std=("greedy_eval_mean", "std"),
            )
            .sort_values("greedy_eval_mean_mean", ascending=False)
            .reset_index(drop=True)
        )

        DataFrameUtils.save_training_rewards(
            reward_history,
            os.path.join(self.config.output_dir, "training_rewards.csv"),
        )

        self.logger.info("Experiment finished successfully")
        return reward_history, eval_df, summary_df