from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import MathUtils


class PlotManager:
    @staticmethod
    def plot_learning_curves_faceted(
        reward_history: dict,
        smoothing_window: int,
        save_path: str,
    ) -> None:
        algorithm_names = list(reward_history.keys())
        n_algorithms = len(algorithm_names)

        ncols = 2
        nrows = math.ceil(n_algorithms / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
        axes = np.array(axes).reshape(-1)

        for idx, algorithm_name in enumerate(algorithm_names):
            ax = axes[idx]
            runs = reward_history[algorithm_name]

            arr = np.vstack(runs)
            mean_rewards = arr.mean(axis=0)
            std_rewards = arr.std(axis=0)

            smooth_mean = MathUtils.moving_average(mean_rewards, smoothing_window)
            smooth_std = MathUtils.moving_average(std_rewards, smoothing_window)

            if len(mean_rewards) >= smoothing_window:
                x = np.arange(smoothing_window, len(mean_rewards) + 1)
            else:
                x = np.arange(1, len(mean_rewards) + 1)
                smooth_std = std_rewards

            ax.plot(x, smooth_mean, linewidth=2)
            ax.fill_between(
                x,
                smooth_mean - smooth_std,
                smooth_mean + smooth_std,
                alpha=0.2,
            )

            ax.set_title(algorithm_name)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Episodic Reward")
            ax.grid(True, alpha=0.3)

        for idx in range(n_algorithms, len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle("CliffWalking-v1: Learning Curves by Algorithm", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    @staticmethod
    def plot_learning_curves_combined(
        reward_history: dict,
        smoothing_window: int,
        save_path: str,
    ) -> None:
        plt.figure(figsize=(12, 7))

        for algorithm_name, runs in reward_history.items():
            arr = np.vstack(runs)
            mean_rewards = arr.mean(axis=0)
            std_rewards = arr.std(axis=0)

            smooth_mean = MathUtils.moving_average(mean_rewards, smoothing_window)
            smooth_std = MathUtils.moving_average(std_rewards, smoothing_window)

            if len(mean_rewards) >= smoothing_window:
                x = np.arange(smoothing_window, len(mean_rewards) + 1)
            else:
                x = np.arange(1, len(mean_rewards) + 1)
                smooth_std = std_rewards

            plt.plot(x, smooth_mean, label=algorithm_name)
            plt.fill_between(
                x,
                smooth_mean - smooth_std,
                smooth_mean + smooth_std,
                alpha=0.15,
            )

        plt.title("CliffWalking-v1: Learning Curves Averaged Across Seeds")
        plt.xlabel("Episode")
        plt.ylabel("Episodic Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_final_comparison(summary_df: pd.DataFrame, save_path: str) -> None:
        plt.figure(figsize=(10, 6))

        plt.bar(
            summary_df["algorithm"],
            summary_df["greedy_eval_mean_mean"],
            yerr=summary_df["greedy_eval_mean_std"],
            alpha=0.85,
        )

        plt.title("Final Performance Comparison")
        plt.ylabel("Average Greedy Evaluation Return")
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def save_tables(eval_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str) -> None:
        eval_df.to_csv(f"{output_dir}/per_seed_results.csv", index=False)
        summary_df.to_csv(f"{output_dir}/summary_results.csv", index=False)