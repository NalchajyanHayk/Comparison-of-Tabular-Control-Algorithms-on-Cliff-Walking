from __future__ import annotations

import os
import traceback

from config import ExperimentConfig
from experiment_manager import ExperimentManager
from plotting import PlotManager
from logger_config import LoggerFactory


class Main:
    @staticmethod
    def run() -> None:
        config = ExperimentConfig()
        logger = LoggerFactory.create_logger(log_dir=config.output_dir)

        try:
            logger.info("Launching main pipeline")
            logger.info("Configuration: %s", config.to_dict())

            manager = ExperimentManager(config)
            reward_history, eval_df, summary_df = manager.run()

            logger.info("Training completed. Creating plots and saving outputs.")

            PlotManager.plot_learning_curves(
                reward_history=reward_history,
                smoothing_window=config.smoothing_window,
                save_path=os.path.join(config.output_dir, "learning_curves.png"),
            )

            PlotManager.plot_final_comparison(
                summary_df=summary_df,
                save_path=os.path.join(config.output_dir, "final_comparison.png"),
            )

            PlotManager.save_tables(eval_df, summary_df, config.output_dir)

            logger.info("Per-seed results:\n%s", eval_df.to_string(index=False))
            logger.info("Summary across seeds:\n%s", summary_df.to_string(index=False))
            logger.info("Artifacts saved under: %s/", config.output_dir)

        except Exception as exc:
            logger.error("Pipeline failed: %s", exc)
            logger.error(traceback.format_exc())
            raise


if __name__ == "__main__":
    Main.run()