"""
Main script to trigger the training pipeline end-to-end.
"""
import argparse
import polars as pl
from pathlib import Path

from src.utils.logging import get_logger
from config.settings import PipelineConfig
from config.paths import MODELS_DIR
from src.pipeline.training_pipeline import TrainingPipeline

logger = get_logger(__name__)


def main(config_path: str = None):
    """
    Execute the full training pipeline.
    """
    logger.info("Starting end-to-end training pipeline...")

    # In a real scenario, we might load config from `config_path` YAML file.
    # Here we use the default dataclass.
    config = PipelineConfig()

    # Initialize the training pipeline
    pipeline = TrainingPipeline(
        validation_days=config.validation_days,
        bucket_name=config.data.bucket_name
    )

    # Run the pipeline
    try:
        metrics, _ = pipeline.run()
        logger.info(f"Training completed successfully. Validation Metrics: {metrics}")

        # Ensure models directory exists and save pipeline
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # Note: Depending on implementation, you might save specific models here
        # E.g., pipeline.ranker.save(str(MODELS_DIR / "lgbm_ranker.txt"))

        logger.info("Training pipeline finished.")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the recommendation training pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file (optional).")
    args = parser.parse_args()

    main(config_path=args.config)
