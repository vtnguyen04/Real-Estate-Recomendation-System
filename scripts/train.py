"""
Main script to trigger the training pipeline end-to-end.
"""
import argparse
import polars as pl
from pathlib import Path

from src.utils.logging import get_logger
from config.settings import PipelineConfig
from config.paths import MODELS_DIR
from src.data.loader import ListingDataLoader, FactUserEventsLoader
from src.pipeline.data_forensics import DataForensicsPipeline
from src.models.candidates.als_recommender import ALSRecommender
from src.models.baselines.popularity import PopularityRecommender
from src.models.ensemble.weighted_ensemble import WeightedEnsembleRecommender
from src.features.feature_engineer import FeatureEngineer
from src.rules.geo_rules import GeoProximityScoreRule
from src.rules.quality_rules import QualityScoreRule
from src.rules.urgency_rules import UrgencyScoreRule
from src.rules.match_rules import MatchScoreRule
from src.rules.value_rules import ValueScoreRule
from src.models.rankers.lgbm_ranker import MultiTaskLGBMRanker
from src.pipeline.training_pipeline import TrainingPipeline

logger = get_logger(__name__)


def main(config_path: str = None):
    """
    Execute the full training pipeline.
    """
    logger.info("Starting end-to-end training pipeline...")

    config = PipelineConfig()

    logger.info("Setting up Loaders...")
    TRAIN_PATH = f"gs://{config.data.bucket_name}/train/"
    
    loader_items = ListingDataLoader(project_id="", gcs_path=f"{TRAIN_PATH}dim_listing/")
    lf_items = loader_items.load()
    
    loader_events = FactUserEventsLoader(project_id="", gcs_path=f"{TRAIN_PATH}fact_user_events/")
    lf_events = loader_events.load()

    logger.info("Setting up Components...")
    als = ALSRecommender(factors=64)
    pop = PopularityRecommender()
    ensemble_cg = WeightedEnsembleRecommender(models=[als, pop], weights=[0.9, 0.1])

    rules = [GeoProximityScoreRule(), QualityScoreRule(), UrgencyScoreRule(), MatchScoreRule(), ValueScoreRule()]
    fe = FeatureEngineer(deterministic_rules=rules)
    ranker = MultiTaskLGBMRanker()

    # Initialize the training pipeline
    pipeline = TrainingPipeline(
        candidate_generator=ensemble_cg,
        feature_engineer=fe,
        ranker=ranker,
        config={"validation_days": config.validation_days}
    )

    # Run the pipeline
    try:
        results = pipeline.run(raw_events=lf_events, item_profile=lf_items)
        logger.info(f"Training completed successfully. Status: {results['status']}")

        # Save pipeline models
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pipeline.save_models(str(MODELS_DIR))

        logger.info("Training pipeline finished.")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the recommendation training pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file (optional).")
    args = parser.parse_args()

    main(config_path=args.config)
