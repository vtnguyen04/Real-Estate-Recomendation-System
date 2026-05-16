"""
Main script to trigger the inference pipeline for generating final submissions.
"""
import argparse
import polars as pl
from pathlib import Path

from src.utils.logging import get_logger
from config.settings import PipelineConfig
from config.paths import SUBMISSIONS_DIR
from src.pipeline.inference_pipeline import InferencePipeline
from src.core.base import RecommendationContext

logger = get_logger(__name__)


def main(output_file: str = "submission.csv", test_users_file: str = None):
    """
    Execute the full inference pipeline and generate recommendations.
    """
    logger.info("Starting inference pipeline for test set...")

    config = PipelineConfig()

    # Initialize the inference pipeline
    pipeline = InferencePipeline()

    logger.info("Loading test users...")
    
    # Load test users from file if provided, otherwise assume predicting for all unique users in a default path
    # In a Datathon scenario, typically a list of users is provided
    try:
        if test_users_file:
            if test_users_file.endswith('.parquet'):
                test_users_df = pl.read_parquet(test_users_file)
            else:
                test_users_df = pl.read_csv(test_users_file)
            
            # Assuming the column is 'user_id'
            test_users = test_users_df["user_id"].to_list()
        else:
            # Fallback for demonstration
            logger.warning("No --test-users file provided. Using sample fallback list.")
            test_users = ["user_1", "user_2", "user_3"]
            
        logger.info(f"Loaded {len(test_users)} users for inference.")
    except Exception as e:
        logger.error(f"Failed to load test users: {str(e)}")
        raise
    
    all_recommendations = []
    
    for user_id in test_users:
        # Pass the string directly as run method expects `user_id: str`
        recs_df = pipeline.run(user_id=user_id, k=config.top_k)

        # Collect and parse
        collected_df = recs_df.collect()
        if not collected_df.is_empty():
            item_ids = collected_df["item_id"].to_list()
            for rank, item_id in enumerate(item_ids, start=1):
                all_recommendations.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "rank": rank
                })

    # Save submission
    if all_recommendations:
        submission_df = pl.DataFrame(all_recommendations)
        output_path = SUBMISSIONS_DIR / output_file
        submission_df.write_csv(output_path)
        logger.info(f"Successfully wrote {len(all_recommendations)} recommendations to {output_path}")
    else:
        logger.warning("No recommendations generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the recommendation inference pipeline.")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output submission filename.")
    parser.add_argument("--test-users", type=str, default=None, help="Path to test users CSV or Parquet.")
    args = parser.parse_args()

    main(output_file=args.output, test_users_file=args.test_users)
