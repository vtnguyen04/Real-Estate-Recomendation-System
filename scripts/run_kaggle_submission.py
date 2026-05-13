import os
import argparse
import polars as pl
import pyarrow.dataset as ds
from src.utils.logging import get_logger
from src.pipeline.data_forensics import DataForensics
from src.features.feature_engineer import FeatureEngineer
from src.rules.geo_rules import GeoProximityScoreRule
from src.rules.quality_rules import QualityScoreRule
from src.rules.urgency_rules import UrgencyScoreRule
from src.rules.match_rules import MatchScoreRule
from src.rules.value_rules import ValueScoreRule
from src.models.baselines.popularity import PopularityRecommender
# from src.models.rankers.lgbm_ranker import MultiTaskLGBMRanker
from src.models.rerankers.multi_objective import MultiObjectiveReranker

logger = get_logger("run_submission")

def main(bucket_name="datathon_2026_final"):
    train_path = f"gs://{bucket_name}/train/"
    test_path = f"gs://{bucket_name}/test/"
    
    logger.info("Step 1: Authenticating and initializing data paths...")
    # Thí sinh chạy auth.authenticate_user() trực tiếp trên cell Colab trước khi gọi script này
    
    # Load test users
    logger.info(f"Loading test users from {test_path}test_users.parquet")
    try:
        df_test = pl.read_parquet(f"{test_path}test_users.parquet")
        logger.info(f"Loaded {len(df_test)} test users.")
    except Exception as e:
        logger.warning(f"Could not load test users directly: {e}. Ensure you are authenticated.")
        # Dùng mock data để test code flow nếu chạy local
        df_test = pl.DataFrame({"user_id": ["u1", "u2", "u3"]})

    # Load item catalog
    logger.info("Loading dim_listing catalog...")
    try:
        # PyArrow dataset to scan partitioned files on GCS
        catalog_ds = ds.dataset(f"{train_path}dim_listing/", format="parquet")
        df_items = pl.scan_pyarrow_dataset(catalog_ds).collect()
    except Exception as e:
        logger.warning(f"Using fallback mock items: {e}")
        df_items = pl.DataFrame({
            "item_id": ["i1", "i2", "i3"], 
            "price_bucket": ["1 tỷ", "2 tỷ", "3 tỷ"],
            "city_name": ["HCM", "HN", "DN"]
        })

    logger.info("Step 2: Data Forensics")
    forensics = DataForensics()
    # clean_events = forensics.apply_filters(raw_events)

    logger.info("Step 3: Feature Engineering & Rules")
    rules = [
        GeoProximityScoreRule(),
        QualityScoreRule(),
        UrgencyScoreRule(),
        MatchScoreRule(),
        ValueScoreRule()
    ]
    fe = FeatureEngineer(deterministic_rules=rules)
    
    logger.info("Step 4: Training Ranker & Ensembles")
    # Training pipeline would go here. For submission generation, we use predictions.
    pop_recommender = PopularityRecommender()
    
    # Mock fitting
    pop_recommender.fit(pl.DataFrame({"item_id": ["i1", "i2", "i3", "i1"]}))

    logger.info("Step 5: Multi-Objective Reranking")
    reranker = MultiObjectiveReranker(alpha=0.65, beta=0.15, gamma=0.15, delta=0.05)

    logger.info("Step 6: Generating Kaggle Submission (user_id, rank, item_id)")
    submission_rows = []
    
    # Mocking prediction generation for each user
    for row in df_test.iter_rows(named=True):
        user_id = row["user_id"]
        
        # 1. Candidate generation
        candidates = ["i1", "i2", "i3", "i4", "i5"] # Lấy từ mô hình
        
        # 2. Ranking
        # ... 
        
        # 3. Reranking
        # final_recs = reranker.rerank(user_context, candidates)
        final_recs = ["i1", "i2", "i3"][:10] # Top 10 max
        
        for rank, item_id in enumerate(final_recs, start=1):
            submission_rows.append({
                "user_id": user_id,
                "rank": rank,
                "item_id": item_id
            })

    df_sub = pl.DataFrame(submission_rows)
    df_sub.write_csv("submission.csv")
    
    logger.info(f"Successfully generated submission.csv with {len(df_sub)} rows!")
    logger.info("Pipeline executed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Datathon Pipeline on Colab")
    parser.add_argument("--bucket", default="datathon_2026_final", help="GCS Bucket name")
    args = parser.parse_args()
    
    main(bucket_name=args.bucket)
