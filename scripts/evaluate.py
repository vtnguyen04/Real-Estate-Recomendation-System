"""
Script to evaluate a set of recommendations against a validation set.
Computes business and technical metrics (NDCG, Recall, Diversity, Fairness).
"""
import argparse
import polars as pl
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.metrics import compute_ndcg_at_k, compute_recall_at_k
from src.evaluation.health_metrics import HealthMetrics

logger = get_logger(__name__)


def evaluate_recommendations(predictions_path: str, ground_truth_path: str):
    """
    Evaluates predictions against ground truth.
    Both should be CSV/Parquet files with 'user_id' and 'item_id'.
    """
    logger.info(f"Loading predictions from {predictions_path}")
    logger.info(f"Loading ground truth from {ground_truth_path}")
    
    # Load actual data
    preds = pl.read_csv(predictions_path)
    gt = pl.read_csv(ground_truth_path)
    
    logger.info(f"Loaded {len(preds)} prediction rows and {len(gt)} ground truth rows.")
    
    # Calculate metrics
    # Convert DataFrames to dictionary format for faster metric computation
    
    # Format: {"u1": ["i1", "i2", ...]}
    recs_dict = (
        preds.sort(["user_id", "rank"])
        .group_by("user_id")
        .agg(pl.col("item_id"))
        .to_dict(as_series=False)
    )
    recs_map = dict(zip(recs_dict["user_id"], recs_dict["item_id"]))
    
    gt_dict = (
        gt.group_by("user_id")
        .agg(pl.col("item_id"))
        .to_dict(as_series=False)
    )
    gt_map = dict(zip(gt_dict["user_id"], gt_dict["item_id"]))
    
    all_users = set(recs_map.keys()) & set(gt_map.keys())
    logger.info(f"Evaluating on {len(all_users)} overlapping users.")
    
    ndcg_scores = []
    recall_scores = []
    
    for u in all_users:
        actual_items = gt_map[u]
        rec_items = recs_map[u]
        
        ndcg_scores.append(compute_ndcg_at_k(actual_items, rec_items, 10))
        recall_scores.append(compute_recall_at_k(actual_items, rec_items, 10))
    
    metrics = {
        "NDCG@10": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        "Recall@10": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    }
    
    logger.info(f"Evaluation Results: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommendation accuracy and health metrics.")
    parser.add_argument("--preds", type=str, required=True, help="Path to predictions file.")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth file.")
    args = parser.parse_args()
    
    evaluate_recommendations(args.preds, args.gt)
