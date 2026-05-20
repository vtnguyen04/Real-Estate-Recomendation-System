"""
Evaluation metrics for recommendation quality.

Competition metrics: Recall@K, NDCG@K
Ground truth: event_type ∈ {view_phone, contact_chat, contact_zalo, contact_sms}
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)

GT_EVENTS = ['view_phone', 'contact_chat', 'other_interaction', 'contact_zalo', 'contact_sms']


def recall_at_k(predicted: list, actual: set, k: int = 10) -> float:
    """Recall@K = |predicted ∩ actual| / |actual|  (per competition formula)"""
    if not actual:
        return 0.0
    return len(set(predicted[:k]) & actual) / len(actual)


def ndcg_at_k(predicted: list, actual: set, k: int = 10) -> float:
    """NDCG@K with binary relevance."""
    if not actual:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, it in enumerate(predicted[:k]) if it in actual)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(actual))))
    return dcg / idcg if idcg > 0 else 0.0


def build_ground_truth(
    lf_events: pl.LazyFrame,
    split_date,
) -> Dict[str, Set[str]]:
    """
    Build ground truth dict from events after split_date.

    Returns:
        Dict[user_id, Set[item_id]] — items contacted in validation period.
    """
    gt_df = (
        lf_events
        .filter(pl.col("date") > split_date)
        .filter(pl.col("event_type").is_in(GT_EVENTS))
        .filter(pl.col("is_login") == "login")
        .filter(pl.col("user_id").is_not_null())
        .select(["user_id", "item_id"])
        .unique()
        .collect()
    )
    gt = defaultdict(set)
    for r in gt_df.iter_rows():
        gt[r[0]].add(r[1])
    return dict(gt)


def evaluate_recommendations(
    user_recs: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate a dict of user→recommendations against ground truth.

    Args:
        user_recs: Dict[user_id, List[item_id]] — predicted top-K per user.
        ground_truth: Dict[user_id, Set[item_id]] — actual contacts.
        k: cutoff.

    Returns:
        Dict with overall and per-segment metrics.
    """
    recall_scores = []
    ndcg_scores = []

    for uid, actual in ground_truth.items():
        preds = user_recs.get(uid, [])
        recall_scores.append(recall_at_k(preds, actual, k))
        ndcg_scores.append(ndcg_at_k(preds, actual, k))

    return {
        "recall_at_k": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "ndcg_at_k": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "num_users": len(ground_truth),
    }
