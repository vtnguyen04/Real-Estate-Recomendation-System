import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional
from src.core.base import RecommendationContext
from src.evaluation.health_metrics import HealthMetrics

class MultiObjectiveReranker:
    """
    Stage 4 Reranker: Greedy selection maximizing Accuracy, Diversity, Fairness, and Freshness.
    Score = α·Accuracy + β·Diversity + γ·Fairness + δ·Freshness
    """
    def __init__(self, 
                 alpha: float = 0.55, 
                 beta: float = 0.15, 
                 gamma: float = 0.15, 
                 delta: float = 0.05,
                 epsilon: float = 0.10,
                 health_metrics: Optional[HealthMetrics] = None,
                 config: Optional[Dict[str, Any]] = None):
        
        self.config = config or {}
        self.alpha = self.config.get('rerank_alpha', alpha)
        self.beta = self.config.get('rerank_beta', beta)
        self.gamma = self.config.get('rerank_gamma', gamma)
        self.delta = self.config.get('rerank_delta', delta)
        self.epsilon = self.config.get('rerank_epsilon', epsilon)
        
        self.metrics = health_metrics or HealthMetrics()

    def rerank(self, candidates: pl.LazyFrame, k: int = 10) -> pl.LazyFrame:
        """
        Greedy re-ranking algorithm.
        """
        # Collect to memory for greedy iteration (usually small N like 30-50)
        df_cand = candidates.collect()
        if df_cand.is_empty():
            return candidates
            
        items_list = df_cand.to_dicts()
        selected = []
        remaining = items_list.copy()
        
        # Min-max normalize 'score' (accuracy) in the candidates
        scores = [it.get('score', 0.0) for it in items_list]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s > min_s else 1.0
        
        # Min-max normalize novelty (-log(contacts + 1)) INS-022
        import math
        contacts_list = [it.get('item_total_contacts', 0.0) for it in items_list]
        novels = [-math.log1p(c) for c in contacts_list]
        min_n, max_n = min(novels), max(novels)
        range_n = max_n - min_n if max_n > min_n else 1.0
        
        for i, it in enumerate(remaining):
            it['norm_accuracy'] = (it.get('score', 0.0) - min_s) / range_s
            it['norm_novelty'] = (novels[i] - min_n) / range_n

        for _ in range(min(k, len(items_list))):
            best_score = -1.0
            best_idx = -1
            
            for idx, item in enumerate(remaining):
                # Try adding this item
                temp_list = selected + [item]
                
                # Accuracy term
                accuracy = item['norm_accuracy']
                novelty = item['norm_novelty']
                
                # Diversity/Fairness/Freshness terms
                diversity = self.metrics.compute_diversity(temp_list)
                fairness = self.metrics.compute_fairness(temp_list)
                freshness = self.metrics.compute_freshness(temp_list)
                
                total_score = (self.alpha * accuracy + 
                               self.beta * diversity + 
                               self.gamma * fairness + 
                               self.delta * freshness +
                               self.epsilon * novelty)
                
                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx
            
            if best_idx != -1:
                selected.append(remaining.pop(best_idx))
            else:
                break
                
        return pl.DataFrame(selected).lazy()

    def rerank_batch(self, candidates_df: pl.DataFrame, k: int = 10) -> pl.DataFrame:
        """
        Applies reranking to a batched DataFrame grouped by user_id.
        """
        # Map columns for HealthMetrics compatibility
        # HealthMetrics expects: category, city_name, seller_type, listing_age_days, item_total_contacts
        if "category" not in candidates_df.columns and "item_cat" in candidates_df.columns:
            candidates_df = candidates_df.with_columns(pl.col("item_cat").alias("category"))
        if "city_name" not in candidates_df.columns and "item_city" in candidates_df.columns:
            candidates_df = candidates_df.with_columns(pl.col("item_city").alias("city_name"))
        if "seller_type" not in candidates_df.columns and "item_is_agent" in candidates_df.columns:
            candidates_df = candidates_df.with_columns(
                pl.when(pl.col("item_is_agent") == 1.0).then(pl.lit("agent")).otherwise(pl.lit("private")).alias("seller_type")
            )
        if "score" not in candidates_df.columns and "lgbm_score" in candidates_df.columns:
            candidates_df = candidates_df.with_columns(pl.col("lgbm_score").alias("score"))

        # Sort by user_id and score descending to get top_n_for_rerank before greedy selection
        top_n = self.config.get("top_n_for_rerank", 30)
        df_top = (
            candidates_df.sort(["user_id", "score"], descending=[False, True])
            .group_by("user_id", maintain_order=True)
            .head(top_n)
        )

        # Rerank per user
        all_selected = []
        user_groups = df_top.partition_by("user_id", as_dict=True)
        for uid, df_group in user_groups.items():
            reranked_df = self.rerank(df_group.lazy(), k=k).collect()
            all_selected.append(reranked_df)

        if not all_selected:
            return pl.DataFrame()
            
        return pl.concat(all_selected, how="diagonal_relaxed")
