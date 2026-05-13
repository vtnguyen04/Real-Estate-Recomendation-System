import polars as pl
import numpy as np
from typing import List, Dict, Any
from src.core.base import RecommendationContext
from src.evaluation.health_metrics import HealthMetrics

class MultiObjectiveReranker:
    """
    Stage 4 Reranker: Greedy selection maximizing Accuracy, Diversity, Fairness, and Freshness.
    Score = α·Accuracy + β·Diversity + γ·Fairness + δ·Freshness
    """
    def __init__(self, 
                 alpha: float = 0.65, 
                 beta: float = 0.15, 
                 gamma: float = 0.15, 
                 delta: float = 0.05,
                 health_metrics: HealthMetrics = None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
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
        
        for it in remaining:
            it['norm_accuracy'] = (it.get('score', 0.0) - min_s) / range_s

        for _ in range(min(k, len(items_list))):
            best_score = -1.0
            best_idx = -1
            
            for idx, item in enumerate(remaining):
                # Try adding this item
                temp_list = selected + [item]
                
                # Accuracy term
                accuracy = item['norm_accuracy']
                
                # Diversity/Fairness/Freshness terms
                diversity = self.metrics.compute_diversity(temp_list)
                fairness = self.metrics.compute_fairness(temp_list)
                freshness = self.metrics.compute_freshness(temp_list)
                
                total_score = (self.alpha * accuracy + 
                               self.beta * diversity + 
                               self.gamma * fairness + 
                               self.delta * freshness)
                
                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx
            
            if best_idx != -1:
                selected.append(remaining.pop(best_idx))
            else:
                break
                
        return pl.DataFrame(selected).lazy()
