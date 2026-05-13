import polars as pl
from typing import List, Optional
from collections import defaultdict
from src.core.base import BaseRecommender, RecommendationContext, Recommendation

class WeightedEnsembleRecommender(BaseRecommender):
    """
    Combines recommendations from multiple candidate models.
    Supports min-max normalization and weight blending.
    Automatically acts as a fallback handler for Cold-Start.
    """
    def __init__(self, models: List[BaseRecommender], weights: List[float], normalize: bool = True):
        super().__init__(name="weighted_ensemble")
        if len(models) != len(weights):
            raise ValueError("Number of models must match number of weights")
        self.models = models
        self.weights = weights
        self.normalize = normalize

    def fit(self, train_data: pl.LazyFrame, **kwargs) -> 'BaseRecommender':
        """Delegates training to all sub-models"""
        for model in self.models:
            model.fit(train_data, **kwargs)
        return self

    def recommend(
        self,
        context: RecommendationContext,
        candidates: Optional[pl.LazyFrame] = None
    ) -> List[Recommendation]:
        
        item_scores = defaultdict(float)
        item_explanations = defaultdict(list)
        
        all_recs_lists = []
        for model in self.models:
            # We explicitly ask each sub-model for context.num_recommendations 
            # to ensure we have a sufficient pool to blend.
            recs = model.recommend(context, candidates)
            all_recs_lists.append(recs)
            
        for idx, recs in enumerate(all_recs_lists):
            if not recs:
                continue
                
            weight = self.weights[idx]
            
            # Normalize scores to [0, 1] if requested to balance disparate score distributions
            if self.normalize and len(recs) > 1:
                raw_scores = [r.score for r in recs]
                min_score = min(raw_scores)
                max_score = max(raw_scores)
                denom = max_score - min_score
            else:
                denom = 0.0
                
            for rec in recs:
                if denom > 0.0:
                    norm_score = (rec.score - min_score) / denom
                else:
                    norm_score = 1.0 if rec.score > 0 else 0.0
                    
                item_scores[rec.item_id] += norm_score * weight
                item_explanations[rec.item_id].append(f"{self.models[idx].name}")
                
        # If no items were recommended by any model
        if not item_scores:
            return []
            
        # Sort by blended score
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_recs = []
        for rank, (item_id, score) in enumerate(sorted_items[:context.num_recommendations]):
            explanation = "Ensemble: " + " + ".join(item_explanations[item_id])
            final_recs.append(Recommendation(
                item_id=item_id,
                score=float(score),
                rank=rank + 1,
                explanation=explanation
            ))
            
        return final_recs

    def save(self, path: str) -> None:
        raise NotImplementedError("Ensemble save strategy requires saving all sub-models")

    def load(self, path: str) -> 'BaseRecommender':
        raise NotImplementedError("Ensemble load strategy requires loading all sub-models")
