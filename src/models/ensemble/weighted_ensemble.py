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
    ) -> pl.LazyFrame:
        
        all_recs = []
        for idx, model in enumerate(self.models):
            recs_df = model.recommend(context, candidates).collect()
            if recs_df.is_empty():
                continue
                
            weight = self.weights[idx]
            
            if self.normalize and recs_df.height > 1:
                min_s = recs_df["score"].min()
                max_s = recs_df["score"].max()
                denom = max_s - min_s if max_s > min_s else 1.0
                recs_df = recs_df.with_columns(
                    ((pl.col("score") - min_s) / denom).alias("score")
                )
            
            recs_df = recs_df.with_columns(
                (pl.col("score") * weight).alias("score")
            )
            all_recs.append(recs_df)
            
        if not all_recs:
            return pl.DataFrame([]).lazy()
            
        # Merge all dataframes and sum scores by item_id
        combined = pl.concat(all_recs)
        final = combined.group_by(["user_id", "item_id"]).agg(
            pl.col("score").sum()
        ).sort("score", descending=True).head(context.num_recommendations)
        
        return final.lazy()

    def save(self, path: str) -> None:
        raise NotImplementedError("Ensemble save strategy requires saving all sub-models")

    def load(self, path: str) -> 'BaseRecommender':
        raise NotImplementedError("Ensemble load strategy requires loading all sub-models")
