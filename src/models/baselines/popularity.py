import polars as pl
from typing import List, Optional
from src.core.base import BaseRecommender, RecommendationContext, Recommendation

class PopularityRecommender(BaseRecommender):
    """
    Recommends the top-K items based on recent popularity (views/contacts).
    Serves as the primary fallback for cold-start users (users with no interaction history).
    """
    def __init__(self, top_k: int = 100):
        super().__init__(name="popularity_recommender")
        self.top_k = top_k
        self.popular_items = []

    def fit(self, train_data: pl.LazyFrame, **kwargs) -> 'BaseRecommender':
        """
        Learns the most popular items globally based on contacts and views.
        Args:
            train_data: A LazyFrame representing item snapshots (e.g., fact_listing_snapshot)
        """
        if isinstance(train_data, pl.DataFrame):
            train_data = train_data.lazy()
            
        schema = train_data.collect_schema().names()
        if "item_id" not in schema:
            return self

        # Heuristic: Heavily weight recent contacts (strong intent), fallback to views
        if "contacts_24h" in schema and "views_24h" in schema:
            popular = train_data.with_columns([
                (pl.col("contacts_24h") * 10 + pl.col("views_24h")).alias("pop_score")
            ]).sort("pop_score", descending=True).select(["item_id", "pop_score"]).head(self.top_k)
        elif "views_24h" in schema:
            popular = train_data.with_columns(
                pl.col("views_24h").alias("pop_score")
            ).sort("pop_score", descending=True).select(["item_id", "pop_score"]).head(self.top_k)
        else:
            # Fallback if no performance data is present
            popular = train_data.with_columns(pl.lit(1.0).alias("pop_score")).head(self.top_k).select(["item_id", "pop_score"])
            
        # Materialize
        collected = popular.collect()
        self.popular_items = list(zip(collected["item_id"].to_list(), collected["pop_score"].to_list()))
        
        return self

    def recommend(
        self,
        context: RecommendationContext,
        candidates: Optional[pl.LazyFrame] = None
    ) -> pl.LazyFrame:
        
        k = context.num_recommendations
        
        # self.popular_items is list of (item_id, score)
        recs = [
            {"user_id": context.user_id, "item_id": item_id, "score": float(score)}
            for item_id, score in self.popular_items[:k]
        ]
        
        return pl.DataFrame(recs).lazy()

    def save(self, path: str) -> None:
        import json
        with open(path, 'w') as f:
            json.dump(self.popular_items, f)

    def load(self, path: str) -> 'BaseRecommender':
        import json
        with open(path, 'r') as f:
            self.popular_items = json.load(f)
        return self
