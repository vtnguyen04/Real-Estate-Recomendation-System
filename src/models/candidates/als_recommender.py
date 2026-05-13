import numpy as np
from typing import List, Optional
from datetime import datetime
import implicit
import polars as pl
from scipy.sparse import csr_matrix

from src.core.base import BaseRecommender, RecommendationContext, Recommendation
from src.features.interaction_matrix import InteractionMatrixBuilder

class ALSRecommender(BaseRecommender):
    """
    Alternating Least Squares (ALS) Collaborative Filtering Recommender.
    Highly optimized implicit matrix factorization for generating personalized candidates.
    Uses InteractionMatrixBuilder to correctly weight business events.
    """
    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 15):
        super().__init__(name="als_recommender")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        
        # use_gpu=False guarantees it won't crash on standard Colab setups if CUDA isn't compiled
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors, 
            regularization=self.regularization, 
            iterations=self.iterations,
            use_gpu=False 
        )
        self.matrix_builder = None
        self.user_items = None # CSR Matrix (Users x Items)

    def fit(self, train_data: pl.LazyFrame, **kwargs) -> 'BaseRecommender':
        """
        Trains the ALS model directly from raw event logs.
        Args:
            train_data: The raw event log (fact_user_events) as a LazyFrame.
            kwargs:
                current_date: datetime object for decay calculation.
                half_life_days: float for temporal decay.
        """
        current_date = kwargs.get('current_date', datetime.now())
        half_life_days = kwargs.get('half_life_days', 14.0)
        
        self.matrix_builder = InteractionMatrixBuilder(half_life_days=half_life_days)
        
        # Build CSR Matrix lazily before collecting to RAM
        self.user_items = self.matrix_builder.build(train_data, current_date=current_date)
        
        # Train ALS model (implicit >= 0.6.0 accepts users x items directly)
        self.model.fit(self.user_items)
        
        return self

    def recommend(
        self,
        context: RecommendationContext,
        candidates: Optional[pl.LazyFrame] = None
    ) -> List[Recommendation]:
        
        if not self.matrix_builder:
            return []
            
        user_idx = self.matrix_builder.get_user_idx(context.user_id)
        if user_idx == -1:
            # User not in training set -> Pure Cold Start
            return []
            
        # Get top K recommendations for the user
        ids, scores = self.model.recommend(
            user_idx, 
            self.user_items[user_idx], 
            N=context.num_recommendations, 
            filter_already_liked_items=True
        )
        
        recs = []
        for i, (item_idx, score) in enumerate(zip(ids, scores)):
            recs.append(Recommendation(
                item_id=self.matrix_builder.get_item_id(item_idx),
                score=float(score),
                rank=i + 1,
                explanation="Personalized collaborative filtering (ALS)"
            ))
        return recs

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> 'BaseRecommender':
        self.model = implicit.als.AlternatingLeastSquares.load(path)
        return self
