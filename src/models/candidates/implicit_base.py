import numpy as np
from typing import List, Optional, Any, Dict
from datetime import datetime
import implicit
import polars as pl

from src.core.base import BaseRecommender, RecommendationContext
from src.features.interaction_matrix import InteractionMatrixBuilder

class ImplicitBaseRecommender(BaseRecommender):
    """
    Base class for Recommenders using the implicit matrix factorization library.
    Handles the boilerplate of fitting, recommending, and matrix building, adhering to DRY.
    """
    def __init__(self, name: str, factors: int = 64, regularization: float = 0.01, iterations: int = 15, use_gpu: bool = False, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.use_gpu = use_gpu
        self.config = config or {}
        self.model: Any = None # Must be instantiated by child class
        self.matrix_builder = None
        self.user_items = None # CSR Matrix (Users x Items)

    def fit(self, train_data: pl.LazyFrame, **kwargs) -> 'BaseRecommender':
        """
        Trains the implicit model directly from raw event logs.
        """
        current_date = kwargs.get('current_date', datetime.now())
        half_life_days = kwargs.get('half_life_days', 7.0) # INS-021: 7-day Golden Moment
        
        self.matrix_builder = InteractionMatrixBuilder(half_life_days=half_life_days)
        
        # Build CSR Matrix lazily before collecting to RAM
        self.user_items = self.matrix_builder.build(train_data, current_date=current_date)
        
        from src.utils.logging import get_logger
        logger = get_logger(__name__)
        logger.info(f"Implicit Matrix Shape: {self.user_items.shape}, NNZ: {self.user_items.nnz}")
        
        # Train model (implicit >= 0.6.0 accepts users x items directly)
        self.model.fit(self.user_items)
        
        return self

    def recommend(self, context: RecommendationContext, candidates: Optional[pl.LazyFrame] = None) -> pl.LazyFrame:
        if not self.matrix_builder:
            return pl.DataFrame([]).lazy()
            
        user_idx = self.matrix_builder.get_user_idx(context.user_id)
        if user_idx == -1:
            # User not in training set -> Pure Cold Start
            return pl.DataFrame([]).lazy()
            
        # Get top K recommendations for the user
        ids, scores = self.model.recommend(
            user_idx, 
            self.user_items[user_idx], 
            N=context.num_recommendations, 
            filter_already_liked_items=True
        )
        
        recs = [
            {
                "user_id": context.user_id,
                "item_id": self.matrix_builder.get_item_id(item_idx),
                "score": float(score)
            }
            for item_idx, score in zip(ids, scores)
        ]
        return pl.DataFrame(recs).lazy()

    def save(self, path: str) -> None:
        import os, pickle
        if os.path.isdir(path):
            path = os.path.join(path, f"{self.name}.npz")
        self.model.save(path)
        with open(path.replace(".npz", "_meta.pkl"), "wb") as f:
            pickle.dump(self.matrix_builder, f)
