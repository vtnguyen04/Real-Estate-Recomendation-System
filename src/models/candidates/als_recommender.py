import implicit
from typing import Dict, Any, Optional
from src.core.base import BaseRecommender
from src.models.candidates.implicit_base import ImplicitBaseRecommender

class ALSRecommender(ImplicitBaseRecommender):
    """
    Alternating Least Squares (ALS) Collaborative Filtering Recommender.
    Highly optimized implicit matrix factorization for generating personalized candidates.
    Uses InteractionMatrixBuilder to correctly weight business events.
    """
    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 15, use_gpu: bool = False, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="als_recommender", factors=factors, regularization=regularization, iterations=iterations, use_gpu=use_gpu, config=config)
        
        # Enable GPU acceleration if configured
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors, 
            regularization=self.regularization, 
            iterations=self.iterations,
            use_gpu=self.use_gpu 
        )

    def load(self, path: str) -> 'BaseRecommender':
        self.model = implicit.als.AlternatingLeastSquares.load(path)
        return self
