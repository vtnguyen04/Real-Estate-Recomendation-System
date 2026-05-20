import implicit
from typing import Dict, Any, Optional
from src.core.base import BaseRecommender
from src.models.candidates.implicit_base import ImplicitBaseRecommender

class BPRRecommender(ImplicitBaseRecommender):
    """
    Bayesian Personalized Ranking (BPR) Collaborative Filtering Recommender.
    Highly optimized pairwise implicit matrix factorization for generating personalized candidates.
    Matches directly with SOTA implicit ranking optimization.
    """
    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 15, use_gpu: bool = False, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="bpr_recommender", factors=factors, regularization=regularization, iterations=iterations, use_gpu=use_gpu, config=config)
        
        # Enable GPU acceleration if configured
        try:
            self.model = implicit.bpr.BayesianPersonalizedRanking(
                factors=self.factors, 
                regularization=self.regularization, 
                iterations=self.iterations,
                use_gpu=self.use_gpu 
            )
        except ValueError:
            # Fallback to CPU if CUDA extension fails
            self.use_gpu = False
            self.model = implicit.bpr.BayesianPersonalizedRanking(
                factors=self.factors, 
                regularization=self.regularization, 
                iterations=self.iterations,
                use_gpu=False 
            )

    def load(self, path: str) -> 'BaseRecommender':
        import os, pickle
        from implicit.cpu.bpr import BayesianPersonalizedRanking
        
        filepath = path
        if os.path.isdir(path):
            filepath = os.path.join(path, f"{self.name}.npz")
            
        self.model = BayesianPersonalizedRanking.load(filepath)
        meta_path = filepath.replace(".npz", "_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.matrix_builder = pickle.load(f)
        return self
