import pytest
import polars as pl
from typing import List, Optional
from src.core.base import BaseRecommender, RecommendationContext, Recommendation
from src.models.ensemble.weighted_ensemble import WeightedEnsembleRecommender

class MockALS(BaseRecommender):
    def __init__(self):
        super().__init__(name="mock_als")
        
    def fit(self, train_data, **kwargs): return self
    
    def recommend(self, context, candidates=None):
        if context.user_id == "u_cold":
            return []  # Cold-Start returns nothing
        return [
            Recommendation(item_id="i1", score=10.0, rank=1, explanation="ALS"),
            Recommendation(item_id="i2", score=5.0, rank=2, explanation="ALS"),
        ]
        
    def save(self, path): pass
    def load(self, path): return self

class MockPop(BaseRecommender):
    def __init__(self):
        super().__init__(name="mock_pop")
        
    def fit(self, train_data, **kwargs): return self
    
    def recommend(self, context, candidates=None):
        return [
            Recommendation(item_id="i3", score=100.0, rank=1, explanation="Pop"),
            Recommendation(item_id="i2", score=50.0, rank=2, explanation="Pop"),
        ]
        
    def save(self, path): pass
    def load(self, path): return self

def test_weighted_ensemble_warm_user():
    """Ensure ensemble correctly blends and normalizes scores."""
    als = MockALS()
    pop = MockPop()
    ensemble = WeightedEnsembleRecommender(
        models=[als, pop],
        weights=[0.7, 0.3], # ALS is prioritized
        normalize=True
    )
    
    ctx = RecommendationContext(user_id="u_warm", timestamp="2026-04-09", num_recommendations=3)
    recs = ensemble.recommend(ctx)
    
    # Expected normalization:
    # ALS max=10, min=5 -> denom=5 -> i1 norm=1.0, i2 norm=0.0
    # Pop max=100, min=50 -> denom=50 -> i3 norm=1.0, i2 norm=0.0
    # Weighted sums:
    # i1 = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
    # i3 = 0.7 * 0.0 + 0.3 * 1.0 = 0.3
    # i2 = 0.7 * 0.0 + 0.3 * 0.0 = 0.0
    
    assert len(recs) == 3
    assert recs[0].item_id == "i1"
    assert recs[1].item_id == "i3"
    assert recs[2].item_id == "i2"

def test_weighted_ensemble_cold_start_fallback():
    """Ensure ensemble correctly falls back to Popularity if ALS returns empty."""
    als = MockALS()
    pop = MockPop()
    ensemble = WeightedEnsembleRecommender(
        models=[als, pop],
        weights=[0.7, 0.3],
        normalize=True
    )
    
    ctx = RecommendationContext(user_id="u_cold", timestamp="2026-04-09", num_recommendations=2)
    recs = ensemble.recommend(ctx)
    
    # ALS returns []
    # Pop returns [i3, i2]
    # Fallback inherently kicks in, filling the void
    assert len(recs) == 2
    assert recs[0].item_id == "i3"
    assert recs[1].item_id == "i2"
    assert "mock_pop" in recs[0].explanation
