import pytest
import polars as pl
from datetime import datetime
from src.models.collaborative import ALSRecommender
from src.core.base import RecommendationContext

def test_als_recommender():
    """Ensure implicit ALS model builds correctly and handles cold-starts cleanly."""
    now = datetime(2026, 4, 1)
    
    # Simple synthetic graph:
    # u1 likes i1
    # u2 likes i1 and i3
    # CF logic -> u1 should be recommended i3
    data = pl.LazyFrame([
        {"user_id": "u1", "item_id": "i1", "event_type": "view_phone", "timestamp": now},
        {"user_id": "u2", "item_id": "i1", "event_type": "contact_chat", "timestamp": now},
        {"user_id": "u2", "item_id": "i3", "event_type": "view_phone", "timestamp": now},
    ])
    
    # Tiny factors for fast unit testing
    model = ALSRecommender(factors=4, iterations=5)
    model.fit(data, current_date=now)
    
    # 1. Warm User Test
    ctx_u1 = RecommendationContext(user_id="u1", timestamp="2026-04-01", num_recommendations=1)
    recs_u1 = model.recommend(ctx_u1)
    
    assert len(recs_u1) == 1
    # CF should recommend i3, because u1 already liked i1 (which is filtered out)
    assert recs_u1[0].item_id == "i3"
    assert recs_u1[0].explanation == "Personalized collaborative filtering (ALS)"
    
    # 2. Pure Cold-Start User Test
    ctx_u_cold = RecommendationContext(user_id="u_cold", timestamp="2026-04-01", num_recommendations=5)
    recs_cold = model.recommend(ctx_u_cold)
    
    # ALS natively returns empty array for completely unseen users
    # This proves we need the HybridOrchestrator to fallback to PopularityRecommender
    assert len(recs_cold) == 0
