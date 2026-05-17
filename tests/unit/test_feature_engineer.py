import pytest
import polars as pl
from src.features.feature_engineer import FeatureEngineer
from src.core.base import BaseRule, RecommendationContext

class MockRule(BaseRule):
    def __init__(self):
        super().__init__(name="mock_rule", is_hard_filter=False)
        self.priority = 10
        
    def apply(self, items: pl.LazyFrame, context: RecommendationContext = None) -> pl.LazyFrame:
        return items.with_columns(pl.lit(99.0).alias("mock_score"))

def test_feature_engineer_aggregations():
    """Ensure FeatureEngineer extracts cold-start, session ratios, and history checks."""
    candidate_items = pl.LazyFrame([
        {"user_id": "u1", "item_id": "i1"},
        {"user_id": "u1", "item_id": "i2"},
        {"user_id": "u2", "item_id": "i1"}
    ])
    
    user_profile = pl.LazyFrame([
        {"user_id": "u1", "user_total_views": 10},
        {"user_id": "u2", "user_total_views": 2} # cold start
    ])
    
    item_profile = pl.LazyFrame([
        {"item_id": "i1", "price_vnd": 100.0, "category": 1, "district_name": "D1"},
        {"item_id": "i2", "price_vnd": 200.0, "category": 1, "district_name": "D1"}
    ])
    
    # u1 viewed i1 in session s1
    interactions = pl.LazyFrame([
        {"user_id": "u1", "item_id": "i1", "session_id": "s1", "event_type": "pageview"}
    ])
    
    session_embeddings = pl.LazyFrame([
        {"user_id": "u1", "session_emb_0": 0.5}
    ])
    
    graph_embeddings = pl.LazyFrame([
        {"item_id": "i1", "graph_emb_0": 0.1}
    ])
    
    fe = FeatureEngineer(deterministic_rules=[MockRule()])
    
    df = fe.engineer_features(
        candidate_items=candidate_items,
        user_profile=user_profile,
        item_profile=item_profile,
        interactions=interactions,
        session_embeddings=session_embeddings,
        graph_embeddings=graph_embeddings
    ).collect()
    
    # 1. Rule Application
    assert "mock_score" in df.columns
    assert df["mock_score"][0] == 99.0
    
    # 2. Cold-Start Logic
    u1_cold = df.filter((pl.col("user_id") == "u1") & (pl.col("item_id") == "i1"))["user_is_cold_start"][0]
    u2_cold = df.filter((pl.col("user_id") == "u2"))["user_is_cold_start"][0]
    assert u1_cold == 0  # 10 views > 5
    assert u2_cold == 1  # 2 views < 5
    
    # 3. Interaction History Check
    u1_i1_viewed = df.filter((pl.col("user_id") == "u1") & (pl.col("item_id") == "i1"))["user_viewed_this_item"][0]
    u1_i2_viewed = df.filter((pl.col("user_id") == "u1") & (pl.col("item_id") == "i2"))["user_viewed_this_item"][0]
    assert u1_i1_viewed == 1  # u1 viewed i1
    assert u1_i2_viewed == 0  # u1 never viewed i2
    
    # 4. Session Price Ratio Logic
    assert "session_price_ratio" in df.columns
    u1_i1_ratio = df.filter((pl.col("user_id") == "u1") & (pl.col("item_id") == "i1"))["session_price_ratio"][0]
    u1_i2_ratio = df.filter((pl.col("user_id") == "u1") & (pl.col("item_id") == "i2"))["session_price_ratio"][0]
    
    # session_avg_price for u1 = 100.0 (they only viewed i1)
    # ratio = price / (avg_price + 1.0)
    assert pytest.approx(u1_i1_ratio, 0.01) == 100.0 / 101.0
    assert pytest.approx(u1_i2_ratio, 0.01) == 200.0 / 101.0
