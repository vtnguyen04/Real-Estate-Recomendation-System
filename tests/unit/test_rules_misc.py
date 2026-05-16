import pytest
import polars as pl
from src.rules.match_rules import MatchScoreRule
from src.rules.urgency_rules import UrgencyScoreRule

def test_match_score_rule():
    rule = MatchScoreRule()
    
    # Needs user_top_categories and category to match
    candidates = pl.LazyFrame({
        "item_id": ["i1", "i2", "i3"],
        "category": [1010, 1020, 1010],
        "user_top_categories": [[1010], [1010], [1010]],
        "item_price_pct": [0.5, 0.9, 0.4],
        "user_price_min": [0.3, 0.3, 0.3],
        "user_price_max": [0.6, 0.6, 0.6],
        "city_name": ["Hanoi", "HCMC", "Da Nang"],
        "user_top_city": ["Hanoi", "Hanoi", "Hanoi"],
        "bedrooms": [2.0, 3.0, 2.0],
        "user_modal_bedrooms": [2.0, 2.0, 1.0]
    })
    
    from src.core.base import RecommendationContext
    context = RecommendationContext(user_id="u1")
    
    # Call apply directly
    result = rule.apply(candidates, context)
    df = result.collect()
    
    assert "match_score" in df.columns
    # i1 matches 1010, price range, city, and bedrooms -> should be highest
    i1_score = df.filter(pl.col("item_id") == "i1")["match_score"][0]
    i2_score = df.filter(pl.col("item_id") == "i2")["match_score"][0]
    assert i1_score > i2_score

def test_urgency_score_rule():
    rule = UrgencyScoreRule()
    
    candidates = pl.LazyFrame({
        "item_id": ["i1", "i2"],
        "listing_age_days": [2.0, 10.0],
        "recent_3d_views": [100.0, 10.0],
        "older_3d_views": [50.0, 20.0],
        "has_price_drop": [True, False]
    })
    
    from src.core.base import RecommendationContext
    context = RecommendationContext(user_id="u1")
    
    result = rule.apply(candidates, context)
    df = result.collect()
    
    assert "urgency_score" in df.columns
    # i1 is younger, has increasing views, and dropped price -> highest urgency
    i1_score = df.filter(pl.col("item_id") == "i1")["urgency_score"][0]
    i2_score = df.filter(pl.col("item_id") == "i2")["urgency_score"][0]
    
    assert i1_score > i2_score
