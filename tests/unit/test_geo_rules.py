import pytest
import polars as pl
from src.rules.geo_rules import GeoProximityScoreRule
from src.core.base import RecommendationContext

def test_geo_proximity_score_rule():
    # Mock similarity dataframe
    similarity_df = pl.DataFrame({
        "district_a": ["D1", "D1", "D2", "D2"],
        "district_b": ["D1", "D2", "D1", "D2"],
        "jaccard_similarity": [1.0, 0.4, 0.4, 1.0]
    })
    
    rule = GeoProximityScoreRule(district_similarity_df=similarity_df)
    
    # Mock data
    candidates = pl.LazyFrame({
        "user_id": ["u1", "u1", "u1"],
        "item_id": ["i1", "i2", "i3"],
        "district_name": ["D1", "D2", "D3"],
        "user_viewed_districts": [
            [{"district": "D1", "weight": 10.0}],
            [{"district": "D1", "weight": 10.0}],
            [{"district": "D1", "weight": 10.0}]
        ]
    })
    
    res = rule.apply(candidates).collect()
    
    assert "geo_score" in res.columns
    scores = res.select(["item_id", "geo_score"]).to_dicts()
    score_dict = {row["item_id"]: row["geo_score"] for row in scores}
    
    # i1: D1 matched with D1 (jaccard=1.0)
    assert abs(score_dict["i1"] - 1.0) < 1e-6
    # i2: D2 matched with D1 (jaccard=0.4)
    assert abs(score_dict["i2"] - 0.4) < 1e-6
    # i3: D3 matched with D1 (jaccard=0.05 default because missing in matrix)
    assert abs(score_dict["i3"] - 0.05) < 1e-6
    
    # Test build_similarity_matrix
    fact_user_events = pl.LazyFrame({
        "user_id": ["u1", "u1", "u2"],
        "session_id": ["s1", "s1", "s2"],
        "item_id": ["i1", "i2", "i1"]
    })
    dim_listing = pl.LazyFrame({
        "item_id": ["i1", "i2"],
        "district_name": ["D1", "D2"]
    })
    
    sim_matrix = GeoProximityScoreRule.build_similarity_matrix(fact_user_events, dim_listing)
    assert sim_matrix.height > 0
    assert "jaccard_similarity" in sim_matrix.columns

