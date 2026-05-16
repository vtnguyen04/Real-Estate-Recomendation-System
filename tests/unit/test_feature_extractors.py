import pytest
import polars as pl
from src.features.user_features import UserBehaviorExtractor
from src.features.item_features import ItemPopularityExtractor

def test_user_behavior_extractor():
    extractor = UserBehaviorExtractor()
    
    data = pl.LazyFrame({
        "user_id": ["u1", "u1", "u2"],
        "event_type": ["view", "contact_chat", "view"],
        "dwell_time_sec": [10.0, 50.0, 5.0],
        "category": [1010, 1010, 1020]
    })
    
    # BaseFeatureExtractor usually implements fit_transform or extract
    # We call extract method
    res = extractor.extract(data).collect()
    
    assert "total_events" in res.columns
    assert "positive_interaction_rate" in res.columns
    
    u1_res = res.filter(pl.col("user_id") == "u1")
    assert u1_res["total_events"][0] == 2
    assert u1_res["total_positive_interactions"][0] == 1 # contact_chat
    assert u1_res["avg_dwell_time_sec"][0] == 30.0
    assert u1_res["positive_interaction_rate"][0] == 0.5

def test_item_popularity_extractor_snapshot():
    extractor = ItemPopularityExtractor()
    
    # Snapshot data
    data1 = pl.LazyFrame({
        "item_id": ["i1", "i1", "i2"],
        "views_24h": [10, 20, 5],
        "contacts_24h": [1, 2, 0],
        "listing_age_days": [2.0, 3.0, 10.0]
    })
    
    res1 = extractor.extract(data1).collect()
    
    assert "contact_conversion_rate" in res1.columns
    i1_res = res1.filter(pl.col("item_id") == "i1")
    assert i1_res["total_views"][0] == 30
    assert i1_res["total_contacts"][0] == 3
    assert i1_res["current_age_days"][0] == 3.0
    
def test_item_popularity_extractor_interactions():
    extractor = ItemPopularityExtractor()
    
    # Interactions data
    data2 = pl.LazyFrame({
        "item_id": ["i1", "i2"],
        "adview_count": [100, 50],
        "lead_count": [10, 0],
        "chat_message_count": [5, 0]
    })
    
    res2 = extractor.extract(data2).collect()
    
    assert "lead_conversion_rate" in res2.columns
    i1_res = res2.filter(pl.col("item_id") == "i1")
    assert i1_res["total_leads"][0] == 10
    assert i1_res["total_adviews"][0] == 100
    
def test_item_popularity_extractor_fallback():
    extractor = ItemPopularityExtractor()
    
    # Fallback data
    data3 = pl.LazyFrame({
        "item_id": ["i1"],
        "other_col": [1]
    })
    
    res3 = extractor.extract(data3).collect()
    assert "other_col" in res3.columns
