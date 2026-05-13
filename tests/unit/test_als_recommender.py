import pytest
import polars as pl
from datetime import datetime
from src.models.candidates.als_recommender import ALSRecommender
from src.core.base import RecommendationContext

def test_als_recommender_fit_predict(tmp_path):
    recommender = ALSRecommender(factors=2, iterations=1)
    
    # Mock train data
    train_data = pl.LazyFrame({
        "user_id": ["u1", "u1", "u2"],
        "item_id": ["i1", "i2", "i3"],
        "event_type": ["view", "contact_chat", "view"],
        "timestamp": ["2026-05-10T10:00:00", "2026-05-11T10:00:00", "2026-05-12T10:00:00"],
        "dwell_time_sec": [10.0, 50.0, 10.0]
    }).with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S"))
    
    # fit
    recommender.fit(train_data, current_date=datetime(2026, 5, 12, 10, 0, 0))
    
    assert recommender.matrix_builder is not None
    assert recommender.user_items is not None
    
    # test inference
    context = RecommendationContext(user_id="u1", num_recommendations=2)
    res = recommender.recommend(context).collect()
    assert res.height > 0
    assert "score" in res.columns
    
    # test cold start user
    context_cold = RecommendationContext(user_id="u99", num_recommendations=2)
    res_cold = recommender.recommend(context_cold).collect()
    assert res_cold.height == 0
    
    # save / load
    save_path = str(tmp_path / "als_model.npz")
    recommender.save(save_path)
    
    rec2 = ALSRecommender()
    rec2.load(save_path)
    assert rec2.model is not None
