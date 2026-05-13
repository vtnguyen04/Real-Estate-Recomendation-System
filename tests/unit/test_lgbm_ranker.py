import pytest
import polars as pl
import numpy as np
from src.models.rankers.lgbm_ranker import MultiTaskLGBMRanker
from src.core.base import RecommendationContext

def test_lgbm_ranker_fit_predict(tmp_path):
    ranker = MultiTaskLGBMRanker(config={'rounds_bin': 2, 'rounds_multi': 2})
    
    # Mock training data
    train_data = pl.LazyFrame({
        "user_id": ["u1", "u1", "u2", "u2"],
        "item_id": ["i1", "i2", "i3", "i4"],
        "feature_1": [0.1, 0.5, 0.9, 0.2],
        "feature_2": [1, 2, 3, 4],
        "label_binary": [1, 0, 1, 0],
        "label_multiclass": [1, 0, 2, 0],
        "group_id": [1, 1, 2, 2]
    })
    
    ranker.fit(train_data)
    
    assert ranker.model_binary is not None
    assert ranker.model_multiclass is not None
    assert "feature_1" in ranker.feature_cols
    assert "feature_2" in ranker.feature_cols
    
    # Mock inference
    context = RecommendationContext(user_id="u1", num_recommendations=2)
    candidates = pl.LazyFrame({
        "user_id": ["u1", "u1"],
        "item_id": ["i1", "i2"],
        "feature_1": [0.3, 0.6],
        "feature_2": [2, 3]
    })
    
    res = ranker.recommend(context, candidates)
    res_df = res.collect()
    
    assert "score" in res_df.columns
    assert res_df.height == 2
    
    # Test save/load
    ranker.save(str(tmp_path))
    
    ranker2 = MultiTaskLGBMRanker()
    ranker2.load(str(tmp_path))
    
    assert ranker2.model_binary is not None
    assert ranker2.feature_cols == ranker.feature_cols
