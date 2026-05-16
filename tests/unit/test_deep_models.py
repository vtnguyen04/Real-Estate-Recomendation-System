import pytest
import polars as pl
import numpy as np
import os

from src.models.deep.session_gru import SessionBasedRecommender, SessionGRU, HAS_TORCH
from src.models.deep.graph_sage import GraphBasedRecommender, HAS_PYG
from src.core.base import RecommendationContext

@pytest.fixture
def sample_events():
    return pl.DataFrame({
        "user_id": ["u1", "u1", "u2", "u3", "u1"],
        "item_id": ["i1", "i2", "i2", "i3", "i3"],
        "timestamp": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-01", "2026-01-03"],
        "dwell_time_sec": [10.0, 20.0, 5.0, 15.0, 30.0]
    }).lazy()

def test_session_gru(sample_events, tmp_path):
    model = SessionBasedRecommender(config={"embedding_dim": 8, "hidden_dim": 8, "max_seq_len": 5})
    model.fit(sample_events)
    
    # Check internal state
    assert len(model.user_histories) == 3
    assert "u1" in model.user_histories
    
    # Test recommendation
    context = RecommendationContext(user_id="u1", num_recommendations=2)
    recs = model.recommend(context).collect()
    
    assert recs.height == 2
    assert "score" in recs.columns
    
    # Test save/load
    model.save(str(tmp_path / "gru"))
    assert (tmp_path / "gru" / "session_gru.pth").exists()
    
    new_model = SessionBasedRecommender(config={"embedding_dim": 8, "hidden_dim": 8})
    new_model.load(str(tmp_path / "gru"))
    assert new_model.item_to_idx == model.item_to_idx

def test_graph_sage(sample_events, tmp_path):
    model = GraphBasedRecommender(config={"in_channels": 8, "hidden_channels": 8})
    model.fit(sample_events)
    
    # Check internal state
    assert model.user_embeddings is not None
    assert model.item_embeddings is not None
    
    # Test recommendation
    context = RecommendationContext(user_id="u1", num_recommendations=2)
    recs = model.recommend(context).collect()
    
    assert recs.height == 2
    assert "score" in recs.columns
    
    # Test save/load
    model.save(str(tmp_path / "graph"))
    assert (tmp_path / "graph" / "graph_sage.pth").exists()
    
    new_model = GraphBasedRecommender(config={"in_channels": 8, "hidden_channels": 8})
    new_model.load(str(tmp_path / "graph"))
    assert new_model.user_to_idx == model.user_to_idx
