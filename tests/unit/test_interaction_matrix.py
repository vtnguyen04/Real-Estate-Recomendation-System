import pytest
import polars as pl
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
from src.features.interaction_matrix import InteractionMatrixBuilder

def test_interaction_matrix_building():
    """Ensure CSR matrix handles weights, decay, and unique mappings correctly."""
    now = datetime(2026, 4, 1, 10, 0, 0)
    data = [
        # u1 interacts with i1 multiple times recently
        {"user_id": "u1", "item_id": "i1", "event_type": "pageview", "timestamp": now}, 
        {"user_id": "u1", "item_id": "i1", "event_type": "view_phone", "timestamp": now},
        
        # u2 interacts with i2, but 14 days ago (half-life)
        {"user_id": "u2", "item_id": "i2", "event_type": "pageview", "timestamp": now - timedelta(days=14)},
        
        # u3 interacts with i1 with an extreme intent (lead)
        {"user_id": "u3", "item_id": "i1", "event_type": "lead", "timestamp": now},
    ]
    df = pl.LazyFrame(data)
    
    builder = InteractionMatrixBuilder(half_life_days=14.0)
    matrix = builder.build(df, current_date=now)
    
    assert isinstance(matrix, csr_matrix)
    # 3 unique users, 2 unique items
    assert matrix.shape == (3, 2)
    
    # Check integer mappings
    assert builder.get_user_idx("u1") != -1
    assert builder.get_user_idx("u2") != -1
    assert builder.get_user_idx("u3") != -1
    
    u1_idx = builder.get_user_idx("u1")
    i1_idx = builder.get_item_idx("i1")
    u2_idx = builder.get_user_idx("u2")
    i2_idx = builder.get_item_idx("i2")
    u3_idx = builder.get_user_idx("u3")
    
    u1_score = matrix[u1_idx, i1_idx]
    u2_score = matrix[u2_idx, i2_idx]
    u3_score = matrix[u3_idx, i1_idx]
    
    # Assert Rules
    # u1: pageview(1) + view_phone(5) = 6.0 (age = 0, so no decay)
    assert pytest.approx(u1_score, 0.01) == 6.0
    
    # u2: pageview(1) but 14 days old (half_life=14), so score should be 0.5 * 1 = 0.5
    assert pytest.approx(u2_score, 0.01) == 0.5
    
    # u3: lead(10.0) -> high weight
    assert pytest.approx(u3_score, 0.01) == 10.0
