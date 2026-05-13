import pytest
import polars as pl
from src.models.rerankers.multi_objective import MultiObjectiveReranker

def test_multi_objective_reranker():
    reranker = MultiObjectiveReranker(alpha=0.6, beta=0.2, gamma=0.2, delta=0.0)
    
    candidates = pl.LazyFrame({
        "item_id": ["i1", "i2", "i3", "i4"],
        "score": [0.9, 0.8, 0.7, 0.6],
        "category": [1, 2, 1, 3],
        "seller_id": ["s1", "s2", "s1", "s3"],
        "listing_age_days": [2, 10, 1, 5]
    })
    
    # Rerank to top 3
    final = reranker.rerank(candidates, k=3).collect()
    
    assert final.height == 3
    assert "item_id" in final.columns
    
    # We shouldn't drop items if we need 3, it should just pick the 3 best
    # based on diversity and fairness.
    # It shouldn't crash if columns are missing either
    candidates_missing = pl.LazyFrame({
        "item_id": ["i1", "i2"],
        "score": [0.9, 0.8]
    })
    final2 = reranker.rerank(candidates_missing, k=2).collect()
    assert final2.height == 2

    # Test coverage for edge cases
    empty = pl.LazyFrame({"item_id": [], "score": []})
    final3 = reranker.rerank(empty, k=2).collect()
    assert final3.height == 0
