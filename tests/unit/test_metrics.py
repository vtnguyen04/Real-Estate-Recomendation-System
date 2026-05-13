import pytest
import numpy as np
from src.utils.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_map,
    compute_diversity,
    compute_novelty
)

def test_precision_recall_at_k():
    actual = ["i1", "i2", "i3"]
    recommended = ["i1", "i4", "i2", "i5"]
    
    # Precision@2: ["i1", "i4"] -> "i1" is relevant -> 1/2 = 0.5
    assert compute_precision_at_k(actual, recommended, 2) == 0.5
    
    # Recall@3: ["i1", "i4", "i2"] -> "i1", "i2" are relevant -> 2/3 = 0.666...
    assert pytest.approx(compute_recall_at_k(actual, recommended, 3), 0.01) == 0.666
    
def test_ndcg_at_k():
    actual = ["i1", "i2"]
    recommended = ["i1", "i3", "i2"]
    
    # DCG@3: rel(i1)/log2(2) + rel(i3)/log2(3) + rel(i2)/log2(4)
    # = 1/1 + 0 + 1/2 = 1.5
    # IDCG@3: 1/log2(2) + 1/log2(3) = 1 + 0.63 = 1.63
    # Result: 1.5 / 1.63
    score = compute_ndcg_at_k(actual, recommended, 3)
    assert score > 0.9  # Should be high as i1 is at top
    
def test_map():
    actual = ["i1", "i3"]
    recommended = ["i1", "i2", "i3"]
    
    # P@1: 1/1 (hit i1)
    # P@2: 1/2 (miss i2)
    # P@3: 2/3 (hit i3)
    # AP: (1/1 + 2/3) / 2 = 0.833...
    assert pytest.approx(compute_map(actual, recommended), 0.01) == 0.833

def test_diversity():
    recommended = ["i1", "i2", "i3"]
    item_metadata = {
        "i1": {"cat": "A"},
        "i2": {"cat": "B"},
        "i3": {"cat": "A"}
    }
    
    # Pairs: (i1,i2) [A,B] - mismatch, (i1,i3) [A,A] - match, (i2,i3) [B,A] - mismatch
    # Matches: 1. Total pairs: 3.
    # Diversity: 1 - 1/3 = 0.666...
    assert pytest.approx(compute_diversity(recommended, item_metadata, "cat"), 0.01) == 0.666

def test_novelty():
    recommended = ["i1", "i2"]
    item_popularity = {"i1": 100, "i2": 10}
    total_users = 1000
    
    # p(i1) = 101/1001, p(i2) = 11/1001
    # novelty = (-log2(0.1) + -log2(0.01)) / 2 approx
    score = compute_novelty(recommended, item_popularity, total_users)
    assert score > 0
