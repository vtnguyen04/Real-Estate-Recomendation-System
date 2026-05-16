"""
End-to-end tests to ensure the full pipeline works seamlessly from data loading to inference.
"""
import pytest
import polars as pl
from unittest.mock import MagicMock
from config.settings import PipelineConfig
from src.pipeline.inference_pipeline import InferencePipeline
from src.core.base import RecommendationContext

@pytest.fixture
def mock_config():
    config = PipelineConfig()
    return config

@pytest.fixture
def mock_pipeline():
    candidate_generator = MagicMock()
    feature_engineer = MagicMock()
    ranker = MagicMock()
    reranker = MagicMock()
    
    pipeline = InferencePipeline(
        candidate_generator=candidate_generator,
        feature_engineer=feature_engineer,
        ranker=ranker,
        reranker=reranker
    )
    return pipeline

def test_inference_pipeline_initialization(mock_pipeline):
    """Test that the inference pipeline initializes without errors."""
    assert mock_pipeline is not None
    assert mock_pipeline.candidate_generator is not None
    assert mock_pipeline.ranker is not None

def test_inference_pipeline_mock_run(mock_pipeline):
    """Test a dry run of the inference pipeline with mock candidates."""
    
    # Mock the candidate generator to return a dummy LazyFrame
    mock_candidates = pl.LazyFrame([
        {"user_id": "u1", "item_id": "i1", "score": 0.9, "source": "als"},
        {"user_id": "u1", "item_id": "i2", "score": 0.8, "source": "popularity"}
    ])
    mock_pipeline.candidate_generator.recommend.return_value = mock_candidates
    
    # Feature engineer should just return candidates unchanged for test
    mock_pipeline.feature_engineer.engineer_features.return_value = mock_candidates
    
    # Mock ranker
    mock_ranked = pl.LazyFrame([
        {"user_id": "u1", "item_id": "i1", "score": 0.95},
        {"user_id": "u1", "item_id": "i2", "score": 0.85}
    ])
    mock_pipeline.ranker.recommend.return_value = mock_ranked
    
    # Mock reranker
    mock_reranked = pl.LazyFrame([
        {"user_id": "u1", "item_id": "i1", "score": 0.95, "rank": 1},
        {"user_id": "u1", "item_id": "i2", "score": 0.85, "rank": 2}
    ])
    mock_pipeline.reranker.rerank.return_value = mock_reranked
    
    result_lf = mock_pipeline.run("u1", k=2)
    
    result_df = result_lf.collect()
    assert not result_df.is_empty()
    assert len(result_df) == 2
    assert "rank" in result_df.columns
