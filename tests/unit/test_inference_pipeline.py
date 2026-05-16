import pytest
import polars as pl
from unittest.mock import MagicMock
from src.pipeline.inference_pipeline import InferencePipeline
from src.features.feature_engineer import FeatureEngineer

class DummyComponent:
    def __init__(self):
        pass

def test_inference_pipeline_save_load(tmp_path):
    mock_generator = DummyComponent()
    mock_ranker = DummyComponent()
    mock_reranker = DummyComponent()
    
    fe = FeatureEngineer(deterministic_rules=[])
    
    pipeline = InferencePipeline(
        candidate_generator=mock_generator,
        feature_engineer=fe,
        ranker=mock_ranker,
        reranker=mock_reranker
    )
    
    save_path = str(tmp_path / "pipeline.pkl")
    pipeline.save(save_path)
    
    loaded = InferencePipeline.load(save_path)
    assert loaded is not None
