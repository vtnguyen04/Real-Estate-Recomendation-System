import pytest
import polars as pl
from src.pipeline.training_pipeline import TrainingPipeline
from src.models.candidates.als_recommender import ALSRecommender
from src.features.feature_engineer import FeatureEngineer
from src.models.rankers.lgbm_ranker import MultiTaskLGBMRanker

def test_training_pipeline_run(tmp_path):
    candidate_generator = ALSRecommender(factors=2, iterations=1)
    feature_engineer = FeatureEngineer(deterministic_rules=[])
    ranker = MultiTaskLGBMRanker(config={'rounds_bin': 1, 'rounds_multi': 1})
    
    pipeline = TrainingPipeline(
        candidate_generator=candidate_generator,
        feature_engineer=feature_engineer,
        ranker=ranker,
        config={"validation_days": 1}
    )
    
    # Mock data
    raw_events = pl.LazyFrame({
        "user_id": ["u1", "u1", "u2"],
        "item_id": ["i1", "i2", "i1"],
        "event_type": ["pageview", "contact_chat", "view_phone"],
        "timestamp": ["2026-03-31T10:00:00", "2026-04-01T10:00:00", "2026-04-02T10:00:00"],
        "dwell_time_sec": [20.0, 10.0, 5.0],
        "session_id": ["s0", "s1", "s2"],
        "label_binary": [0, 1, 1],
        "label_multiclass": [0, 1, 2],
        "group_id": [1, 1, 2]
    }).with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S"))
    
    item_profile = pl.LazyFrame({
        "item_id": ["i1", "i2"],
        "category": [1010, 1020],
        "price_vnd": [1000000, 2000000]
    })
    
    res = pipeline.run(raw_events, item_profile)
    
    assert res["status"] == "success"
    
    # Test save
    pipeline.save_models(str(tmp_path))
    assert (tmp_path / "candidate_generator.pkl").exists() or (tmp_path / "candidate_generator.pkl.npz").exists() or (tmp_path / "candidate_generator.npz").exists()
    assert (tmp_path / "ranker.pkl").exists() or (tmp_path / "binary_ranker.txt").exists()
