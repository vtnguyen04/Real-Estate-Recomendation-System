import pytest
import polars as pl
from src.pipeline.inference_pipeline import InferencePipeline
from src.features.feature_engineer import FeatureEngineer
from src.models.baselines.popularity import PopularityRecommender
from src.models.rerankers.multi_objective import MultiObjectiveReranker
from src.rules.geo_rules import GeoProximityScoreRule
from src.rules.quality_rules import QualityScoreRule
from src.core.base import BaseRecommender, RecommendationContext

class DummyRanker(BaseRecommender):
    def __init__(self, name: str = "dummy"):
        super().__init__(name=name)
    def fit(self, train_data): pass
    def load(self, path): pass
    def save(self, path): pass
    def recommend(self, context, candidates):
        # Just add a constant score if not present
        if "score" not in candidates.collect_schema().names():
            return candidates.with_columns(pl.lit(0.5).alias("score"))
        return candidates

def test_inference_pipeline_end_to_end(mock_user_events, mock_listings):
    # 1. Setup components
    # Stage 1
    pop_recommender = PopularityRecommender()
    pop_recommender.fit(mock_user_events)
    
    # Stage 2 Rules
    from src.rules.value_rules import ValueScoreRule
    rules = [
        GeoProximityScoreRule(),
        QualityScoreRule(),
        ValueScoreRule()
    ]
    fe = FeatureEngineer(deterministic_rules=rules)
    
    # Stage 3 Ranker (Dummy)
    ranker = DummyRanker()
    
    # Stage 4 Reranker
    reranker = MultiObjectiveReranker()
    
    # Pipeline
    pipeline = InferencePipeline(
        candidate_generator=pop_recommender,
        feature_engineer=fe,
        ranker=ranker,
        reranker=reranker,
        config={"top_n_for_rerank": 5}
    )
    
    # 2. Run pipeline
    # Prep profiles (minimal for test)
    user_profile = pl.LazyFrame([{"user_id": "u1", "user_total_views": 10}])
    item_profile = mock_listings
    
    results = pipeline.run(
        user_id="u1",
        k=2,
        user_profile=user_profile,
        item_profile=item_profile
    )
    
    # 3. Assertions
    final_df = results.collect()
    assert len(final_df) <= 2
    assert "score" in final_df.columns
    assert "quality_score" in final_df.columns
    assert "value_score" in final_df.columns
    # Check that it's sorted or at least has results
    assert final_df.height > 0
