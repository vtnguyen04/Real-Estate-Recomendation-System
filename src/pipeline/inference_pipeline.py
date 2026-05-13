import polars as pl
from typing import List, Dict, Any, Optional
from src.core.base import RecommendationContext, BaseRecommender, BaseRule
from src.features.feature_engineer import FeatureEngineer

class InferencePipeline:
    """
    Production Inference Pipeline orchestrating the 4-stage hybrid system.
    """
    def __init__(self, 
                 candidate_generator: BaseRecommender,
                 feature_engineer: FeatureEngineer,
                 ranker: BaseRecommender,
                 reranker: Any, # MultiObjectiveReranker
                 config: Dict[str, Any] = None):
        self.candidate_generator = candidate_generator
        self.feature_engineer = feature_engineer
        self.ranker = ranker
        self.reranker = reranker
        self.config = config or {}

    def run(self, 
            user_id: str, 
            k: int = 10,
            user_profile: pl.LazyFrame = None,
            item_profile: pl.LazyFrame = None,
            interactions: pl.LazyFrame = None,
            session_embeddings: pl.LazyFrame = None,
            graph_embeddings: pl.LazyFrame = None) -> pl.LazyFrame:
        """
        Executes the end-to-end recommendation flow.
        """
        context = RecommendationContext(user_id=user_id, num_recommendations=k)
        
        # STAGE 1: Candidate Generation (Stage 1)
        # Returns ~500 candidates
        candidates = self.candidate_generator.recommend(context)
        
        # STAGE 2: Feature Engineering & Deterministic Scoring (Stage 2)
        # Apply rules and prepare dense feature matrix
        featured_candidates = self.feature_engineer.engineer_features(
            candidate_items=candidates,
            user_profile=user_profile,
            item_profile=item_profile,
            interactions=interactions,
            session_embeddings=session_embeddings,
            graph_embeddings=graph_embeddings,
            context=context
        )
        
        # STAGE 3: ML Ranking (Stage 3)
        # Ranks candidates based on multi-task LightGBM scores
        ranked_candidates = self.ranker.recommend(context, featured_candidates)
        
        # STAGE 4: Multi-Objective Re-ranking (Stage 4)
        # Greedy selection for health objectives (Diversity, Fairness, Freshness)
        # Usually re-ranks top 30-50 candidates down to top-k
        top_n_for_rerank = self.config.get("top_n_for_rerank", 30)
        final_recommendations = self.reranker.rerank(
            ranked_candidates.head(top_n_for_rerank), 
            k=k
        )
        
        return final_recommendations

    @classmethod
    def load(cls, path: str):
        """
        Placeholder for loading a serialized pipeline.
        In a real scenario, this would load all sub-components.
        """
        import joblib
        return joblib.load(path)

    def save(self, path: str):
        """
        Placeholder for saving the pipeline.
        """
        import joblib
        joblib.dump(self, path)
