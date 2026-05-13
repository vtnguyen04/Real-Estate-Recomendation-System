import polars as pl
from src.core.base import BaseFeatureExtractor

class UserBehaviorExtractor(BaseFeatureExtractor):
    """
    Extracts aggregated behavioral features for each user from clickstream data.
    """
    def __init__(self, name: str = "user_behavior_extractor", **kwargs):
        super().__init__(name=name, **kwargs)

    def _validate_input(self, data: pl.LazyFrame) -> None:
        expected_cols = {'user_id', 'event_type', 'dwell_time_sec'}
        # In a real setup, we might check data.columns if it was a DataFrame
        # For LazyFrame, we assume the pipeline guarantees schema.
        pass

    def _compute_features(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Computes user-level features:
        - Total number of interactions
        - Total positive interactions (contact intents)
        - Average dwell time
        """
        positive_events = [
            'view_phone', 'contact_chat', 'contact_zalo', 'contact_sms', 'other_interaction'
        ]
        
        # Calculate features per user
        user_features = data.group_by("user_id").agg([
            pl.len().alias("total_events"),
            pl.col("event_type").is_in(positive_events).sum().alias("total_positive_interactions"),
            pl.col("dwell_time_sec").mean().alias("avg_dwell_time_sec"),
            pl.col("category").mode().first().alias("favorite_category") # mode returns a list, take first
        ])
        
        # Calculate positive interaction rate
        user_features = user_features.with_columns([
            (pl.col("total_positive_interactions") / pl.col("total_events")).alias("positive_interaction_rate")
        ])
        
        return user_features

    def _post_process(self, features: pl.LazyFrame) -> pl.LazyFrame:
        # Fill nulls for avg_dwell_time_sec
        return features.fill_null(0.0)
