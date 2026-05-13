import polars as pl
from typing import List
from src.core.base import BaseRule, RecommendationContext

class FeatureEngineer:
    """
    Aggregates all deterministic rules, user profiles, and item profiles 
    to create the final feature matrix for the ML Ranker.
    """
    def __init__(self, deterministic_rules: List[BaseRule]):
        self.rules = deterministic_rules

    def engineer_features(self, 
                          candidate_items: pl.LazyFrame, 
                          user_profile: pl.LazyFrame,
                          item_profile: pl.LazyFrame,
                          context: RecommendationContext) -> pl.LazyFrame:
        """
        Builds the (user, item) feature matrix.
        
        Args:
            candidate_items: LazyFrame with pairs of (user_id, item_id)
            user_profile: LazyFrame with user aggregate features
            item_profile: LazyFrame with item aggregate features + dimensions
            context: The recommendation context
        """
        
        # 1. Join user features
        df = candidate_items.join(user_profile, on="user_id", how="left")
        
        # 2. Join item features
        df = df.join(item_profile, on="item_id", how="left")
        
        # 3. Apply all deterministic scoring rules
        # Rules like QualityScore, UrgencyScore, MatchScore expect the required
        # columns (like 'images_count', 'bedrooms', 'user_modal_bedrooms') to be present in df.
        for rule in sorted(self.rules, key=lambda x: x.priority, reverse=True):
            df = rule.apply(df, context)
            
        return df
