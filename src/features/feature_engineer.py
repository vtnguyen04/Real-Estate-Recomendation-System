import polars as pl
from typing import List
from src.core.base import BaseRule, RecommendationContext

class FeatureEngineer:
    """
    Aggregates deterministic rules, user profiles, item profiles, and embeddings 
    to create the final dense feature matrix for the ML Ranker (LightGBM).
    Implementation exactly mirrors Task 2.6 in the Strategy Playbook, optimized via Polars LazyFrames.
    """
    def __init__(self, deterministic_rules: List[BaseRule]):
        # Base rules: geo_score, urgency_score, quality_score, match_score, value_score
        self.rules = deterministic_rules

    def engineer_features(self, 
                          candidate_items: pl.LazyFrame, 
                          user_profile: pl.LazyFrame,
                          item_profile: pl.LazyFrame,
                          interactions: pl.LazyFrame = None,
                          session_embeddings: pl.LazyFrame = None,
                          graph_embeddings: pl.LazyFrame = None,
                          context: RecommendationContext = None) -> pl.LazyFrame:
        """
        Builds the comprehensive (user, item) feature matrix.
        
        Args:
            candidate_items: LazyFrame with pairs of (user_id, item_id)
            user_profile: LazyFrame with user aggregate features
            item_profile: LazyFrame with item aggregate features + dimensions
            interactions: Optional LazyFrame containing historical interaction logs
            session_embeddings: Optional LazyFrame for GRU session vectors
            graph_embeddings: Optional LazyFrame for GraphSAGE item vectors
            context: The recommendation context
        """
        
        # 1. Base Joins
        df = candidate_items.join(user_profile, on="user_id", how="left")
        df = df.join(item_profile, on="item_id", how="left")
        
        # 2. Extract Cross-Features & Derived Signals
        schema = df.collect_schema().names()
        
        # Cold-Start Detection
        if "user_total_views" in schema:
            df = df.with_columns([
                (pl.col("user_total_views") < 5).cast(pl.Int32).alias("user_is_cold_start")
            ])
            
        # 3. Incorporate Historical Interactions (if provided)
        if interactions is not None:
            # Did the user already view this specific item?
            view_history = interactions.filter(pl.col('event_type') == 'pageview') \
                                       .select(['user_id', 'item_id']) \
                                       .with_columns(pl.lit(1).alias("user_viewed_this_item")) \
                                       .unique()
            df = df.join(view_history, on=["user_id", "item_id"], how="left")
            df = df.with_columns(pl.col("user_viewed_this_item").fill_null(0))
            
            # Session Price Rank
            # (Rank of this item's price compared to everything else user viewed in current session)
            if "price_vnd" in schema and "session_id" in interactions.collect_schema().names():
                session_prices = interactions.join(item_profile.select(['item_id', 'price_vnd']), on='item_id', how='inner')
                session_stats = session_prices.group_by("user_id").agg([
                    pl.col("price_vnd").mean().alias("session_avg_price")
                ])
                df = df.join(session_stats, on="user_id", how="left")
                df = df.with_columns([
                    (pl.col("price_vnd") / (pl.col("session_avg_price") + 1.0)).alias("session_price_ratio")
                ])
                
        # 4. Apply Deterministic Scoring Rules (Geo, Urgency, Quality, Match)
        # Sort by priority just in case rules have dependencies
        for rule in sorted(self.rules, key=lambda x: getattr(x, 'priority', 0), reverse=True):
            df = rule.apply(df, context)
            
        # 5. Attach Advanced Deep Learning Embeddings
        if session_embeddings is not None:
            # Expects columns: ['user_id', 'session_emb_0', ..., 'session_emb_N']
            df = df.join(session_embeddings, on="user_id", how="left")
            
        if graph_embeddings is not None:
            # Expects columns: ['item_id', 'graph_emb_0', ..., 'graph_emb_N']
            df = df.join(graph_embeddings, on="item_id", how="left")
            
        # 6. Final Data Cleaning
        # Fill missing numeric values with -1 (Standard for LightGBM)
        df = df.fill_null(-1)
            
        return df
