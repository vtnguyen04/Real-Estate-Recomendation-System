import polars as pl
from src.core.base import BaseRule, RecommendationContext

class MatchScoreRule(BaseRule):
    """
    Explicit preference matching between User Profile and Item Profile.
    Evaluates:
    - Category match (checks membership in user_top_categories array)
    - Price bucket match (percentile constraints)
    - Geographic & Bedroom exact matches
    """
    def __init__(self, name: str = "match_scorer"):
        super().__init__(name=name, is_hard_filter=False)

    def apply(self, items: pl.LazyFrame, context: RecommendationContext = None) -> pl.LazyFrame:
        score_expr = pl.lit(1.0)
        schema_names = items.collect_schema().names()
        
        # Rule 1: Category Match
        # user_top_categories is expected to be a List[str]
        if "category" in schema_names and "user_top_categories" in schema_names:
            score_expr = pl.when(pl.col("user_top_categories").list.contains(pl.col("category"))) \
                .then(score_expr * 1.6) \
                .otherwise(score_expr * 0.4) # Strong penalty for unseen category
                
        # Rule 2: Price Range (Income constraint)
        # Assuming item_price_pct (0.0-1.0) and user_price_min, user_price_max are provided
        if "item_price_pct" in schema_names and "user_price_min" in schema_names and "user_price_max" in schema_names:
            in_range = (pl.col("item_price_pct") >= pl.col("user_price_min")) & \
                       (pl.col("item_price_pct") <= pl.col("user_price_max"))
                       
            # Calculate distance to mid range
            mid_range = (pl.col("user_price_min") + pl.col("user_price_max")) / 2
            close_enough = (pl.col("item_price_pct") - mid_range).abs() < 0.2
            
            score_expr = pl.when(in_range) \
                .then(score_expr * 1.4) \
                .when(close_enough) \
                .then(score_expr * 1.1) \
                .otherwise(score_expr * 0.7)
                
        # Rule 3: Geo Preference (Strong signal)
        if "city_name" in schema_names and "user_top_city" in schema_names:
            score_expr = pl.when(pl.col("city_name") == pl.col("user_top_city")) \
                .then(score_expr * 1.3) \
                .otherwise(score_expr)
                
        # Rule 4: Bedrooms Match (Strong signal for family/rentals)
        if "bedrooms" in schema_names and "user_modal_bedrooms" in schema_names:
            score_expr = pl.when(pl.col("bedrooms") == pl.col("user_modal_bedrooms")) \
                .then(score_expr * 1.5) \
                .otherwise(score_expr)
                
        # Cap score
        score_expr = score_expr.clip(lower_bound=0.2, upper_bound=2.5)
        
        return items.with_columns(score_expr.alias("match_score"))

