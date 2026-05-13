import polars as pl
from src.core.base import BaseRule, RecommendationContext

class UrgencyScoreRule(BaseRule):
    """
    Computes an urgency score for each listing based on:
    - Listing age (exponential decay, fresh items are more urgent)
    - View velocity trend (recent 3 days vs previous 3 days)
    - Price drop detection
    
    This rule expects `item_stats` (daily snapshots) joined into `items` as a list of structs,
    or the velocity pre-calculated. For full Polars performance, we do the trend calculation
    if 'recent_3d_views' and 'older_3d_views' are provided.
    """
    def __init__(self, name: str = "urgency_scorer"):
        super().__init__(name=name, is_hard_filter=False)

    def apply(self, items: pl.LazyFrame, context: RecommendationContext = None) -> pl.LazyFrame:
        schema_names = items.collect_schema().names()
        
        # Component 1: Age score -> Exponential decay (e^(-0.03 * age_days))
        if "listing_age_days" in schema_names:
            age_score_expr = pl.col("listing_age_days") * -0.03
            age_score_expr = age_score_expr.exp()
        else:
            age_score_expr = pl.lit(0.5) # Default middle score if no age
            
        # Component 2: View velocity trend
        if "recent_3d_views" in schema_names and "older_3d_views" in schema_names:
            # velocity_change = (recent - older) / older
            # To avoid division by zero:
            velocity_change = (pl.col("recent_3d_views") - pl.col("older_3d_views")) / \
                              (pl.col("older_3d_views") + 1.0)
                              
            velocity_score_expr = pl.lit(1.0) + velocity_change.clip(-0.5, 1.0)
        else:
            velocity_score_expr = pl.lit(1.0)
            
        # Component 3: Price drop (boolean flag 'has_price_drop')
        if "has_price_drop" in schema_names:
            price_drop_expr = pl.when(pl.col("has_price_drop")).then(1.5).otherwise(1.0)
        else:
            price_drop_expr = pl.lit(1.0)

        # Final urgency
        urgency_expr = age_score_expr * velocity_score_expr * price_drop_expr
        
        # Cap at 2.0
        urgency_expr = urgency_expr.clip(lower_bound=0.0, upper_bound=2.0)
        
        return items.with_columns(urgency_expr.alias("urgency_score"))

