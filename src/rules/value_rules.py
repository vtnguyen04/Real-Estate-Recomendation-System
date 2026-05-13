import polars as pl
from src.core.base import BaseRule, RecommendationContext

class ValueScoreRule(BaseRule):
    """
    Computes a value score based on the item's price relative to the market average
    for its specific location (City/District) and Category.
    Items priced below the market average receive a boost.
    """
    def __init__(self, name: str = "value_scorer"):
        super().__init__(name=name, is_hard_filter=False)

    def apply(self, items: pl.LazyFrame, context: RecommendationContext = None) -> pl.LazyFrame:
        schema_names = items.collect_schema().names()
        
        # We expect 'item_price' and 'market_avg_price' (for same category/geo) to be in the LazyFrame
        # These would typically be joined in the FeatureEngineer stage
        if "price" in schema_names and "market_avg_price" in schema_names:
            # price_ratio = price / market_avg_price
            price_ratio = pl.col("price") / (pl.col("market_avg_price") + 1e-10)
            
            # Value score: inverse of price ratio
            # If price_ratio = 0.8 (20% discount), score = 1.25
            # If price_ratio = 1.2 (20% markup), score = 0.83
            value_score_expr = 1.0 / price_ratio
            
            # Dampen the effect to avoid extreme boosts for likely fake low prices
            # Use a non-linear mapping or clipping
            value_score_expr = value_score_expr.clip(lower_bound=0.5, upper_bound=1.8)
        else:
            # Fallback if market data is missing
            value_score_expr = pl.lit(1.0)
            
        return items.with_columns(value_score_expr.alias("value_score"))
