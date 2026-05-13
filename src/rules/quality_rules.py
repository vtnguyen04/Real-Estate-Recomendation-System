import polars as pl
from src.core.base import BaseRule, RecommendationContext

class QualityScoreRule(BaseRule):
    """
    Computes a quality score for each listing based on:
    - Image count (8-12 is optimal)
    - Attribute completeness
    - Legal status (Sổ đỏ/hồng)
    - Title quality
    - Furnishing status (for rentals)
    
    This is not a hard filter, so it adds a 'quality_score' column.
    """
    def __init__(self, name: str = "quality_scorer"):
        super().__init__(name=name, is_hard_filter=False)

    def apply(self, items: pl.LazyFrame, context: RecommendationContext = None) -> pl.LazyFrame:
        # Base score is 1.0
        score_expr = pl.lit(1.0)
        
        # Rule 1: Image count logic (magic numbers from BĐS domain)
        # 8-12 images => * 1.4
        # >= 5 images => * 1.1
        # < 3 images => * 0.6
        score_expr = pl.when((pl.col("images_count") >= 8) & (pl.col("images_count") <= 12)) \
            .then(score_expr * 1.4) \
            .when(pl.col("images_count") >= 5) \
            .then(score_expr * 1.1) \
            .when(pl.col("images_count") < 3) \
            .then(score_expr * 0.6) \
            .otherwise(score_expr)
            
        # Rule 2: Attribute completeness
        required_fields = ['area_sqm', 'bedrooms', 'bathrooms', 'direction', 'legal_status']
        # Compute how many are non-null
        filled_count = pl.sum_horizontal([
            pl.col(f).is_not_null().cast(pl.Float32) for f in required_fields
        ])
        completeness_ratio = filled_count / len(required_fields)
        
        # Multiply score by (0.7 + 0.3 * completeness_ratio)
        score_expr = score_expr * (0.7 + 0.3 * completeness_ratio)
        
        # Rule 3: Legal status
        # If the property has legal documentation, it increases trust
        score_expr = pl.when(pl.col("legal_status").is_in(['Đã có sổ', 'Sổ hồng riêng'])) \
            .then(score_expr * 1.3) \
            .otherwise(score_expr)
            
        # Rule 4: Title quality (penalize if too generic or short)
        # Check if length < 20
        score_expr = pl.when(pl.col("title").str.len_chars() < 20) \
            .then(score_expr * 0.85) \
            .otherwise(score_expr)
            
        # Rule 5: Furnished (for rental ad_type == 'let')
        score_expr = pl.when((pl.col("ad_type") == 'let') & (pl.col("furnishing") == 'Nội thất đầy đủ')) \
            .then(score_expr * 1.2) \
            .otherwise(score_expr)
            
        # Cap the score between 0.3 and 2.0 to prevent extreme anomalies
        score_expr = score_expr.clip(lower_bound=0.3, upper_bound=2.0)
        
        return items.with_columns(score_expr.alias("quality_score"))
