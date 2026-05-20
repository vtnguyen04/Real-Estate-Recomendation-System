import polars as pl
from typing import Dict, Optional, TYPE_CHECKING

from src.features.base import BaseHeuristicExtractor

if TYPE_CHECKING:
    from src.features.feature_context import FeatureContext


class ItemQualityExtractor(BaseHeuristicExtractor):
    """
    Item quality features: photos, completeness, legal status, listing age,
    city, category, price bucket, ad type.

    Inference: reads from context.item_metadata dict (O(1)).
    Training:  build_feature_df() returns a polars DataFrame with all item quality columns.

    Args:
        df_listing: dim_listing DataFrame.
        split_date: reference date for listing_age_days; if None, column is omitted.
    """

    def __init__(self, df_listing: pl.DataFrame, split_date=None):
        self._df_listing = df_listing
        self._split_date = split_date

    @property
    def join_key(self) -> str:
        return "item_id"

    # ── Inference ──────────────────────────────────────────────

    def extract_scores(
        self,
        uid: str,  # noqa: ARG002
        context: "FeatureContext",
        features_dict: Dict[str, Dict[str, float]],
    ) -> None:
        for it in features_dict:
            meta = context.item_metadata.get(it, {})
            features_dict[it]["item_completeness"] = float(meta.get("completeness", 0.0))
            features_dict[it]["item_photos"]        = float(meta.get("images_count", 0.0))
            features_dict[it]["item_has_so_hong"]   = float(meta.get("has_so_hong", 0.0))
            features_dict[it]["item_is_apartment"]  = float(meta.get("is_apartment", 0.0))
            features_dict[it]["item_is_agent"]      = float(meta.get("is_agent", 0.0))
            features_dict[it]["item_has_noi_that_cao_cap"] = float(meta.get("has_noi_that_cao_cap", 0.0))

    # ── Training ───────────────────────────────────────────────

    def build_feature_df(self, context: Optional["FeatureContext"] = None) -> pl.DataFrame:  # noqa: ARG002
        """Return item_meta_df with quality + city/cat/price/ad_type + optional listing_age_days."""
        df = self._df_listing.with_columns([
            pl.col("images_count").fill_null(0).cast(pl.Float32).alias("item_photos"),
            (
                (pl.col("images_count").fill_null(0) > 5).cast(pl.Int32) +
                pl.col("area_sqm").is_not_null().cast(pl.Int32) +
                pl.col("bedrooms").is_not_null().cast(pl.Int32) +
                pl.col("bathrooms").is_not_null().cast(pl.Int32)
            ).cast(pl.Float32).alias("item_completeness"),
            (
                pl.col("legal_status").fill_null("")
                .str.contains("(?i)sổ hồng|sổ đỏ").cast(pl.Int32)
            ).cast(pl.Float32).alias("item_has_so_hong"),
            pl.col("project_id").is_not_null().cast(pl.Float32).alias("item_is_apartment"),
            (pl.col("seller_type").fill_null("").str.to_lowercase() == "agent").cast(pl.Float32).alias("item_is_agent"),
            (pl.col("furnishing").fill_null("").str.contains("(?i)cao cấp")).cast(pl.Float32).alias("item_has_noi_that_cao_cap"),
            pl.col("city_name").fill_null("Unknown").cast(pl.Categorical).alias("item_city"),
            pl.col("category").cast(pl.Int64).alias("item_cat"),
            pl.col("price_bucket").alias("item_price"),
            pl.col("ad_type").alias("item_ad_type"),
        ])
        select_cols = [
            "item_id", "item_city", "item_cat",
            "item_photos", "item_completeness", "item_has_so_hong",
            "item_is_apartment", "item_is_agent", "item_has_noi_that_cao_cap",
            "item_price", "item_ad_type",
        ]
        if self._split_date is not None:
            df = df.with_columns(
                (pl.lit(self._split_date) - pl.col("posted_date")).dt.total_days()
                .clip(lower_bound=0).cast(pl.Float32).alias("listing_age_days")
            )
            select_cols.append("listing_age_days")
        return df.select(select_cols)
