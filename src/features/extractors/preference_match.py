import polars as pl
from typing import Dict, TYPE_CHECKING, Optional

from src.features.base import BaseHeuristicExtractor

if TYPE_CHECKING:
    from src.features.feature_context import FeatureContext


class PreferenceMatchExtractor(BaseHeuristicExtractor):
    """
    Pairwise features: computes matching scores between user preferences and item attributes.
    This extractor runs *after* user and item features have been joined.
    """

    @property
    def join_key(self) -> str:
        return "pairs"  # Special key to indicate it operates on the full pairs dataframe

    # ── Inference ──────────────────────────────────────────────

    def extract_scores(
        self,
        uid: str,
        context: "FeatureContext",
        features_dict: Dict[str, Dict[str, float]],
    ) -> None:
        """
        In inference, user_stats and item_metadata are in the context.
        We compute the match on the fly.
        """
        user_stats = context.user_stats.get(uid, {})
        pref_city = user_stats.get("pref_city")
        pref_cat = user_stats.get("pref_cat")
        pref_price = user_stats.get("pref_price")
        pref_ad_type = user_stats.get("pref_ad_type")

        for it in features_dict:
            item_meta = context.item_metadata.get(it, {})
            
            city_match = 1.0 if pref_city and pref_city == item_meta.get("city_name") else 0.0
            cat_match = 1.0 if pref_cat and str(pref_cat) == str(item_meta.get("category")) else 0.0
            price_match = 1.0 if pref_price and pref_price == item_meta.get("price_bucket") else 0.0
            ad_type_match = 1.0 if pref_ad_type and pref_ad_type == item_meta.get("ad_type") else 0.0
            
            features_dict[it]["city_match"] = city_match
            features_dict[it]["cat_match"] = cat_match
            features_dict[it]["price_match"] = price_match
            features_dict[it]["ad_type_match"] = ad_type_match

    # ── Training ───────────────────────────────────────────────

    def build_feature_df(self, context: Optional["FeatureContext"] = None) -> Optional[pl.DataFrame]:
        """
        For pairwise extractors, build_feature_df is not used directly.
        Instead, we provide a method to compute features on the joined dataframe.
        """
        return None

    def compute_match_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append city_match, cat_match, price_match, ad_type_match columns if source cols exist."""
        match_exprs = []
        if "pref_city" in df.columns and "item_city" in df.columns:
            match_exprs.append(
                (pl.col("pref_city").cast(pl.Utf8) == pl.col("item_city").cast(pl.Utf8))
                .cast(pl.Int32).fill_null(0).alias("city_match")
            )
        if "pref_cat" in df.columns and "item_cat" in df.columns:
            match_exprs.append(
                (pl.col("pref_cat").cast(pl.Int64) == pl.col("item_cat").cast(pl.Int64))
                .cast(pl.Int32).fill_null(0).alias("cat_match")
            )
        if "pref_price" in df.columns and "item_price" in df.columns:
            match_exprs.append(
                (pl.col("pref_price").cast(pl.Utf8) == pl.col("item_price").cast(pl.Utf8))
                .cast(pl.Int32).fill_null(0).alias("price_match")
            )
        if "pref_ad_type" in df.columns and "item_ad_type" in df.columns:
            match_exprs.append(
                (pl.col("pref_ad_type").cast(pl.Utf8) == pl.col("item_ad_type").cast(pl.Utf8))
                .cast(pl.Int32).fill_null(0).alias("ad_type_match")
            )
        return df.with_columns(match_exprs) if match_exprs else df
