import polars as pl
from typing import Dict, TYPE_CHECKING

from src.features.base import BaseHeuristicExtractor

if TYPE_CHECKING:
    from src.features.feature_context import FeatureContext


class UserBehaviorExtractor(BaseHeuristicExtractor):
    """
    User-level features: activity statistics and category/price preferences.

    Inference: reads from context.user_stats dict (O(1)).
    Training:  build_feature_df() returns a polars DataFrame with all user columns.

    Args:
        contacts:   contact_pairs DataFrame (user_id, item_id, count, city_name, category, last_date).
        df_listing: dim_listing DataFrame — needed to compute pref_price / pref_ad_type.
    """

    def __init__(self, contacts: pl.DataFrame, df_listing: pl.DataFrame):
        self._contacts = contacts
        self._df_listing = df_listing

    @property
    def join_key(self) -> str:
        return "user_id"

    # ── Inference ──────────────────────────────────────────────

    def extract_scores(
        self,
        uid: str,
        context: "FeatureContext",
        features_dict: Dict[str, Dict[str, float]],
    ) -> None:
        stats = context.user_stats.get(uid, {})  # noqa: read from inference dict
        event_count  = float(stats.get("event_count", 0.0))
        contact_rate = float(stats.get("contact_rate", 0.0))
        for it in features_dict:
            features_dict[it]["event_count"]  = event_count
            features_dict[it]["contact_rate"] = contact_rate

    # ── Training ───────────────────────────────────────────────

    def build_feature_df(self, context: "FeatureContext" = None) -> pl.DataFrame:
        """Return user_stats_df with: event_count, contact_rate, pref_city, pref_cat,
        pref_price, pref_ad_type."""
        contacts = self._contacts

        user_agg = (
            contacts.group_by("user_id")
            .agg([
                pl.len().alias("event_count"),
                pl.col("count").sum().alias("contact_sum"),
            ])
            .with_columns(
                (pl.col("contact_sum") / pl.col("event_count").cast(pl.Float32))
                .alias("contact_rate")
            )
            .select(["user_id", "event_count", "contact_rate"])
        )
        user_city_cat = contacts.group_by("user_id").agg([
            pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
            pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("pref_cat"),
        ])
        contact_with_listing = contacts.join(
            self._df_listing.select(["item_id", "price_bucket", "ad_type"]),
            on="item_id", how="left",
        )
        user_price = contact_with_listing.group_by("user_id").agg([
            pl.col("price_bucket").drop_nulls().mode().first().alias("pref_price"),
            pl.col("ad_type").drop_nulls().mode().first().alias("pref_ad_type"),
        ])
        return (
            user_agg
            .join(user_city_cat, on="user_id", how="left")
            .join(user_price,    on="user_id", how="left")
        )
