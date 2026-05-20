import polars as pl
from typing import Dict, Optional
from src.features.base import BaseHeuristicExtractor
from src.features.feature_context import FeatureContext


class SellerAffinityExtractor(BaseHeuristicExtractor):
    """
    Extracts features based on the user's affinity towards sellers they've contacted before.
    Computes 'score_seller' with geographic boosting.
    """
    def __init__(
        self,
        contacts: Optional[pl.DataFrame] = None,
        df_listing: Optional[pl.DataFrame] = None,
    ):
        self._contacts = contacts
        self._df_listing = df_listing
        self._user_sellers: Optional[pl.DataFrame] = None  # cache

    @property
    def join_key(self) -> str:
        return "pairs"

    def extract_scores(self, uid: str, context: FeatureContext, features_dict: Dict[str, Dict[str, float]]):
        pref_city, _pref_cat = context.user_prefs.get(uid, (None, None))

        for seller_id in context.user_contacted_sellers.get(uid, set()):
            for it in context.seller_items.get(seller_id, [])[:5]:
                if it in context.valid_items:
                    boost = 2.0 if context.item_city.get(it) == pref_city else 0.3
                    features_dict[it]["score_seller"] += 15.0 * boost

    def _build_user_sellers(self) -> pl.DataFrame:
        """Build and cache user→seller lookup table."""
        if self._user_sellers is not None:
            return self._user_sellers
        self._user_sellers = (
            self._contacts.join(
                self._df_listing.select(["item_id", "seller_id", "city_name"]),
                on="item_id", how="left",
            )
            .filter(pl.col("seller_id").is_not_null())
            .select(["user_id", "seller_id"])
            .unique()
        )
        return self._user_sellers

    def compute_match_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._contacts is None or self._df_listing is None:
            return df.with_columns(pl.lit(0.0).cast(pl.Float32).alias("score_seller"))

        user_sellers = self._build_user_sellers()

        # Get seller_id + city_name for candidate items
        cand_with_seller = df.select(["user_id", "item_id"]).join(
            self._df_listing.select(["item_id", "seller_id", "city_name"]),
            on="item_id", how="left",
        )

        # Inner join to find items whose seller the user has contacted before
        matched = cand_with_seller.join(user_sellers, on=["user_id", "seller_id"], how="inner")

        # Get user pref_city from df if available
        if "pref_city" in df.columns:
            matched = matched.join(
                df.select(["user_id", "pref_city"]).unique(), on="user_id", how="left"
            )
            matched = matched.with_columns(
                pl.when(pl.col("city_name") == pl.col("pref_city"))
                .then(30.0).otherwise(4.5)
                .cast(pl.Float32).alias("score_seller")
            )
        else:
            matched = matched.with_columns(
                pl.lit(15.0).cast(pl.Float32).alias("score_seller")
            )

        # Keep only the max score per (user, item) to avoid duplicates
        matched = (
            matched.group_by(["user_id", "item_id"])
            .agg(pl.col("score_seller").max())
        )

        if "score_seller" in df.columns:
            df = df.drop("score_seller")
        df = df.join(matched.select(["user_id", "item_id", "score_seller"]),
                      on=["user_id", "item_id"], how="left")
        df = df.with_columns(pl.col("score_seller").fill_null(0.0))
        return df
