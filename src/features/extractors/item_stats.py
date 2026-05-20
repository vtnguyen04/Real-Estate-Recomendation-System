import polars as pl
from typing import Dict, Optional, TYPE_CHECKING

from src.features.base import BaseHeuristicExtractor

if TYPE_CHECKING:
    from src.features.feature_context import FeatureContext


class ItemStatsExtractor(BaseHeuristicExtractor):
    """
    Item-level engagement statistics: total contacts, total views, recency score.

    Inference: reads from context.item_stats dict (O(1)).
    Training:  build_feature_df() returns a polars DataFrame with all item stats columns.

    Args:
        contacts:      contact_pairs DataFrame (item_id, count, last_date, ...).
        als_pageviews: als_pageview_pairs DataFrame (item_id, view_count).
        pos_cutoff:    datetime; contacts after this date count as 'recent'.
    """

    def __init__(self, contacts: pl.DataFrame, als_pageviews: pl.DataFrame, pos_cutoff=None):
        self._contacts = contacts
        self._als_pageviews = als_pageviews
        self._pos_cutoff = pos_cutoff

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
            stats = context.item_stats.get(it, {})
            features_dict[it]["item_total_contacts"] = float(stats.get("contacts", 0.0))
            features_dict[it]["item_total_views"]    = float(stats.get("views", 0.0))
            features_dict[it]["item_novelty_score"]  = float(stats.get("novelty_score", 0.0))

    # ── Training ───────────────────────────────────────────────

    def build_feature_df(self, context: Optional["FeatureContext"] = None) -> pl.DataFrame:  # noqa: ARG002
        """Return item_stats_df with: item_total_contacts, item_total_views,
        item_recent_contacts, item_recency_score."""
        item_contact_df = (
            self._contacts.group_by("item_id")
            .agg(pl.col("count").sum().cast(pl.Float32).alias("item_total_contacts"))
        )
        item_view_df = (
            self._als_pageviews.group_by("item_id")
            .agg(pl.col("view_count").sum().cast(pl.Float32).alias("item_total_views"))
        )
        recent_contacts_df = pl.DataFrame(
            {"item_id": [], "item_recent_contacts": []},
            schema={"item_id": pl.Utf8, "item_recent_contacts": pl.Float32},
        )
        if self._pos_cutoff is not None:
            recent_contacts_df = (
                self._contacts.filter(pl.col("last_date") > self._pos_cutoff)
                .group_by("item_id")
                .agg(pl.col("count").sum().cast(pl.Float32).alias("item_recent_contacts"))
            )
        return (
            item_contact_df
            .join(item_view_df, on="item_id", how="full", coalesce=True)
            .join(recent_contacts_df, on="item_id", how="left")
            .fill_null(0)
            .with_columns([
                (pl.col("item_recent_contacts") / (pl.col("item_total_contacts") + 1e-6)).alias("item_recency_score")
            ])
            .sort("item_total_contacts", descending=True)
            .with_row_index("popularity_rank")
            .with_columns(
                (1.0 - (pl.col("popularity_rank") / pl.len())).cast(pl.Float32).alias("item_novelty_score")
            )
            .drop("popularity_rank")
        )
