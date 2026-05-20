import polars as pl
from typing import Dict, Optional
from src.features.base import BaseHeuristicExtractor
from src.features.feature_context import FeatureContext


class RecentHistoryExtractor(BaseHeuristicExtractor):
    """
    Extracts features based on the user's recent interaction history.
    Computes 'score_prev'.
    """
    def __init__(self, contacts: Optional[pl.DataFrame] = None):
        self._contacts = contacts
        self._recent_df: Optional[pl.DataFrame] = None  # cache

    @property
    def join_key(self) -> str:
        return "pairs"

    def extract_scores(self, uid: str, context: FeatureContext, features_dict: Dict[str, Dict[str, float]]):
        # Top 5 most recent contacts
        for rank, it in enumerate(context.user_prev.get(uid, [])[:5]):
            features_dict[it]["score_prev"] = 200.0 - rank * 10

    def _build_recent_lookup(self) -> pl.DataFrame:
        """Build and cache the recent contacts lookup (top 5 per user)."""
        if self._recent_df is not None:
            return self._recent_df
        self._recent_df = (
            self._contacts
            .sort(["user_id", "last_date"], descending=[False, True])
            .group_by("user_id").head(5)
            .with_columns(
                pl.arange(0, pl.len()).over("user_id").alias("_rank")
            )
            .with_columns(
                (200.0 - pl.col("_rank") * 10).cast(pl.Float32).alias("score_prev")
            )
            .select(["user_id", "item_id", "score_prev"])
        )
        return self._recent_df

    def compute_match_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._contacts is None:
            return df.with_columns(pl.lit(0.0).cast(pl.Float32).alias("score_prev"))

        recent_df = self._build_recent_lookup()
        # Left join — only fill score_prev column, leave other columns untouched
        if "score_prev" in df.columns:
            df = df.drop("score_prev")
        df = df.join(recent_df, on=["user_id", "item_id"], how="left")
        df = df.with_columns(pl.col("score_prev").fill_null(0.0))
        return df
