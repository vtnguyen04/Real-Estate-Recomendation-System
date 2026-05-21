"""
src/features/extractors/item_snapshot.py — Item features from fact_listing_snapshot.

Provides daily-aggregated item performance metrics:
- item_avg_views_7d: average views_24h in last 7 days of training
- item_avg_contacts_7d: average contacts_24h in last 7 days of training
- item_conversion_rate: contacts / (views + 1) — non-linear signal (INS-042)
- item_trend_score: recent_views / (prior_views + 1) — momentum
- item_is_active: had any snapshot activity in last 7 days

Uses cached snapshot_stats.parquet from preprocessor.
"""
import os
from typing import Dict, Optional, TYPE_CHECKING

import polars as pl

from src.features.base import BaseHeuristicExtractor
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.features.feature_context import FeatureContext

logger = get_logger("item_snapshot_extractor")

SNAPSHOT_COLS = [
    "item_avg_views_7d", "item_avg_contacts_7d",
    "item_conversion_rate", "item_trend_score", "item_is_active",
]


class ItemSnapshotExtractor(BaseHeuristicExtractor):
    """
    Extracts item-level features from pre-aggregated fact_listing_snapshot data.

    Designed for LightGBM reranker features. Follows SOLID — single responsibility
    for snapshot-derived item features.
    """

    def __init__(self, snapshot_stats_path: str):
        """
        Args:
            snapshot_stats_path: Path to snapshot_stats.parquet (built by preprocessor)
        """
        self._path = snapshot_stats_path
        self._df: Optional[pl.DataFrame] = None
        self._lookup: Dict[str, dict] = {}

    def _ensure_loaded(self) -> None:
        if self._df is None:
            if os.path.exists(self._path):
                self._df = pl.read_parquet(self._path)
                logger.info(f"Snapshot stats loaded: {len(self._df):,} items from {self._path}")
                # Build lookup for inference mode
                for r in self._df.iter_rows(named=True):
                    self._lookup[r["item_id"]] = {c: r[c] for c in SNAPSHOT_COLS if c in r}
            else:
                logger.warning(f"Snapshot stats not found at {self._path}, creating empty")
                self._df = pl.DataFrame(schema={
                    "item_id": pl.Utf8,
                    **{c: pl.Float32 for c in SNAPSHOT_COLS},
                })

    @property
    def join_key(self) -> str:
        return "item_id"

    def extract_scores(
        self,
        uid: str,
        context: "FeatureContext",
        features_dict: Dict[str, Dict[str, float]],
    ) -> None:
        """Enrich item features with snapshot metrics."""
        self._ensure_loaded()
        for iid in features_dict:
            snap = self._lookup.get(iid, {})
            for col in SNAPSHOT_COLS:
                features_dict[iid][col] = snap.get(col, 0.0)

    def build_feature_df(self, context: "FeatureContext") -> Optional[pl.DataFrame]:
        """Return snapshot features DataFrame for join-based training."""
        self._ensure_loaded()
        return self._df

    def attach(self, df_pairs: pl.DataFrame) -> pl.DataFrame:
        """Convenience: join snapshot features onto a pairs DataFrame."""
        self._ensure_loaded()
        df = df_pairs.join(self._df, on="item_id", how="left")
        for c in SNAPSHOT_COLS:
            if c in df.columns:
                df = df.with_columns(pl.col(c).fill_null(0.0))
        return df

