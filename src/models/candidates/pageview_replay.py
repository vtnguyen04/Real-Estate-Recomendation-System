"""
PageviewReplayRecommender — Recommend items the user recently viewed/contacted.

Single Responsibility: Extract user's recent interactions from raw events
and rank them by weighted engagement (contact events >> pageviews).

INS-025: 85.5% of GT items are NEW to the user, BUT diagnostic showed
that pageview replay alone achieves Recall@10 = 0.197 — 5x better than
the ALS+SegPop pipeline. Users who view an item are highly likely to
contact it in the near future.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Set

import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Events considered positive (from competition spec)
GT_EVENTS = frozenset([
    "view_phone", "contact_chat", "other_interaction",
    "contact_zalo", "contact_sms",
])

# Weight for contact events vs pageviews
_CONTACT_WEIGHT = 10.0
_PAGEVIEW_WEIGHT = 1.0


class PageviewReplayRecommender:
    """
    Recommender that replays a user's recent browsing history.

    Scores items by: (contact_count * 10 + pageview_count * 1),
    tie-broken by recency (most recent interaction first).

    This is the strongest single signal for this dataset because
    users who view an item tend to contact it within days.
    """

    def __init__(self, window_days: int = 14, max_items_per_user: int = 50):
        """
        Args:
            window_days: How many days of recent events to consider.
            max_items_per_user: Maximum items to store per user.
        """
        self.window_days = window_days
        self.max_items_per_user = max_items_per_user
        self._user_items: Dict[str, List[str]] = {}

    def fit(
        self,
        events_path: str,
        user_ids: Optional[Set[str]] = None,
        cutoff_date: Optional[object] = None,  # date or datetime
    ) -> "PageviewReplayRecommender":
        """
        Build per-user ranked item lists from raw event data.

        Args:
            events_path: Glob path to fact_user_events parquet files.
            user_ids: If provided, only fit for these users.
            cutoff_date: Only use events before this date.

        Returns:
            self (for method chaining)
        """
        from datetime import timedelta

        scan = pl.scan_parquet(events_path)
        filters = []

        if cutoff_date is not None:
            window_start = cutoff_date - timedelta(days=self.window_days)
            filters.append(pl.col("event_ts") >= window_start)
            filters.append(pl.col("event_ts") <= cutoff_date)

        if user_ids is not None:
            filters.append(pl.col("user_id").is_in(list(user_ids)))

        if filters:
            for f in filters:
                scan = scan.filter(f)

        events = scan.select(["user_id", "item_id", "event_ts", "event_type"]).collect()
        logger.info(f"PageviewReplay: loaded {len(events):,} events ({self.window_days}d window)")

        # Weight contact events 10x vs pageviews
        events = events.with_columns(
            pl.when(pl.col("event_type").is_in(list(GT_EVENTS)))
            .then(_CONTACT_WEIGHT)
            .otherwise(_PAGEVIEW_WEIGHT)
            .alias("weight")
        )

        # Aggregate per (user, item): total weight + last interaction time
        ranked = (
            events
            .group_by(["user_id", "item_id"])
            .agg([
                pl.col("weight").sum().alias("total_weight"),
                pl.col("event_ts").max().alias("last_event"),
            ])
            .sort(
                ["user_id", "total_weight", "last_event"],
                descending=[False, True, True],
            )
        )

        # Build per-user lists
        self._user_items = {}
        for r in ranked.iter_rows(named=True):
            uid = r["user_id"]
            if uid not in self._user_items:
                self._user_items[uid] = []
            if len(self._user_items[uid]) < self.max_items_per_user:
                self._user_items[uid].append(r["item_id"])

        logger.info(
            f"PageviewReplay fitted: {len(self._user_items):,} users, "
            f"avg {sum(len(v) for v in self._user_items.values()) / max(1, len(self._user_items)):.1f} items/user"
        )
        return self

    def recommend(self, user_id: str, k: int = 10) -> List[str]:
        """Return top-k items for a single user."""
        return self._user_items.get(user_id, [])[:k]

    def recommend_batch(
        self,
        user_ids: List[str],
        k: int = 10,
    ) -> Dict[str, List[str]]:
        """Return top-k items for multiple users."""
        return {uid: self.recommend(uid, k) for uid in user_ids}

    def has_recommendations(self, user_id: str) -> bool:
        """Check if this recommender has data for a user."""
        return user_id in self._user_items and len(self._user_items[user_id]) > 0

    @property
    def coverage(self) -> int:
        """Number of users with recommendations."""
        return len(self._user_items)

    # ── Persistence ──────────────────────────────────────────
    def save(self, path: str) -> None:
        """Serialize to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "window_days": self.window_days,
                "max_items_per_user": self.max_items_per_user,
                "user_items": self._user_items,
            }, f)
        logger.info(f"PageviewReplay saved to {path}")

    def load(self, path: str) -> "PageviewReplayRecommender":
        """Load from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.window_days = data["window_days"]
        self.max_items_per_user = data["max_items_per_user"]
        self._user_items = data["user_items"]
        logger.info(f"PageviewReplay loaded: {len(self._user_items):,} users")
        return self
