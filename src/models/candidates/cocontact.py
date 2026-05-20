"""
CoContactRecommender — Item-item co-contact graph recommender.

Single Responsibility: Build and query an item-item similarity graph
based on co-contact patterns (users who contacted item A also contacted
item B). Expands a user's contact history into related items.

Diagnostic showed this achieves Recall@10 = 0.107 standalone,
and is the best fallback for users without recent pageviews.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Set
from collections import defaultdict

import polars as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CoContactRecommender:
    """
    Item-item collaborative filtering via co-contact graph.

    For each item X, stores the top-K items most frequently co-contacted
    by the same users. At recommendation time, takes a user's recent
    contact history and expands it through the graph.
    """

    def __init__(
        self,
        window_days: int = 30,
        max_coitems_per_item: int = 50,
        max_user_items: int = 100,
        max_seed_items: int = 10,
        max_expand_per_seed: int = 10,
    ):
        """
        Args:
            window_days: Days of contact history to build graph from.
            max_coitems_per_item: Max co-contacted items stored per item.
            max_user_items: Skip users with more items (likely bots).
            max_seed_items: How many of user's recent contacts to use as seeds.
            max_expand_per_seed: How many co-items to expand per seed.
        """
        self.window_days = window_days
        self.max_coitems_per_item = max_coitems_per_item
        self.max_user_items = max_user_items
        self.max_seed_items = max_seed_items
        self.max_expand_per_seed = max_expand_per_seed
        # item_id -> [(co_item_id, count), ...] sorted by count desc
        self._graph: Dict[str, List[tuple]] = {}

    def fit(
        self,
        contact_pairs: pl.DataFrame,
        cutoff_date: Optional[object] = None,
    ) -> "CoContactRecommender":
        """
        Build the co-contact graph from contact pairs.

        Args:
            contact_pairs: DataFrame with columns [user_id, item_id, last_date, count].
            cutoff_date: Only use contacts before this date.

        Returns:
            self (for method chaining)
        """
        from datetime import timedelta

        data = contact_pairs
        if cutoff_date is not None:
            window_start = cutoff_date - timedelta(days=self.window_days)
            data = data.filter(pl.col("last_date") > window_start)

        # Build user -> items mapping
        user_items: Dict[str, Set[str]] = defaultdict(set)
        for r in data.iter_rows(named=True):
            user_items[r["user_id"]].add(r["item_id"])

        # Count co-occurrences (skip bot-like users)
        cocount: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        n_users_used = 0
        for uid, items in user_items.items():
            if len(items) > self.max_user_items:
                continue
            n_users_used += 1
            items_list = list(items)
            # Pairwise co-occurrence (capped to avoid O(n^2) explosion)
            for i in range(len(items_list)):
                for j in range(i + 1, min(len(items_list), i + 20)):
                    cocount[items_list[i]][items_list[j]] += 1
                    cocount[items_list[j]][items_list[i]] += 1

        # Keep top-K co-items per item
        self._graph = {}
        for item_id, coitems in cocount.items():
            top_pairs = sorted(coitems.items(), key=lambda x: -x[1])[
                : self.max_coitems_per_item
            ]
            self._graph[item_id] = top_pairs

        logger.info(
            f"CoContact graph built: {len(self._graph):,} items, "
            f"{n_users_used:,} users used (window={self.window_days}d)"
        )
        return self

    def recommend(
        self,
        seed_items: List[str],
        k: int = 10,
        exclude: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Expand seed items through co-contact graph.

        Args:
            seed_items: User's recent contact history (ordered by recency).
            k: Number of items to return.
            exclude: Items to exclude (e.g., seed items themselves).

        Returns:
            List of recommended item_ids, ordered by aggregated co-contact score.
        """
        exclude_set = exclude or set()
        expanded: Dict[str, float] = defaultdict(float)

        for seed in seed_items[: self.max_seed_items]:
            coitems = self._graph.get(seed, [])
            for co_item, cnt in coitems[: self.max_expand_per_seed]:
                if co_item not in exclude_set:
                    expanded[co_item] += cnt

        top_items = sorted(expanded.items(), key=lambda x: -x[1])[:k]
        return [item_id for item_id, _ in top_items]

    def recommend_batch(
        self,
        user_histories: Dict[str, List[str]],
        k: int = 10,
    ) -> Dict[str, List[str]]:
        """
        Batch recommend for multiple users.

        Args:
            user_histories: {user_id: [recent_item_ids]} ordered by recency.
            k: Number of items per user.

        Returns:
            {user_id: [recommended_item_ids]}
        """
        result = {}
        for uid, history in user_histories.items():
            exclude = set(history)
            result[uid] = self.recommend(history, k=k, exclude=exclude)
        return result

    @property
    def graph_size(self) -> int:
        """Number of items in the co-contact graph."""
        return len(self._graph)

    # ── Persistence ──────────────────────────────────────────
    def save(self, path: str) -> None:
        """Serialize to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "window_days": self.window_days,
                "max_coitems_per_item": self.max_coitems_per_item,
                "max_user_items": self.max_user_items,
                "max_seed_items": self.max_seed_items,
                "max_expand_per_seed": self.max_expand_per_seed,
                "graph": self._graph,
            }, f)
        logger.info(f"CoContact saved to {path}")

    def load(self, path: str) -> "CoContactRecommender":
        """Load from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.window_days = data["window_days"]
        self.max_coitems_per_item = data["max_coitems_per_item"]
        self.max_user_items = data["max_user_items"]
        self.max_seed_items = data["max_seed_items"]
        self.max_expand_per_seed = data["max_expand_per_seed"]
        self._graph = data["graph"]
        logger.info(f"CoContact loaded: {len(self._graph):,} items")
        return self
