"""
Item-to-Item Co-occurrence Recommender.

Two co-occurrence signals:
1. Session co-occurrence: items viewed together in same session → strongly related
2. Contact co-occurrence: items contacted by same user → similar intent

For each user, find items similar to what they've contacted/viewed,
weighted by dwell_time and co-occurrence frequency.

Insight basis:
  - INS-025: 85.5% GT items are NEW → need item similarity, not user history replay
  - INS-026: 91.9% city match → co-occurrence naturally captures geographic locality
  - Session data has 161M events with session_id grouping
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import gc
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, lil_matrix

from src.core.base import BaseRecommender, RecommendationContext
from src.utils.logging import get_logger

logger = get_logger(__name__)

GT_EVENTS = ['view_phone', 'contact_chat', 'contact_zalo', 'contact_sms']


class Item2ItemRecommender(BaseRecommender):
    """
    Item-to-Item CF using session co-occurrence.

    For each item, stores the top-N most co-occurring items.
    Recommendation: given user's contacted items, aggregate co-occurring items
    weighted by co-occurrence strength.
    """

    def __init__(self, top_k_similar: int = 50, min_cooccurrence: int = 3):
        super().__init__(name="item2item")
        self.top_k_similar = top_k_similar
        self.min_cooccurrence = min_cooccurrence
        # item_id → [(similar_item_id, score), ...]
        self._similar: Dict[str, List[Tuple[str, float]]] = {}

    def fit(self, train_data: pl.LazyFrame, **kwargs) -> "Item2ItemRecommender":
        """
        Build item co-occurrence matrix from session data.

        Args:
            train_data: fact_user_events LazyFrame.
            kwargs:
                valid_items (Set[str]): whitelist.
        """
        valid_items: Set[str] = kwargs.get("valid_items", set())

        # Step 1: Get session-item pairs (only pageviews + contacts in sessions)
        logger.info("Building session co-occurrence...")
        session_items = (
            train_data
            .filter(pl.col("is_login") == "login")
            .filter(pl.col("session_id").is_not_null())
            .filter(pl.col("event_type").is_in(["pageview"] + GT_EVENTS))
            .select(["session_id", "item_id", "event_type"])
            .collect(streaming=True)
        )
        logger.info(f"  Session-item pairs: {len(session_items):,}")

        # Step 2: Build item index
        items = session_items["item_id"].unique().to_list()
        if valid_items:
            items = [i for i in items if i in valid_items]
        item2idx = {it: i for i, it in enumerate(items)}
        n_items = len(items)
        logger.info(f"  Unique items: {n_items:,}")

        # Step 3: Group by session, build co-occurrence
        # For memory efficiency, process in chunks by session
        sessions = session_items.group_by("session_id").agg([
            pl.col("item_id").alias("items"),
        ])

        # Count co-occurrences using sparse matrix
        cooc = lil_matrix((n_items, n_items), dtype=np.float32)

        for row in sessions.iter_rows(named=True):
            items_in_session = list(set(row["items"]))  # unique items
            if len(items_in_session) < 2 or len(items_in_session) > 50:
                continue  # skip trivial or spam sessions
            # Get valid indices
            indices = [item2idx[it] for it in items_in_session if it in item2idx]
            if len(indices) < 2:
                continue
            # Increment co-occurrence for all pairs
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    cooc[indices[i], indices[j]] += 1.0
                    cooc[indices[j], indices[i]] += 1.0

        del sessions, session_items; gc.collect()
        logger.info(f"  Co-occurrence matrix built: {cooc.nnz:,} non-zeros")

        # Step 4: Convert to CSR and extract top-K similar per item
        cooc_csr = cooc.tocsr()
        del cooc; gc.collect()

        for i in range(n_items):
            row = cooc_csr.getrow(i)
            if row.nnz == 0:
                continue
            cols = row.indices
            vals = row.data

            # Filter by min_cooccurrence
            mask = vals >= self.min_cooccurrence
            if not mask.any():
                continue

            cols = cols[mask]
            vals = vals[mask]

            # Top-K by score
            if len(cols) > self.top_k_similar:
                top_idx = np.argpartition(vals, -self.top_k_similar)[-self.top_k_similar:]
                cols = cols[top_idx]
                vals = vals[top_idx]

            # Sort by score descending
            order = np.argsort(-vals)
            similar = [(items[cols[j]], float(vals[j])) for j in order]
            self._similar[items[i]] = similar

        del cooc_csr; gc.collect()
        logger.info(f"  Items with similar: {len(self._similar):,}")
        return self

    def recommend_for_user(
        self,
        seed_items: List[str],
        exclude: Optional[Set[str]] = None,
        k: int = 10,
        city_filter: Optional[str] = None,
        item_city: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Recommend items similar to user's seed items.

        Args:
            seed_items: items the user has contacted/viewed.
            exclude: items to skip.
            k: number of recs.
            city_filter: only return items in this city.
            item_city: item_id → city_name mapping.
        """
        seen = set(exclude or [])
        # Aggregate scores across all seed items
        item_scores: Dict[str, float] = defaultdict(float)

        for seed in seed_items:
            if seed in self._similar:
                for sim_item, score in self._similar[seed]:
                    if sim_item not in seen:
                        # Boost if city matches
                        if city_filter and item_city and item_city.get(sim_item) == city_filter:
                            item_scores[sim_item] += score * 1.5
                        elif not city_filter:
                            item_scores[sim_item] += score
                        else:
                            item_scores[sim_item] += score * 0.5

        if not item_scores:
            return []

        # Sort by aggregated score
        ranked = sorted(item_scores.items(), key=lambda x: -x[1])
        return [it for it, _ in ranked[:k]]

    def recommend(self, context: RecommendationContext, candidates=None) -> pl.LazyFrame:
        meta = context.metadata or {}
        seed = meta.get("seed_items", [])
        recs = self.recommend_for_user(seed, k=context.num_recommendations)
        rows = [{"user_id": context.user_id, "item_id": it, "score": float(len(recs) - i)}
                for i, it in enumerate(recs)]
        return pl.DataFrame(rows).lazy()

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self._similar, f)

    def load(self, path: str) -> "Item2ItemRecommender":
        import pickle
        with open(path, "rb") as f:
            self._similar = pickle.load(f)
        return self
