"""
User-KNN Recommender via contact co-occurrence.

Algorithm:
  1. fit(): Build item→neighbors and neighbor→contacts lookup maps
             scoped only to items in the query users' history (memory-safe).
  2. recommend(): Pool candidates from neighbor contacts, scored by frequency.

Design (SOLID):
  - Single Responsibility: pure neighborhood-based CF, no other logic.
  - Open/Closed: extend via subclass, never modify this class for new models.
  - Liskov: fully swappable with any other BaseRecommender.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

import polars as pl

from src.core.base import BaseRecommender, Recommendation, RecommendationContext


class UserKNNRecommender(BaseRecommender):
    """
    Neighborhood-based CF: "users who contacted the same items also contacted X."

    Memory-safe design: item→neighbor maps are built only for items
    that appear in query_user_ids' contact history, not for the full corpus.
    """

    def __init__(
        self,
        max_neighbors_per_item: int = 30,
        max_items_per_neighbor: int = 30,
        top_history_items: int = 5,
    ):
        super().__init__(name="user_knn")
        self.max_neighbors = max_neighbors_per_item
        self.max_items_per_neighbor = max_items_per_neighbor
        self.top_history = top_history_items

        # Populated by fit()
        self._user_contacts: Dict[str, List[str]] = {}    # uid → [item_id, ...]
        self._item_to_nbrs: Dict[str, List[str]] = {}     # item → [neighbor_uid, ...]
        self._nbr_contacts: Dict[str, List[str]] = {}     # neighbor_uid → [item_id, ...]

    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, train_data: pl.LazyFrame, **kwargs) -> "UserKNNRecommender":
        """
        Build lookup maps from positive interaction data.

        Args:
            train_data: LazyFrame with columns (user_id, item_id, lead_count).
                        Rows with lead_count > 0 are treated as positive.
            kwargs:
                query_user_ids (Set[str]): users for whom we will call recommend().
                                           Scopes the item→neighbor map to their history
                                           items only — critical for memory safety.
                valid_items    (Set[str]): whitelist for candidate items.
        """
        query_user_set: Set[str] = kwargs.get("query_user_ids", set())
        valid_items: Set[str]    = kwargs.get("valid_items", set())

        # 1. Materialise positive pairs
        pos_df = (
            train_data
            .filter(pl.col("count") > 0)
            .select(["user_id", "item_id"])
            .collect()
        )

        # 2. Build user → contacts (all users, for neighbor lookups)
        user_contacts: Dict[str, List[str]] = defaultdict(list)
        seen: Dict[str, Set[str]] = defaultdict(set)
        for uid, iid in pos_df.iter_rows():
            if iid not in seen[uid] and len(user_contacts[uid]) < self.max_items_per_neighbor:
                user_contacts[uid].append(iid)
                seen[uid].add(iid)
        self._user_contacts = dict(user_contacts)

        # 3. Identify items contacted by query users (scope reduction)
        query_history_items: Set[str] = {
            iid
            for uid in query_user_set
            for iid in user_contacts.get(uid, [])
        }

        # 4. item → neighbor users (non-query, capped at max_neighbors)
        item_to_nbrs: Dict[str, List[str]] = defaultdict(list)
        for uid, iid in pos_df.iter_rows():
            if iid in query_history_items and uid not in query_user_set:
                if len(item_to_nbrs[iid]) < self.max_neighbors:
                    item_to_nbrs[iid].append(uid)
        self._item_to_nbrs = dict(item_to_nbrs)

        # 5. neighbor → their other contacts (already in user_contacts, just filter)
        neighbor_set = {uid for users in item_to_nbrs.values() for uid in users}
        self._nbr_contacts = {
            uid: contacts
            for uid, contacts in user_contacts.items()
            if uid in neighbor_set
        }

        return self

    # ─────────────────────────────────────────────────────────────────────────
    def recommend(
        self,
        context: RecommendationContext,
        candidates: Optional[pl.LazyFrame] = None,
    ) -> pl.LazyFrame:
        """
        Return top-K co-occurred items for context.user_id.
        Score = number of neighbors who also contacted the candidate item.
        """
        uid   = context.user_id
        k     = context.num_recommendations
        hist  = self._user_contacts.get(uid, [])
        hist_set = set(hist)

        cand_scores: Counter = Counter()
        for hist_item in hist[: self.top_history]:
            for nbr in self._item_to_nbrs.get(hist_item, []):
                for item in self._nbr_contacts.get(nbr, []):
                    if item not in hist_set:
                        cand_scores[item] += 1

        recs = [
            {"user_id": uid, "item_id": item, "score": float(count)}
            for item, count in cand_scores.most_common(k)
        ]
        schema = {"user_id": pl.Utf8, "item_id": pl.Utf8, "score": pl.Float32}
        return (pl.DataFrame(recs, schema=schema) if recs else pl.DataFrame(schema=schema)).lazy()

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(
                (self._user_contacts, self._item_to_nbrs, self._nbr_contacts), f
            )

    def load(self, path: str) -> "UserKNNRecommender":
        import pickle
        with open(path, "rb") as f:
            self._user_contacts, self._item_to_nbrs, self._nbr_contacts = pickle.load(f)
        return self
