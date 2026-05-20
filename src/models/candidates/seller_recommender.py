"""
Seller Expansion Recommender.

Rationale (from domain knowledge):
  In real-estate marketplaces, a motivated buyer often browses MULTIPLE listings
  from the same agent/seller. If a user contacted item X from seller S, they
  likely want to see S's other active listings too.

Design (SOLID):
  - Single Responsibility: only seller-based candidate expansion.
  - fit() builds item→seller and seller→items maps from dim_listing data.
  - recommend() expands a user's contacted sellers into new candidate items.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set

import polars as pl

from src.core.base import BaseRecommender, RecommendationContext


class SellerExpansionRecommender(BaseRecommender):
    """
    Candidate expansion: user contacted seller X → recommend X's other listings.

    fit() takes TWO inputs (via kwargs):
      - interactions LazyFrame (user_id, item_id, lead_count) for user history
      - listing_df DataFrame   (item_id, seller_id)           for seller map
    """

    def __init__(self, max_items_per_seller: int = 10, max_sellers_per_user: int = 5):
        super().__init__(name="seller_expansion")
        self.max_items_per_seller = max_items_per_seller
        self.max_sellers_per_user = max_sellers_per_user

        self._user_history: Dict[str, List[str]] = {}   # uid → [item_id]
        self._item_to_seller: Dict[str, str]     = {}   # item_id → seller_id
        self._seller_to_items: Dict[str, List[str]] = {}  # seller_id → [item_id]

    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, train_data: pl.LazyFrame, **kwargs) -> "SellerExpansionRecommender":
        """
        Args:
            train_data: interactions LazyFrame with (user_id, item_id, lead_count).
            kwargs:
                listing_df (pl.DataFrame): dim_listing with columns (item_id, seller_id).
                valid_items (Set[str]):    whitelist; items outside this are ignored.
        """
        listing_df: pl.DataFrame = kwargs["listing_df"]
        valid_items: Set[str]    = kwargs.get("valid_items", set())

        # 1. item → seller map
        self._item_to_seller = {
            row["item_id"]: row["seller_id"]
            for row in listing_df.select(["item_id", "seller_id"]).iter_rows(named=True)
            if row["seller_id"]
        }

        # 2. seller → items map (all valid items)
        seller_to_items: Dict[str, List[str]] = defaultdict(list)
        for row in listing_df.select(["item_id", "seller_id"]).iter_rows(named=True):
            sid, iid = row["seller_id"], row["item_id"]
            if sid and (not valid_items or iid in valid_items):
                seller_to_items[sid].append(iid)
        self._seller_to_items = dict(seller_to_items)

        # 3. user → contact history (from interactions)
        user_hist: Dict[str, List[str]] = defaultdict(list)
        seen: Dict[str, Set[str]]       = defaultdict(set)
        for uid, iid in (
            train_data
            .filter(pl.col("count") > 0)
            .select(["user_id", "item_id"])
            .collect()
            .iter_rows()
        ):
            if iid not in seen[uid]:
                user_hist[uid].append(iid)
                seen[uid].add(iid)
        self._user_history = dict(user_hist)

        return self

    # ─────────────────────────────────────────────────────────────────────────
    def recommend(
        self,
        context: RecommendationContext,
        candidates: Optional[pl.LazyFrame] = None,
    ) -> pl.LazyFrame:
        """
        Return items from the same sellers as the user's contact history.
        Score = 1 / rank_within_seller (earlier sellers in history = higher weight).
        """
        uid      = context.user_id
        k        = context.num_recommendations
        history  = self._user_history.get(uid, [])
        hist_set = set(history)

        pool: List[dict] = []
        seen_sellers: Set[str] = set()

        for rank, hist_item in enumerate(history[: self.max_sellers_per_user * 3]):
            sid = self._item_to_seller.get(hist_item)
            if not sid or sid in seen_sellers:
                continue
            seen_sellers.add(sid)
            seller_items = self._seller_to_items.get(sid, [])
            score = 1.0 / (rank + 1)
            for alt in seller_items[: self.max_items_per_seller]:
                if alt not in hist_set:
                    pool.append({"user_id": uid, "item_id": alt, "score": score})

            if len(pool) >= k * 3:
                break

        # De-duplicate keeping highest score per item
        best: Dict[str, dict] = {}
        for entry in pool:
            iid = entry["item_id"]
            if iid not in best or entry["score"] > best[iid]["score"]:
                best[iid] = entry

        recs = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:k]
        schema = {"user_id": pl.Utf8, "item_id": pl.Utf8, "score": pl.Float32}
        return (pl.DataFrame(recs, schema=schema) if recs else pl.DataFrame(schema=schema)).lazy()

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(
                (self._user_history, self._item_to_seller, self._seller_to_items), f
            )

    def load(self, path: str) -> "SellerExpansionRecommender":
        import pickle
        with open(path, "rb") as f:
            self._user_history, self._item_to_seller, self._seller_to_items = pickle.load(f)
        return self
