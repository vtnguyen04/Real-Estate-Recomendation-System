"""
Burst Trending Recommender.

Algorithm — Contact Burst Detection:
  burst_score(item) = recent_leads / (prior_leads_normalised + ε)

  where:
    recent_leads  = lead count in last `recent_days` before cutoff
    prior_leads   = lead count in the preceding `window_days - recent_days`

  Items with burst_score > 1 are "heating up" and should be prioritised
  for cold-start users over plain all-time popularity.

Cold-start dispatch (via context.metadata):
  If context.metadata contains 'pref_city' and/or 'pref_cat',
  the recommender returns city×cat → city → cat → global trending, in order.

Design (SOLID):
  - Single Responsibility: contact-burst trending only.
  - Open/Closed: subclass to change scoring formula.
  - Liskov: fully swappable with PopularityRecommender.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple

import polars as pl

from src.core.base import BaseRecommender, RecommendationContext


class BurstTrendingRecommender(BaseRecommender):
    """
    Popularity recommender with contact-burst detection for cold-start.

    fit() accepts the interactions table (item_id, date, lead_count) and
    a listing metadata DataFrame (item_id, city_name, category) for
    city/category-level trending lists.
    """

    def __init__(
        self,
        recent_days: int = 7,
        window_days: int = 30,
        global_k: int = 300,
        segment_k: int = 100,
        cc_k: int = 50,
        target_agent_ratio: float = 0.52,   # from gt_dist (EDA R09: INS-023)
    ):
        super().__init__(name="burst_trending")
        self.recent_days         = recent_days
        self.window_days         = window_days
        self.global_k            = global_k
        self.segment_k           = segment_k
        self.cc_k                = cc_k
        self.target_agent_ratio  = target_agent_ratio

        self._global: List[str]                     = []
        self._city:   Dict[str, List[str]]           = defaultdict(list)
        self._cat:    Dict[int, List[str]]            = defaultdict(list)
        self._cc:     Dict[Tuple[str, int], List[str]] = defaultdict(list)

    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, train_data: pl.LazyFrame, **kwargs) -> "BurstTrendingRecommender":
        """
        Build burst-weighted trending lists.

        Args:
            train_data: interactions LazyFrame with (item_id, date, lead_count).
            kwargs:
                cutoff_date  (date):         Last training date.
                listing_slim (pl.DataFrame): (item_id, city_name, category_dim).
                valid_items  (Set[str]):      Whitelist — items outside are excluded.
        """
        cutoff: date        = kwargs["cutoff_date"]
        listing_slim: pl.DataFrame = kwargs["listing_slim"]
        valid_items: Set[str]      = kwargs.get("valid_items", set())

        recent_start = cutoff - timedelta(days=self.recent_days)
        window_start = cutoff - timedelta(days=self.window_days)

        # Collect window interactions
        df = (
            train_data
            .filter(pl.col("date") >= pl.lit(window_start))
            .filter(pl.col("lead_count") > 0)
            .select(["item_id", "date", "lead_count"])
            .collect()
        )

        # Split into recent vs prior
        recent = (
            df.filter(pl.col("date") >= pl.lit(recent_start))
            .group_by("item_id")
            .agg(pl.col("lead_count").sum().alias("recent_leads"))
        )
        prior = (
            df.filter(pl.col("date") < pl.lit(recent_start))
            .group_by("item_id")
            .agg(pl.col("lead_count").sum().alias("prior_leads"))
        )

        # Join & compute burst score
        trend = (
            recent
            .join(prior, on="item_id", how="outer_coalesce" if False else "full", coalesce=True)
            .with_columns([
                pl.col("recent_leads").fill_null(0.0),
                pl.col("prior_leads").fill_null(0.0),
            ])
            .with_columns(
                (
                    pl.col("recent_leads") * 3.0
                    + pl.col("recent_leads")
                    / (pl.col("prior_leads") / self.recent_days + 1.0)
                ).alias("trend_score")
            )
            .join(listing_slim, on="item_id", how="left")
            .sort("trend_score", descending=True)
        )

        if valid_items:
            trend = trend.filter(pl.col("item_id").is_in(list(valid_items)))

        # Join seller_type from listing_slim for fairness selection
        if "seller_type" in listing_slim.columns:
            trend = trend.join(
                listing_slim.select(["item_id", "seller_type"]), on="item_id", how="left"
            )

        # Global — fairness-aware selection (interleave agent + private by target ratio)
        self._global = self._fair_select(trend, self.global_k)

        # City-level — group once, then fair-select per city (avoids N repeated filter calls)
        if "city_name" in trend.columns:
            city_groups = (
                trend.filter(pl.col("city_name").is_not_null())
                .sort("trend_score", descending=True)
                .partition_by("city_name", as_dict=True)
            )
            for city, city_df in city_groups.items():
                # partition_by returns dict key as tuple or scalar depending on version
                city_key = city[0] if isinstance(city, tuple) else city
                self._city[city_key] = self._fair_select(city_df, self.segment_k)

        # Category-level — group once
        if "category_dim" in trend.columns:
            cat_groups = (
                trend.filter(pl.col("category_dim").is_not_null())
                .sort("trend_score", descending=True)
                .partition_by("category_dim", as_dict=True)
            )
            for cat, cat_df in cat_groups.items():
                cat_key = cat[0] if isinstance(cat, tuple) else cat
                self._cat[cat_key] = self._fair_select(cat_df, self.segment_k)

        # City × Category
        cc_df = (
            trend
            .filter(pl.col("city_name").is_not_null() & pl.col("category_dim").is_not_null())
            .sort(["city_name", "category_dim", "trend_score"], descending=[False, False, True])
        )
        for row in cc_df.iter_rows(named=True):
            key = (row["city_name"], row["category_dim"])
            if len(self._cc[key]) < self.cc_k:
                self._cc[key].append(row["item_id"])

        return self

    def _fair_select(self, df: "pl.DataFrame", k: int) -> List[str]:
        """
        Greedy selection ensuring agent/private ratio ≈ target_agent_ratio.
        Items are interleaved: for every N agent items, add M private items.
        Falls back to pure score order when one seller type runs out.
        """
        if df.is_empty():
            return []

        has_seller = "seller_type" in df.columns
        if not has_seller:
            return df.head(k)["item_id"].to_list()

        agents   = df.filter(pl.col("seller_type") == "agent")["item_id"].to_list()
        privates = df.filter(pl.col("seller_type") != "agent")["item_id"].to_list()
        others   = df.filter(pl.col("seller_type").is_null())["item_id"].to_list()

        result: List[str] = []
        ai = pi = oi = 0          # index pointers into each seller bucket
        agent_count: int = 0      # O(1) tracking — no loop needed
        target = self.target_agent_ratio

        while len(result) < k:
            n_sel = max(1, len(result))
            cur_agent_ratio = agent_count / n_sel

            # Greedily fill to maintain agent ratio ≈ target
            if cur_agent_ratio < target and ai < len(agents):
                result.append(agents[ai]); ai += 1; agent_count += 1
            elif pi < len(privates):
                result.append(privates[pi]); pi += 1
            elif ai < len(agents):
                result.append(agents[ai]); ai += 1; agent_count += 1
            elif oi < len(others):
                result.append(others[oi]); oi += 1
            else:
                break

        return result[:k]

    # ─────────────────────────────────────────────────────────────────────────
    def recommend(
        self,
        context: RecommendationContext,
        candidates: Optional[pl.LazyFrame] = None,
    ) -> pl.LazyFrame:
        """
        Dispatch to best segment using context.metadata:
          - metadata['pref_city'] and metadata['pref_cat'] → city×cat → city → cat → global
          - If metadata is None → global trending only
        """
        uid      = context.user_id
        k        = context.num_recommendations
        meta     = context.metadata or {}
        pref_city: Optional[str] = meta.get("pref_city")
        pref_cat:  Optional[int] = meta.get("pref_cat")

        seen: Set[str]  = set(meta.get("exclude_items", []))
        recs: List[str] = []

        def _fill(pool: List[str]) -> None:
            for item in pool:
                if item not in seen and len(recs) < k:
                    recs.append(item); seen.add(item)

        if pref_city and pref_cat:
            _fill(self._cc.get((pref_city, pref_cat), []))
        if len(recs) < k and pref_city:
            _fill(self._city.get(pref_city, []))
        if len(recs) < k and pref_cat:
            _fill(self._cat.get(pref_cat, []))
        _fill(self._global)

        rows = [{"user_id": uid, "item_id": item, "score": float(k - i)}
                for i, item in enumerate(recs)]
        schema = {"user_id": pl.Utf8, "item_id": pl.Utf8, "score": pl.Float32}
        return (pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)).lazy()

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump((self._global, self._city, self._cat, self._cc), f)

    def load(self, path: str) -> "BurstTrendingRecommender":
        import pickle
        with open(path, "rb") as f:
            self._global, self._city, self._cat, self._cc = pickle.load(f)
        return self
