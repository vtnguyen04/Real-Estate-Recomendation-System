"""
Segment Popularity Recommender.

KEY INSIGHT (INS-025, INS-026):
  - 85.5% of GT items are COMPLETELY NEW to the user
  - 91.9% of GT items match user's preferred CITY
  - 72.2% match preferred CATEGORY
  - 68.5% match BOTH city + category

=> Segment popularity within (city, category) is the PRIMARY signal,
   NOT collaborative filtering.

Cascade: (city+cat+district) → (city+cat) → (city) → (cat) → global
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set, Any

import polars as pl

from src.core.base import BaseRecommender, RecommendationContext
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Competition ground truth events
GT_EVENTS = ['view_phone', 'contact_chat', 'other_interaction', 'contact_zalo', 'contact_sms']


class SegmentPopularityRecommender(BaseRecommender):
    """
    Recommends most popular items within user's preferred segment.

    Segments are hierarchical:
      1. (city, category, district) — finest grain
      2. (city, category)            — main signal (68.5% match rate)
      3. (city)                      — strong fallback (91.9% match rate)
      4. (category)                  — secondary
      5. global                      — last resort

    fit() builds popularity tables from GT contact events.
    recommend() dispatches to best matching segment.
    """

    def __init__(
        self,
        global_k: int = 500,
        segment_k: int = 200,
        cc_k: int = 200,
        ccd_k: int = 50,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="segment_popularity")
        self.config = config or {}
        self.global_k = self.config.get('segpop_global_k', global_k)
        self.segment_k = self.config.get('segpop_segment_k', segment_k)
        self.cc_k = self.config.get('segpop_cc_k', cc_k)
        self.ccd_k = self.config.get('segpop_ccd_k', ccd_k)

        self._global: List[str] = []
        self._city: Dict[str, List[str]] = defaultdict(list)
        self._cat: Dict[int, List[str]] = defaultdict(list)
        self._cc: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        self._ccd: Dict[Tuple[str, int, str], List[str]] = defaultdict(list)

    def fit(
        self,
        train_data: pl.LazyFrame,
        **kwargs,
    ) -> "SegmentPopularityRecommender":
        """
        Build segment popularity tables from GT contact events.

        Args:
            train_data: fact_user_events LazyFrame.
            kwargs:
                valid_items (Set[str]): items to include.
                listing_df (pl.DataFrame): with (item_id, district_name) for CCD.
        """
        valid_items: Set[str] = kwargs.get("valid_items", set())
        listing_df: Optional[pl.DataFrame] = kwargs.get("listing_df")

        # Filter to GT events only — use streaming for memory safety
        contacts = (
            train_data
            .filter(pl.col("is_login") == "login")
            .filter(pl.col("event_type").is_in(GT_EVENTS))
            .select(["user_id", "item_id", "city_name", "category"])
            .collect()
        )
        logger.info(f"GT contacts for popularity: {len(contacts):,}")

        valid_set = valid_items if valid_items else None

        # Global
        global_pop = (
            contacts.group_by("item_id")
            .agg(pl.len().alias("c"))
            .sort("c", descending=True)
        )
        if valid_set:
            global_pop = global_pop.filter(pl.col("item_id").is_in(list(valid_set)))
        self._global = global_pop.head(self.global_k)["item_id"].to_list()

        # City
        city_pop = (
            contacts.filter(pl.col("city_name").is_not_null())
            .group_by(["city_name", "item_id"])
            .agg(pl.len().alias("c"))
            .sort(["city_name", "c"], descending=[False, True])
        )
        for r in city_pop.iter_rows(named=True):
            if (not valid_set or r["item_id"] in valid_set) and len(self._city[r["city_name"]]) < self.segment_k:
                self._city[r["city_name"]].append(r["item_id"])

        # Category
        cat_pop = (
            contacts.filter(pl.col("category").is_not_null())
            .group_by(["category", "item_id"])
            .agg(pl.len().alias("c"))
            .sort(["category", "c"], descending=[False, True])
        )
        for r in cat_pop.iter_rows(named=True):
            if (not valid_set or r["item_id"] in valid_set) and len(self._cat[r["category"]]) < self.segment_k:
                self._cat[r["category"]].append(r["item_id"])

        # City × Category
        cc_pop = (
            contacts
            .filter(pl.col("city_name").is_not_null() & pl.col("category").is_not_null())
            .group_by(["city_name", "category", "item_id"])
            .agg(pl.len().alias("c"))
            .sort(["city_name", "category", "c"], descending=[False, False, True])
        )
        for r in cc_pop.iter_rows(named=True):
            key = (r["city_name"], r["category"])
            if (not valid_set or r["item_id"] in valid_set) and len(self._cc[key]) < self.cc_k:
                self._cc[key].append(r["item_id"])

        # City × Category × District (finest grain)
        if listing_df is not None:
            contacts_with_dist = contacts.join(
                listing_df.select(["item_id", "district_name"]), on="item_id", how="left"
            )
            ccd_pop = (
                contacts_with_dist
                .filter(
                    pl.col("city_name").is_not_null()
                    & pl.col("category").is_not_null()
                    & pl.col("district_name").is_not_null()
                )
                .group_by(["city_name", "category", "district_name", "item_id"])
                .agg(pl.len().alias("c"))
                .sort(["city_name", "category", "district_name", "c"], descending=[False, False, False, True])
            )
            for r in ccd_pop.iter_rows(named=True):
                key = (r["city_name"], r["category"], r["district_name"])
                if (not valid_set or r["item_id"] in valid_set) and len(self._ccd[key]) < self.ccd_k:
                    self._ccd[key].append(r["item_id"])

        logger.info(
            f"Segments: global={len(self._global)}, city={len(self._city)}, "
            f"cat={len(self._cat)}, cc={len(self._cc)}, ccd={len(self._ccd)}"
        )
        return self

    def fit_from_pairs(
        self,
        contact_pairs: pl.DataFrame,
        valid_items: Optional[Set[str]] = None,
        listing_df: Optional[pl.DataFrame] = None,
    ) -> "SegmentPopularityRecommender":
        """
        Build segment popularity tables from pre-aggregated contact pairs.
        Expects columns: user_id, item_id, city_name, category, count.
        INS-025/021: 85.5% of GT items are NEW to the user, and 69.7% of contacts
        happen within 7 days of posting. Raw popularity within segments is the
        best candidate signal because LightGBM can then learn to rerank using
        listing_age_days and item_recency_score features.
        """
        valid_set = valid_items if valid_items else None

        # ── Build scored DataFrame with district info ────────────
        scored = (
            contact_pairs
            .group_by(["item_id", "city_name", "category"])
            .agg(pl.col("count").sum().alias("score"))
            .with_columns(pl.col("score").cast(pl.Float64))
        )
        if listing_df is not None:
            scored = scored.join(
                listing_df.select(["item_id", "district_name"]),
                on="item_id", how="left"
            )

        # ── Global ───────────────────────────────────────────────
        global_pop = (
            scored.group_by("item_id")
            .agg(pl.col("score").sum().alias("s"))
            .sort("s", descending=True)
        )
        if valid_set:
            global_pop = global_pop.filter(pl.col("item_id").is_in(list(valid_set)))
        self._global = global_pop.head(self.global_k)["item_id"].to_list()

        # ── City ─────────────────────────────────────────────────
        city_pop = (
            scored.filter(pl.col("city_name").is_not_null())
            .group_by(["city_name", "item_id"])
            .agg(pl.col("score").sum().alias("s"))
            .sort(["city_name", "s"], descending=[False, True])
        )
        for r in city_pop.iter_rows(named=True):
            if (not valid_set or r["item_id"] in valid_set) and len(self._city[r["city_name"]]) < self.segment_k:
                self._city[r["city_name"]].append(r["item_id"])

        # ── Category ─────────────────────────────────────────────
        cat_pop = (
            scored.filter(pl.col("category").is_not_null())
            .group_by(["category", "item_id"])
            .agg(pl.col("score").sum().alias("s"))
            .sort(["category", "s"], descending=[False, True])
        )
        for r in cat_pop.iter_rows(named=True):
            if (not valid_set or r["item_id"] in valid_set) and len(self._cat[r["category"]]) < self.segment_k:
                self._cat[r["category"]].append(r["item_id"])

        # ── City × Category ─────────────────────────────────────
        cc_pop = (
            scored
            .filter(pl.col("city_name").is_not_null() & pl.col("category").is_not_null())
            .group_by(["city_name", "category", "item_id"])
            .agg(pl.col("score").sum().alias("s"))
            .sort(["city_name", "category", "s"], descending=[False, False, True])
        )
        for r in cc_pop.iter_rows(named=True):
            key = (r["city_name"], r["category"])
            if (not valid_set or r["item_id"] in valid_set) and len(self._cc[key]) < self.cc_k:
                self._cc[key].append(r["item_id"])

        # ── City × Category × District ──────────────────────────
        if "district_name" in scored.columns:
            ccd_pop = (
                scored
                .filter(
                    pl.col("city_name").is_not_null()
                    & pl.col("category").is_not_null()
                    & pl.col("district_name").is_not_null()
                )
                .group_by(["city_name", "category", "district_name", "item_id"])
                .agg(pl.col("score").sum().alias("s"))
                .sort(["city_name", "category", "district_name", "s"], descending=[False, False, False, True])
            )
            for r in ccd_pop.iter_rows(named=True):
                key = (r["city_name"], r["category"], r["district_name"])
                if (not valid_set or r["item_id"] in valid_set) and len(self._ccd[key]) < self.ccd_k:
                    self._ccd[key].append(r["item_id"])

        logger.info(
            f"Segments fitted from pairs: global={len(self._global)}, city={len(self._city)}, "
            f"cat={len(self._cat)}, cc={len(self._cc)}, ccd={len(self._ccd)}"
        )
        return self

    def recommend(
        self,
        context: RecommendationContext,
        candidates: Optional[pl.LazyFrame] = None,
    ) -> pl.LazyFrame:
        """
        Recommend from best matching segment.

        context.metadata should contain:
          - pref_city (str): user's preferred city
          - pref_cat (int): user's preferred category
          - pref_district (str, optional): user's preferred district
          - exclude_items (List[str], optional): items to skip
        """
        uid = context.user_id
        k = context.num_recommendations
        meta = context.metadata or {}
        pref_city = meta.get("pref_city")
        pref_cat = meta.get("pref_cat")
        pref_district = meta.get("pref_district")

        seen: set = set(meta.get("exclude_items", []))
        recs: List[str] = []

        def _fill(pool: List[str]) -> None:
            for item in pool:
                if item not in seen and len(recs) < k:
                    recs.append(item)
                    seen.add(item)

        # Cascade: finest → coarsest
        if pref_city and pref_cat and pref_district:
            _fill(self._ccd.get((pref_city, pref_cat, pref_district), []))
        if len(recs) < k and pref_city and pref_cat:
            _fill(self._cc.get((pref_city, pref_cat), []))
        if len(recs) < k and pref_city:
            _fill(self._city.get(pref_city, []))
        if len(recs) < k and pref_cat:
            _fill(self._cat.get(pref_cat, []))
            
        # INS-032: Cold-Start Fallback -> Force HCM/HN and 1010/1020
        if len(recs) < k and not pref_city and not pref_cat:
            _fill(self._cc.get(("Hồ Chí Minh", 1010), []))
            _fill(self._cc.get(("Hồ Chí Minh", 1020), []))
            _fill(self._cc.get(("Hà Nội", 1010), []))
            _fill(self._cc.get(("Hà Nội", 1020), []))

        _fill(self._global)

        rows = [
            {"user_id": uid, "item_id": item, "score": float(k - i)}
            for i, item in enumerate(recs)
        ]
        schema = {"user_id": pl.Utf8, "item_id": pl.Utf8, "score": pl.Float32}
        return (pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)).lazy()

    def get_segment_items(
        self,
        pref_city: Optional[str],
        pref_cat: Optional[int],
        pref_district: Optional[str] = None,
        exclude: Optional[set] = None,
        k: int = 10,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """Direct access to segment items without context overhead."""
        seen = set(exclude or [])
        recs = []

        def _fill(pool):
            for item in pool:
                if item not in seen and len(recs) < k:
                    recs.append(item); seen.add(item)

        if pref_city and pref_cat and pref_district:
            _fill(self._ccd.get((pref_city, pref_cat, pref_district), []))
        if len(recs) < k and pref_city and pref_cat:
            _fill(self._cc.get((pref_city, pref_cat), []))
        if len(recs) < k and pref_city:
            _fill(self._city.get(pref_city, []))
        if len(recs) < k and pref_cat:
            _fill(self._cat.get(pref_cat, []))
            
        # INS-032: Cold-Start Fallback → distribute blind users across top segments
        # using hash of user_id for deterministic but diverse assignment
        if len(recs) < k and not pref_city and not pref_cat:
            # Top segments by contact volume (covers ~95% of all contacts)
            _top_segments = [
                ("Tp Hồ Chí Minh", 1020),  # 41.8%
                ("Tp Hồ Chí Minh", 1050),  # 18.0%
                ("Tp Hồ Chí Minh", 1010),  # 13.6%
                ("Tp Hồ Chí Minh", 1030),  #  4.3%
                ("Tp Hồ Chí Minh", 1040),  #  2.4%
                ("Đà Nẵng", 1020),          #  2.0%
                ("Hà Nội", 1020),           #  1.8%
                ("Bình Dương", 1020),        #  1.3%
                ("Đà Nẵng", 1040),          #  1.1%
                ("Đà Nẵng", 1010),          #  0.9%
                ("Hà Nội", 1010),           #  0.9%
                ("Hà Nội", 1050),           #  0.9%
            ]
            # Use hash to assign each blind user to a different segment
            if user_id:
                seg_idx = hash(user_id) % len(_top_segments)
                primary_seg = _top_segments[seg_idx]
                _fill(self._cc.get(primary_seg, []))
                # Also fill from neighboring segments for diversity
                for offset in [1, 2, 3]:
                    if len(recs) >= k:
                        break
                    next_seg = _top_segments[(seg_idx + offset) % len(_top_segments)]
                    _fill(self._cc.get(next_seg, []))
            else:
                # No user context, use all top segments
                for seg in _top_segments:
                    if len(recs) >= k:
                        break
                    _fill(self._cc.get(seg, []))
            
        _fill(self._global)
        return recs

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(
                (self._global, dict(self._city), dict(self._cat),
                 dict(self._cc), dict(self._ccd)),
                f,
            )

    def load(self, path: str) -> "SegmentPopularityRecommender":
        import pickle
        with open(path, "rb") as f:
            g, ci, ca, cc, ccd = pickle.load(f)
            self._global = g
            self._city = defaultdict(list, ci)
            self._cat = defaultdict(list, ca)
            self._cc = defaultdict(list, cc)
            self._ccd = defaultdict(list, ccd)
        return self
