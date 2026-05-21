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
import hashlib
from typing import List, Dict, Tuple, Optional, Set, Any

import polars as pl

from src.core.base import BaseRecommender, RecommendationContext
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Competition ground truth events
GT_EVENTS = ['view_phone', 'contact_chat', 'other_interaction', 'contact_zalo', 'contact_sms']


def _stable_hash_int(value: Optional[str]) -> int:
    """Stable process-independent hash for deterministic user diversification."""
    if not value:
        return 0
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


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
        self._blind_global: List[str] = []

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

    def set_blind_global_from_snapshot(
        self,
        snapshot_df: pl.DataFrame,
        valid_items: Optional[Set[str]] = None,
        top_k: int = 500,
    ) -> "SegmentPopularityRecommender":
        """
        Set no-preference blind fallback from item-side snapshot demand.

        This is only used when a user has no city/category preference. In that
        case user-side personalization is unavailable, so recent marketplace
        demand is a stronger prior than hash-diversified historical contacts.
        Supports either raw fact_listing_snapshot columns or cached
        snapshot_stats.parquet columns.
        """
        valid_set = valid_items if valid_items else None
        df = snapshot_df
        if valid_set:
            df = df.filter(pl.col("item_id").is_in(list(valid_set)))

        cols = set(df.columns)
        if {"views_24h", "contacts_24h"}.issubset(cols):
            scored = (
                df.group_by("item_id")
                .agg([
                    pl.col("views_24h").sum().alias("views"),
                    pl.col("contacts_24h").sum().alias("contacts"),
                ])
                .with_columns((pl.col("contacts") * 20 + pl.col("views")).alias("score"))
            )
        elif {"item_avg_views_7d", "item_avg_contacts_7d"}.issubset(cols):
            scored = df.with_columns(
                (pl.col("item_avg_contacts_7d").fill_null(0) * 20
                 + pl.col("item_avg_views_7d").fill_null(0)).alias("score")
            )
        else:
            logger.warning("Snapshot blind fallback skipped: unsupported snapshot schema")
            return self

        self._blind_global = (
            scored.sort("score", descending=True)
            .head(top_k)["item_id"].to_list()
        )
        logger.info(f"Blind snapshot fallback set: {len(self._blind_global):,} items")
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
            
        # INS-064/H-024: Proportional cold-start fallback
        if len(recs) < k and not pref_city and not pref_cat:
            _weighted = [
                ("Tp Hồ Chí Minh", 1050), ("Tp Hồ Chí Minh", 1020),
                ("Tp Hồ Chí Minh", 1010), ("Đà Nẵng", 1020),
                ("Hà Nội", 1020), ("Tp Hồ Chí Minh", 1040),
            ]
            for seg in _weighted:
                if len(recs) >= k:
                    break
                _fill(self._cc.get(seg, []))

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
            
        # INS-064/H-024 + INS-054: Proportional blind allocation based on blind
        # contact distribution, but keep the item set fixed. Previous hash-offset
        # diversity changed the candidate set and hurt relevance; for no-signal
        # users, the best prior should be shared. We rotate only final order so
        # rank-1 exposure is not concentrated while Recall@10 is preserved.
        # Blind users: 1050=39.6%, 1020=30.5%, 1010=15.9%, 1040=7.8%, 1030=6.2%; HCM=73.8%
        if len(recs) < k and not pref_city and not pref_cat:
            if self._blind_global:
                _fill(self._blind_global)
            remaining = k - len(recs)
            _weighted_segments = [
                # (segment, fraction of blind contacts)
                (("Tp Hồ Chí Minh", 1050), 0.293),  # HCM × 1050 = 39.6% × 0.74
                (("Tp Hồ Chí Minh", 1020), 0.225),  # HCM × 1020 = 30.5% × 0.74
                (("Tp Hồ Chí Minh", 1010), 0.117),  # HCM × 1010 = 15.9% × 0.74
                (("Tp Hồ Chí Minh", 1040), 0.058),  # HCM × 1040
                (("Tp Hồ Chí Minh", 1030), 0.046),  # HCM × 1030
                (("Đà Nẵng", 1020),         0.048),  # Đà Nẵng (6.5%)
                (("Hà Nội", 1020),          0.047),  # Hà Nội (6.4%)
                (("Bình Dương", 1020),       0.020),  # Bình Dương
                (("Đà Nẵng", 1050),         0.020),
                (("Hà Nội", 1050),          0.020),
            ]
            slot_alloc = []
            for seg, frac in _weighted_segments:
                n_slots = max(1, round(frac * remaining))
                slot_alloc.append((seg, n_slots))

            for seg, n_slots in slot_alloc:
                if len(recs) >= k:
                    break
                pool = self._cc.get(seg, [])
                taken = 0
                for item in pool:
                    if len(recs) >= k or taken >= n_slots:
                        break
                    if item not in seen:
                        recs.append(item); seen.add(item)
                        taken += 1

        _fill(self._global)
        if user_id and not pref_city and not pref_cat and len(recs) > 1:
            shift = _stable_hash_int(user_id) % min(len(recs), k)
            recs = recs[shift:] + recs[:shift]
        return recs

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(
                (self._global, dict(self._city), dict(self._cat),
                 dict(self._cc), dict(self._ccd), self._blind_global),
                f,
            )

    def load(self, path: str) -> "SegmentPopularityRecommender":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            if len(data) == 5:
                g, ci, ca, cc, ccd = data
                blind_global = []
            else:
                g, ci, ca, cc, ccd, blind_global = data
            self._global = g
            self._city = defaultdict(list, ci)
            self._cat = defaultdict(list, ca)
            self._cc = defaultdict(list, cc)
            self._ccd = defaultdict(list, ccd)
            self._blind_global = blind_global
        return self
