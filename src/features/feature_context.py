"""
src/features/feature_context.py

Holds ALL shared state needed by feature extractors and the training pipeline.
Built once; accessed O(1) during inference via dicts, O(n) joins during training via DataFrames.
"""
import polars as pl
from collections import defaultdict
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger("feature_context")


class FeatureContext:
    """
    Central store of user/item statistics and lookup tables.

    fit() computes everything from raw data once.
    Results available as:
      - Python dicts (inference: O(1) per user/item)
      - Polars DataFrames (training: efficient batch joins)
    """

    def __init__(self):
        # ── Inference dicts (O(1) lookup) ──────────────────────
        self.valid_items: set = set()
        self.item_city: dict = {}
        self.seller_items: dict = defaultdict(list)
        self.user_prefs: dict = {}           # {uid: (city, cat)}
        self.user_prev: dict = defaultdict(list)
        self.user_contacted_sellers: dict = defaultdict(set)
        self.item_stats: dict = defaultdict(lambda: {"contacts": 0, "views": 0})
        self.item_metadata: dict = defaultdict(dict)
        self.user_stats: dict = defaultdict(dict)
        self.global_top: list = []
        self.cc_lists: dict = defaultdict(list)
        self.city_lists: dict = defaultdict(list)
        self.cat_lists: dict = defaultdict(list)

    def fit(
        self,
        df_listing_collected: pl.DataFrame,
        interactions: pl.DataFrame,
        pageviews: "pl.DataFrame | pl.LazyFrame",
        df_snapshot_collected: Optional[pl.DataFrame] = None,
    ):
        """
        Build all shared state from raw data.

        Args:
            df_listing_collected: dim_listing with full item catalogue.
            interactions:         contact_pairs parquet (user_id, item_id, count, city_name,
                                  category, last_date).
            pageviews:            als_pageview_pairs (user_id, item_id, view_count).
            pos_cutoff:           Only contacts after this date count as 'recent'.
            split_date:           Listing age is computed relative to this date.
        """
        logger.info("Initializing FeatureContext...")

        # ── Valid items ────────────────────────────────────────
        self.valid_items = set(df_listing_collected["item_id"].to_list())
        self.item_city = dict(zip(
            df_listing_collected["item_id"].to_list(),
            df_listing_collected["city_name"].to_list(),
        ))

        # ── Seller index ────────────────────────────────────────
        seller_df = df_listing_collected.group_by("seller_id").agg(pl.col("item_id").alias("items"))
        self.seller_items = defaultdict(list, zip(
            seller_df["seller_id"].to_list(), seller_df["items"].to_list()
        ))

        # ── User stats (dicts) ──────────────────────────────────
        logger.info("Computing user stats...")
        user_agg = (
            interactions.group_by("user_id")
            .agg([
                pl.len().alias("event_count"),
                pl.col("count").sum().alias("contact_sum"),
            ])
            .with_columns(
                (pl.col("contact_sum") / pl.col("event_count").cast(pl.Float32)).alias("contact_rate")
            )
        )
        for r in user_agg.select(["user_id", "event_count", "contact_rate"]).iter_rows(named=True):
            self.user_stats[r["user_id"]]["event_count"] = r["event_count"]
            self.user_stats[r["user_id"]]["contact_rate"] = r["contact_rate"]

        # ── User preferences ────────────────────────────────────
        logger.info("Computing user preferences...")
        prefs = interactions.group_by("user_id").agg([
            pl.col("city_name").drop_nulls().mode().first().alias("city"),
            pl.col("category").drop_nulls().mode().first().alias("cat"),
        ])
        self.user_prefs = {u: (c, ca) for u, c, ca in prefs.iter_rows()}

        # ── User history ────────────────────────────────────────
        uniq = interactions.select(["user_id", "item_id"]).unique()
        hist = uniq.filter(pl.col("item_id").is_in(self.valid_items))
        hist_grp = hist.group_by("user_id").agg(pl.col("item_id").head(30).alias("items"))
        self.user_prev = defaultdict(list, zip(
            hist_grp["user_id"].to_list(), hist_grp["items"].to_list()
        ))

        sellers_grp = (
            hist.join(df_listing_collected.select(["item_id", "seller_id"]), on="item_id", how="inner")
            .select(["user_id", "seller_id"]).unique()
            .group_by("user_id").agg(pl.col("seller_id").alias("sellers"))
        )
        self.user_contacted_sellers = defaultdict(set, {
            u: set(s) for u, s in sellers_grp.iter_rows()
        })

        # ── Item stats (dicts) ──────────────────────────────────
        logger.info("Computing item stats...")
        contacts_agg = interactions.group_by(["city_name", "category", "item_id"]).agg(
            pl.len().alias("count")
        )
        if len(contacts_agg) > 0:
            self.global_top = (
                contacts_agg.group_by("item_id").agg(pl.col("count").sum().alias("c"))
                .sort("c", descending=True).head(500)["item_id"].to_list()
            )
            sorted_c = contacts_agg.sort("count", descending=True)
            for city, cat, items in sorted_c.group_by(["city_name", "category"]).agg(
                pl.col("item_id").head(300).alias("items")
            ).iter_rows():
                if city and cat:
                    self.cc_lists[(city, cat)] = list(items)
            for city, items in sorted_c.group_by("city_name").agg(
                pl.col("item_id").head(300).alias("items")
            ).iter_rows():
                if city:
                    self.city_lists[city] = list(items)
            for cat, items in sorted_c.group_by("category").agg(
                pl.col("item_id").head(300).alias("items")
            ).iter_rows():
                if cat:
                    self.cat_lists[cat] = list(items)

        pv = pageviews.collect() if isinstance(pageviews, pl.LazyFrame) else pageviews
        pv_agg = pv.group_by("item_id").agg(pl.col("view_count").sum().alias("views"))
        ic_agg = contacts_agg.group_by("item_id").agg(pl.col("count").sum().alias("contacts"))
        item_stats_full = (
            ic_agg.join(pv_agg, on="item_id", how="full", coalesce=True).fill_null(0)
        )
        self.item_stats = defaultdict(
            lambda: {"contacts": 0, "views": 0},
            {iid: {"contacts": c, "views": v}
             for iid, c, v in item_stats_full.select(["item_id", "contacts", "views"]).iter_rows()},
        )

        # ── Item metadata (dicts) ───────────────────────────────
        logger.info("Computing item metadata...")
        df_listing_meta = df_listing_collected.with_columns([
            pl.col("images_count").fill_null(0).alias("images_count"),
            (
                (pl.col("images_count").fill_null(0) > 5).cast(pl.Int32) +
                pl.col("area_sqm").is_not_null().cast(pl.Int32) +
                pl.col("bedrooms").is_not_null().cast(pl.Int32) +
                pl.col("bathrooms").is_not_null().cast(pl.Int32)
            ).alias("completeness"),
            (
                pl.col("legal_status").fill_null("").str.contains("(?i)sổ hồng|sổ đỏ").cast(pl.Int32)
            ).alias("has_so_hong"),
        ])
        for r in df_listing_meta.select([
            "item_id", "images_count", "completeness", "has_so_hong"
        ]).iter_rows(named=True):
            self.item_metadata[r["item_id"]].update({
                "images_count": r["images_count"],
                "completeness": r["completeness"],
                "has_so_hong": r["has_so_hong"],
            })

        logger.info("FeatureContext initialized.")

