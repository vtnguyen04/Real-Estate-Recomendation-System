import os
import polars as pl
import time
from typing import List

from config.settings import PipelineConfig
from src.utils.logging import get_logger

logger = get_logger("preprocessor")

class DataPreprocessor:
    """
    Centralized module for pre-aggregating large datasets into compact caches.
    Adheres to SOLID by encapsulating the aggregation logic away from scripts.
    """
    def __init__(self, config: PipelineConfig, cache_dir: str):
        self.config = config
        self.cache_dir = cache_dir
        self.gt_events = config.data.positive_events
        os.makedirs(self.cache_dir, exist_ok=True)

    def process_and_cache(self, lf: pl.LazyFrame):
        """
        Executes all 6 pre-aggregation steps on the user events LazyFrame.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PREPROCESSING — Aggregate 41GB events → compact cache")
        logger.info("=" * 60)

        self._process_contact_pairs(lf)
        self._process_als_contact_pairs(lf)
        self._process_pageview_pairs(lf)
        self._process_date_range(lf)
        self._process_session_items(lf)
        self._process_cold_user_prefs(lf)

        logger.info(f"Preprocessing done. Time: {time.time() - t0:.1f}s")
        logger.info("=" * 60)

    def _process_contact_pairs(self, lf: pl.LazyFrame):
        logger.info("── [1/6] Contact pairs ──")
        contacts = (
            lf
            .filter(pl.col("is_login") == "login")
            .filter(pl.col("event_type").is_in(self.gt_events))
            .group_by(["user_id", "item_id", "city_name", "category"])
            .agg([
                pl.len().alias("count"),
                pl.col("date").max().alias("last_date"),
            ])
            .collect(engine="streaming")
        )
        out = os.path.join(self.cache_dir, "contact_pairs.parquet")
        contacts.write_parquet(out)
        logger.info(f"  {out}: {len(contacts):,} pairs, {os.path.getsize(out)/1e6:.1f}MB")

    def _process_als_contact_pairs(self, lf: pl.LazyFrame):
        logger.info("── [2/6] ALS contact pairs ──")
        als_contacts = (
            lf
            .filter(pl.col("is_login") == "login")
            .filter(pl.col("is_contact") == 1)
            .group_by(["user_id", "item_id"])
            .agg(pl.len().alias("score"))
            .collect(engine="streaming")
        )
        out = os.path.join(self.cache_dir, "als_contact_pairs.parquet")
        als_contacts.write_parquet(out)
        logger.info(f"  {out}: {len(als_contacts):,} pairs, {os.path.getsize(out)/1e6:.1f}MB")

    def _process_pageview_pairs(self, lf: pl.LazyFrame):
        logger.info("── [3/6] Pageview pairs ──")
        pageview_pairs = (
            lf
            .filter(pl.col("is_login") == "login")
            .filter(pl.col("event_type") == "pageview")
            .group_by(["user_id", "item_id"])
            .agg([
                pl.len().alias("view_count"),
                pl.col("dwell_time_sec").mean().alias("avg_dwell"),
            ])
            .collect(engine="streaming")
        )
        out = os.path.join(self.cache_dir, "als_pageview_pairs.parquet")
        pageview_pairs.write_parquet(out)
        logger.info(f"  {out}: {len(pageview_pairs):,} pairs, {os.path.getsize(out)/1e6:.1f}MB")

    def _process_date_range(self, lf: pl.LazyFrame):
        logger.info("── [4/6] Date range ──")
        date_range = lf.select([
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
        ]).collect()
        out = os.path.join(self.cache_dir, "date_range.parquet")
        date_range.write_parquet(out)
        logger.info(f"  Date: {date_range['min_date'][0]} → {date_range['max_date'][0]}")

    def _process_session_items(self, lf: pl.LazyFrame):
        logger.info("── [5/6] Session co-occurrence ──")
        session_items = (
            lf
            .filter(pl.col("is_login") == "login")
            .filter(pl.col("session_id").is_not_null())
            .group_by(["session_id", "item_id"])
            .agg(pl.len().alias("n"))
            .group_by("session_id")
            .agg([
                pl.col("item_id").alias("items"),
                pl.len().alias("n_items"),
            ])
            .filter((pl.col("n_items") >= self.config.data.session_min_items) & (pl.col("n_items") <= self.config.data.session_max_items))
            .collect(engine="streaming")
        )
        out = os.path.join(self.cache_dir, "session_items.parquet")
        session_items.write_parquet(out)
        logger.info(f"  {out}: {len(session_items):,} sessions, {os.path.getsize(out)/1e6:.1f}MB")

    def _process_cold_user_prefs(self, lf: pl.LazyFrame):
        """
        INS-027: Extract city/category preferences for cold-start users
        from pageview events. These users have NO contact history but
        DO have browsing signals that reveal geographic/category intent.

        Output: cold_user_prefs.parquet with columns:
          user_id, pref_city, pref_cat (from pageview city_name/category mode)
        """
        logger.info("── [6/6] Cold user preferences (from pageviews) ──")

        # Users with contact history (warm users)
        contact_users = (
            lf
            .filter(pl.col("is_login") == "login")
            .filter(pl.col("event_type").is_in(self.gt_events))
            .select("user_id").unique()
            .collect(engine="streaming")
        )["user_id"].to_list()
        warm_set = set(contact_users)

        # Extract prefs from pageview city/category for non-warm users
        cold_prefs = (
            lf
            .filter(pl.col("is_login") == "login")
            .filter(pl.col("event_type") == "pageview")
            .filter(~pl.col("user_id").is_in(list(warm_set)))
            .group_by("user_id")
            .agg([
                pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
                pl.col("category").drop_nulls().mode().first().alias("pref_cat"),
            ])
            .filter(
                pl.col("pref_city").is_not_null() | pl.col("pref_cat").is_not_null()
            )
            .collect(engine="streaming")
        )

        out = os.path.join(self.cache_dir, "cold_user_prefs.parquet")
        cold_prefs.write_parquet(out)
        logger.info(
            f"  {out}: {len(cold_prefs):,} cold users with prefs, "
            f"{os.path.getsize(out)/1e6:.1f}MB"
        )

