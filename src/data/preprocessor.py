import os
import polars as pl
import time
from typing import List

from datetime import date, timedelta

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

    def process_and_cache(self, lf: pl.LazyFrame, snapshot_path: str = ""):
        """
        Executes all pre-aggregation steps on the user events LazyFrame.
        Optionally processes fact_listing_snapshot if snapshot_path provided.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PREPROCESSING — Aggregate events → compact cache")
        logger.info("=" * 60)

        self._process_contact_pairs(lf)
        self._process_als_contact_pairs(lf)
        self._process_weighted_als(lf)
        self._process_pageview_pairs(lf)
        self._process_date_range(lf)
        self._process_session_items(lf)
        self._process_cold_user_prefs(lf)

        if snapshot_path and os.path.exists(snapshot_path):
            self._process_snapshot_stats(snapshot_path)

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

    def _process_weighted_als(self, lf: pl.LazyFrame):
        """
        INS-062: Build weighted ALS contact pairs.
        Real contacts (view_phone, chat, zalo, sms) get weight=3,
        other_interaction gets weight=1.
        """
        logger.info("── [3b/8] Weighted ALS pairs (INS-062) ──")
        real_contacts = ["view_phone", "contact_chat", "contact_zalo", "contact_sms"]
        weighted = (
            lf
            .filter(pl.col("is_login") == "login")
            .filter(pl.col("is_contact") == 1)
            .with_columns(
                pl.when(pl.col("event_type").is_in(real_contacts))
                .then(pl.lit(3.0))
                .otherwise(pl.lit(1.0))
                .alias("w")
            )
            .group_by(["user_id", "item_id"])
            .agg(pl.col("w").sum().alias("score"))
            .collect(engine="streaming")
        )
        out = os.path.join(self.cache_dir, "als_weighted_contact.parquet")
        weighted.write_parquet(out)
        logger.info(f"  {out}: {len(weighted):,} pairs, {os.path.getsize(out)/1e6:.1f}MB")

    def _process_snapshot_stats(self, snapshot_path: str):
        """
        F-012/F-031: Aggregate fact_listing_snapshot into per-item features.
        - item_avg_views_7d, item_avg_contacts_7d
        - item_conversion_rate = contacts / (views + 1)
        - item_trend_score = recent_views / (prior_views + 1)
        - item_is_active = had activity in last 7 days
        """
        logger.info("── [8/8] fact_listing_snapshot → item features ──")
        snap = pl.scan_parquet(os.path.join(snapshot_path, "*.parquet"))
        max_date = snap.select(pl.col("date").max()).collect().item()
        d7 = max_date - timedelta(days=7)
        d30 = max_date - timedelta(days=30)

        # Recent 7d stats
        recent = (
            snap.filter(pl.col("date") >= d7)
            .group_by("item_id").agg([
                pl.col("views_24h").mean().cast(pl.Float32).alias("item_avg_views_7d"),
                pl.col("contacts_24h").mean().cast(pl.Float32).alias("item_avg_contacts_7d"),
            ])
            .collect()
        )

        # Prior 8-30d stats for trend
        prior = (
            snap.filter((pl.col("date") >= d30) & (pl.col("date") < d7))
            .group_by("item_id").agg([
                pl.col("views_24h").mean().cast(pl.Float32).alias("prior_views"),
            ])
            .collect()
        )

        stats = recent.join(prior, on="item_id", how="left").with_columns([
            (pl.col("item_avg_contacts_7d").fill_null(0) / (pl.col("item_avg_views_7d").fill_null(0) + 1))
                .cast(pl.Float32).alias("item_conversion_rate"),
            (pl.col("item_avg_views_7d").fill_null(0) / (pl.col("prior_views").fill_null(0) + 1))
                .cast(pl.Float32).alias("item_trend_score"),
            pl.lit(1.0).cast(pl.Float32).alias("item_is_active"),
        ]).drop("prior_views")

        out = os.path.join(self.cache_dir, "snapshot_stats.parquet")
        stats.write_parquet(out)
        logger.info(f"  {out}: {len(stats):,} items, {os.path.getsize(out)/1e6:.1f}MB")

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

