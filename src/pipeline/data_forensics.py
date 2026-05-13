"""
Data Forensics Pipeline.
Responsible for deep cleaning, bot detection, price outlier removal, and zombie listing filtering.
Implements the exact business heuristics discovered in the Datathon 2026 strategies.
"""
import polars as pl
from datetime import timedelta
import numpy as np
from typing import List
from src.core.base import BaseRule, RecommendationContext

class FutureLeakageGuard(BaseRule):
    """
    [CRITICAL] Strictly filters out any data occurring after the competition cutoff date.
    Ensures zero future-leakage in train/val sets as per Datathon rules.
    """
    def __init__(self, cutoff_date: str = '2026-04-09'):
        super().__init__(name="future_leakage_guard", is_hard_filter=True)
        self.priority = 100
        self.cutoff_date = cutoff_date

    def apply(self, items: pl.LazyFrame, context: RecommendationContext = None) -> pl.LazyFrame:
        schema = items.collect_schema().names()
        if "date" in schema:
            return items.filter(pl.col('date') <= self.cutoff_date)
        return items


class BotActivityFilter(BaseRule):
    """
    Advanced Bot Detection using multi-factor behavioral heuristics.
    
    RED FLAGS for bot detection (From Datathon Insights):
    1. velocity_abuse: User views >50 listings in <10 minutes (Velocity > 5/min)
    2. zero_dwell: avg_dwell_time < 1s across all pageviews
    3. non_human_hours: Excessive activity between 2 AM - 5 AM
    
    Implementation scores each user based on flags. Users scoring >= 4 points are blacklisted.
    """
    def __init__(self, bot_score_threshold: int = 4):
        super().__init__(name="bot_activity_filter", is_hard_filter=True)
        self.priority = 90
        self.bot_score_threshold = bot_score_threshold

    def apply(self, events: pl.LazyFrame, context: RecommendationContext = None) -> pl.LazyFrame:
        # This rule applies primarily to fact_user_events
        schema = events.collect_schema().names()
        if "user_id" not in schema or "session_id" not in schema:
            return events
            
        # Identify correct timestamp and dwell time columns
        ts_col = "timestamp" if "timestamp" in schema else "event_ts" if "event_ts" in schema else None
        dwell_col = "dwell_time_sec" if "dwell_time_sec" in schema else "page_dwell_time_sec" if "page_dwell_time_sec" in schema else None
        
        if not ts_col:
            return events

        # Group by user_id to compute bot signals
        user_stats = events.group_by('user_id').agg([
            pl.len().alias('total_events'),
            pl.col(ts_col).max().alias('max_ts'),
            pl.col(ts_col).min().alias('min_ts'),
            (pl.col(dwell_col).mean()).alias('avg_dwell') if dwell_col else pl.lit(5.0).alias('avg_dwell'),
            # Check for non-human hours (2 AM - 5 AM)
            (pl.col(ts_col).dt.hour().is_between(2, 5)).sum().alias('night_events')
        ])
        
        # Compute derived signals
        # Session minutes = (max_ts - min_ts) in seconds / 60
        user_stats = user_stats.with_columns([
            ((pl.col('max_ts') - pl.col('min_ts')).dt.total_seconds() / 60.0).alias('session_minutes')
        ])
        
        user_stats = user_stats.with_columns([
            (pl.col('total_events') / (pl.col('session_minutes') + 1.0)).alias('velocity'),
            (pl.col('night_events') / (pl.col('total_events') + 1.0)).alias('night_ratio')
        ])
        
        # Calculate Bot Score based on Red Flags
        user_stats = user_stats.with_columns([
            (
                # Flag 1: High velocity (Velocity > 5/min) -> 3 points
                (pl.col('velocity') > 5.0).cast(pl.Int32) * 3 +
                # Flag 2: Zero engagement (avg dwell < 1s) -> 2 points
                (pl.col('avg_dwell') < 1.0).cast(pl.Int32) * 2 +
                # Flag 3: Nocturnal activity (> 80% night events) -> 1 point
                (pl.col('night_ratio') > 0.8).cast(pl.Int32) * 1
            ).alias('bot_score')
        ])
        
        # Identify legitimate users (score < 4)
        legit_users = user_stats.filter(pl.col('bot_score') < self.bot_score_threshold).select('user_id')
        
        # Join back to filter out bots
        return events.join(legit_users, on='user_id', how='inner')


class BounceSessionFilter(BaseRule):
    """
    Removes accidental clicks and "bounce" interactions:
    - Dwell time < 3 seconds AND event is not a contact/lead conversion.
    This distills the implicit feedback (pageviews) down to only 'high-intent' signals.
    """
    def __init__(self, min_valid_dwell_sec: float = 3.0):
        super().__init__(name="bounce_session_filter", is_hard_filter=True)
        self.priority = 80
        self.min_valid_dwell_sec = min_valid_dwell_sec

    def apply(self, events: pl.LazyFrame, context: RecommendationContext = None) -> pl.LazyFrame:
        schema = events.collect_schema().names()
        dwell_col = "dwell_time_sec" if "dwell_time_sec" in schema else "page_dwell_time_sec" if "page_dwell_time_sec" in schema else None
        
        if not dwell_col or "event_type" not in schema:
            return events
            
        positive_events = ['view_phone', 'contact_chat', 'contact_zalo', 'contact_sms', 'lead']
        
        # Keep if dwell >= 3 OR event is a high intent contact OR dwell is null (implicit assumption)
        return events.filter(
            (pl.col(dwell_col) >= self.min_valid_dwell_sec) |
            (pl.col('event_type').is_in(positive_events)) |
            (pl.col(dwell_col).is_null())
        )


class PriceOutlierFilter(BaseRule):
    """
    Detects and filters out "junk" listings with absurd pricing using robust z-scores.
    Grouped by category and district to avoid dropping naturally expensive items.
    Prevents ML models from learning false price anchoring behaviors.
    """
    def __init__(self, z_score_threshold: float = 3.5):
        super().__init__(name="price_outlier_filter", is_hard_filter=True)
        self.priority = 70
        self.z_score_threshold = z_score_threshold

    def apply(self, items: pl.LazyFrame, context: RecommendationContext = None) -> pl.LazyFrame:
        schema = items.collect_schema().names()
        if "price_vnd" not in schema or "category" not in schema or "district_name" not in schema:
            return items

        # Calculate robust median and MAD per district & category
        items = items.with_columns([
            pl.col('price_vnd').median().over(['category', 'district_name']).alias('median_price')
        ])
        
        items = items.with_columns([
            (pl.col('price_vnd') - pl.col('median_price')).abs().median().over(['category', 'district_name']).alias('mad_price')
        ])

        # Robust z-score calculation (0.6745 relates MAD to standard deviation)
        items = items.with_columns([
            (0.6745 * (pl.col('price_vnd') - pl.col('median_price')) / (pl.col('mad_price') + 1e-6)).abs().alias('robust_z_score')
        ])

        # Filter and clean up
        filtered = items.filter(pl.col('robust_z_score') <= self.z_score_threshold)
        return filtered.drop(['median_price', 'mad_price', 'robust_z_score'])


class ZombieListingFilter(BaseRule):
    """
    Removes listings that are functionally dead (likely sold but not unlisted).
    Insight: Listings > 60 days old with declining views (<5 in 24h) and 0 recent contacts are "Zombies".
    """
    def __init__(self, max_age_days: int = 60, min_recent_views: int = 5):
        super().__init__(name="zombie_listing_filter", is_hard_filter=True)
        self.priority = 60
        self.max_age_days = max_age_days
        self.min_recent_views = min_recent_views

    def apply(self, data: pl.LazyFrame, context: RecommendationContext = None) -> pl.LazyFrame:
        """
        Applies zombie logic conditionally if snapshot metrics are present, 
        otherwise falls back to basic structural integrity checks.
        """
        schema = data.collect_schema().names()
        
        # If we have snapshot stats joined (Advanced Rule)
        if "listing_age_days" in schema and "views_24h" in schema and "contacts_24h" in schema:
            zombie_condition = (
                (pl.col('listing_age_days') > self.max_age_days) &
                (pl.col('views_24h') < self.min_recent_views) &
                (pl.col('contacts_24h') == 0)
            )
            data = data.filter(~zombie_condition)
            
        # Fallback structural "naked listing" check (Basic Rule)
        if "images_count" in schema:
            data = data.filter(pl.col("images_count") >= 1)
            
        return data


class DataForensicsPipeline:
    """
    Orchestrates the execution of all forensic rules sequentially.
    Sorted by priority to ensure heavy filters (Leakage, Bots) run before complex aggregations.
    """
    def __init__(self, rules: List[BaseRule] = None):
        if rules is None:
            self.rules = sorted([
                FutureLeakageGuard(),
                BotActivityFilter(),
                BounceSessionFilter(),
                PriceOutlierFilter(),
                ZombieListingFilter()
            ], key=lambda x: x.priority, reverse=True)
        else:
            self.rules = sorted(rules, key=lambda x: x.priority, reverse=True)

    def clean(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Runs the LazyFrame through the pipeline of applicable rules.
        Polars query optimizer will fuse these operations efficiently without materializing.
        """
        for rule in self.rules:
            data = rule.apply(data)
        return data
