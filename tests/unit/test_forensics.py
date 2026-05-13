import polars as pl
from src.pipeline.data_forensics import (
    FutureLeakageGuard,
    BotActivityFilter,
    BounceSessionFilter,
    PriceOutlierFilter,
    ZombieListingFilter,
    DataForensicsPipeline
)

def test_future_leakage_guard(mock_user_events):
    """Ensure data strictly conforms to the competition cut-off rule."""
    rule = FutureLeakageGuard(cutoff_date='2026-04-09')
    result = rule.apply(mock_user_events).collect()
    
    users = result["user_id"].to_list()
    assert "u3" not in users  # u3 occurred on 2026-04-15
    assert "u1" in users      # u1 occurred on 2026-04-01

def test_bot_activity_filter(mock_user_events):
    """Ensure users exceeding velocity/dwell/volume heuristics are blacklisted."""
    rule = BotActivityFilter(bot_score_threshold=4)
    result = rule.apply(mock_user_events).collect()
    
    users = set(result["user_id"].to_list())
    assert "bot1" not in users  # bot1 has 60 events in 1 min + 0.5s dwell
    assert "u1" in users        # u1 is a legitimate user

def test_bounce_session_filter(mock_user_events):
    """Ensure implicit low-dwell noise is stripped, preserving explicit signals."""
    rule = BounceSessionFilter(min_valid_dwell_sec=3.0)
    result = rule.apply(mock_user_events).collect()
    
    users = set(result["user_id"].to_list())
    assert "u2" not in users  # u2 bounced in 1.5s
    assert "u1" in users      # u1 has long dwell and explicit contact event

def test_price_outlier_filter(mock_listings):
    """Ensure items with absurdly high prices are dropped via robust z-score."""
    rule = PriceOutlierFilter(z_score_threshold=3.5)
    result = rule.apply(mock_listings).collect()
    
    items = set(result["item_id"].to_list())
    assert "i3" not in items  # i3 is an extreme outlier
    assert "i1" in items
    assert "i2" in items

def test_zombie_listing_filter(mock_listings):
    """Ensure technically active but functionally dead listings are removed."""
    rule = ZombieListingFilter(max_age_days=60, min_recent_views=5)
    result = rule.apply(mock_listings).collect()
    
    items = set(result["item_id"].to_list())
    assert "i4" not in items  # i4 is a 100-day-old zombie with no contacts
    assert "i5" not in items  # i5 has 0 images
    assert "i1" in items

def test_pipeline_orchestration(mock_user_events):
    """Ensure rules can be cleanly chained without eager execution faults."""
    pipeline = DataForensicsPipeline([
        FutureLeakageGuard(cutoff_date='2026-04-09'),
        BounceSessionFilter(min_valid_dwell_sec=3.0)
    ])
    result = pipeline.clean(mock_user_events).collect()
    
    users = set(result["user_id"].to_list())
    assert "u3" not in users # Filtered by LeakageGuard
    assert "u2" not in users # Filtered by BounceFilter
    assert "u1" in users     # Passed both
