import pytest
import polars as pl
from unittest.mock import patch, MagicMock

from src.evaluation.cross_validator import TimeBasedSplitter
from datetime import datetime, timedelta

def test_time_series_cross_validator():
    # Create sample DataFrame
    df = pl.DataFrame({
        "user_id": ["u1", "u2", "u3", "u1", "u2"],
        "item_id": ["i1", "i2", "i3", "i4", "i5"],
        "timestamp": [
            datetime(2026, 4, 1),
            datetime(2026, 4, 2),
            datetime(2026, 4, 8),
            datetime(2026, 4, 10),
            datetime(2026, 4, 11)
        ]
    })
    
    # 3 validation days from 2026-04-11 means >= 2026-04-08
    validator = TimeBasedSplitter(validation_days=3, timestamp_col="timestamp")
    
    # Test split
    train_lf, val_lf = validator.split(df.lazy())
    
    train_df = train_lf.collect()
    val_df = val_lf.collect()
    
    split_point = datetime(2026, 4, 11) - timedelta(days=3)
    
    # Train should have dates < split_point
    assert len(train_df) == 2
    assert train_df["timestamp"].max() < split_point
    
    # Val should have dates >= split_point
    assert len(val_df) == 3
    assert val_df["timestamp"].min() >= split_point

from src.evaluation.health_metrics import HealthMetrics

def test_health_metrics_compute_diversity():
    metrics = HealthMetrics()
    items = [
        {"category": 1010, "city_name": "Hanoi", "district_name": "Hoan Kiem"},
        {"category": 1020, "city_name": "HCMC", "district_name": "Q1"},
        {"category": 1030, "city_name": "Da Nang", "district_name": "Hai Chau"}
    ]
    diversity = metrics.compute_diversity(items)
    assert diversity > 0.0

def test_health_metrics_compute_fairness():
    metrics = HealthMetrics()
    items = [
        {"category": 1010, "seller_type": "agent"},
        {"category": 1020, "seller_type": "individual"},
        {"category": 1030, "seller_type": "agent"}
    ]
    fairness = metrics.compute_fairness(items)
    assert fairness > 0.0

def test_health_metrics_compute_freshness():
    metrics = HealthMetrics()
    items = [
        {"listing_age_days": 1},
        {"listing_age_days": 5},
        {"listing_age_days": 10}
    ]
    freshness = metrics.compute_freshness(items)
    assert freshness > 0.0

