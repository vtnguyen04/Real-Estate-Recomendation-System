import pytest
import polars as pl
from datetime import datetime, timedelta

@pytest.fixture
def mock_user_events() -> pl.LazyFrame:
    """Synthetic user events for testing bot, bounce, and leakage filters."""
    now = datetime(2026, 4, 1, 10, 0, 0)
    data = []
    
    # Legit user (u1): 2 events, valid dwell, normal velocity
    data.extend([
        {"user_id": "u1", "session_id": "s1", "event_type": "pageview", "timestamp": now, "dwell_time_sec": 30.0, "date": "2026-04-01"},
        {"user_id": "u1", "session_id": "s1", "event_type": "view_phone", "timestamp": now + timedelta(minutes=5), "dwell_time_sec": 10.0, "date": "2026-04-01"},
    ])
    
    # Bot user (bot1): Velocity abuse (60 events in 1 minute) and zero dwell
    for i in range(60):
        data.append({
            "user_id": "bot1", "session_id": "s2", "event_type": "pageview", 
            "timestamp": now + timedelta(seconds=i), "dwell_time_sec": 0.5, "date": "2026-04-01"
        })
        
    # Bounce user (u2): Dwell < 3s, not a contact event
    data.append({"user_id": "u2", "session_id": "s3", "event_type": "pageview", "timestamp": now, "dwell_time_sec": 1.5, "date": "2026-04-01"})
    
    # Leakage user (u3): Date > 2026-04-09
    data.append({"user_id": "u3", "session_id": "s4", "event_type": "pageview", "timestamp": datetime(2026, 4, 15, 10, 0, 0), "dwell_time_sec": 50.0, "date": "2026-04-15"})
    
    return pl.LazyFrame(data)


@pytest.fixture
def mock_listings() -> pl.LazyFrame:
    """Synthetic listings for testing price outliers and zombie logic."""
    data = [
        # Legit listing
        {"item_id": "i1", "category": 1010, "district_name": "D1", "price_vnd": 5_000_000_000.0, "listing_age_days": 10, "views_24h": 50, "contacts_24h": 2, "images_count": 5},
        # Legitimate expensive listing in D1
        {"item_id": "i2", "category": 1010, "district_name": "D1", "price_vnd": 5_500_000_000.0, "listing_age_days": 5, "views_24h": 100, "contacts_24h": 5, "images_count": 8},
        # Price outlier in D1 (Absurdly high compared to median)
        {"item_id": "i3", "category": 1010, "district_name": "D1", "price_vnd": 500_000_000_000.0, "listing_age_days": 5, "views_24h": 100, "contacts_24h": 5, "images_count": 8},
        # Zombie listing (Age > 60, views < 5, 0 contacts)
        {"item_id": "i4", "category": 1010, "district_name": "D1", "price_vnd": 4_000_000_000.0, "listing_age_days": 100, "views_24h": 2, "contacts_24h": 0, "images_count": 5},
        # Naked listing (0 images, fallback test)
        {"item_id": "i5", "category": 1010, "district_name": "D1", "price_vnd": 3_000_000_000.0, "listing_age_days": 10, "views_24h": 50, "contacts_24h": 1, "images_count": 0},
    ]
    return pl.LazyFrame(data)
