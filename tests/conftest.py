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
        {"user_id": "u1", "session_id": "s1", "event_type": "pageview", "timestamp": now, "dwell_time_sec": 30.0, "date": "2026-04-01", "item_id": "i1"},
        {"user_id": "u1", "session_id": "s1", "event_type": "view_phone", "timestamp": now + timedelta(minutes=5), "dwell_time_sec": 10.0, "date": "2026-04-01", "item_id": "i2"},
    ])
    
    # Bot user (bot1): Velocity abuse (60 events in 1 minute) and zero dwell
    for i in range(60):
        data.append({
            "user_id": "bot1", "session_id": "s2", "event_type": "pageview", 
            "timestamp": now + timedelta(seconds=i), "dwell_time_sec": 0.5, "date": "2026-04-01", "item_id": "i3"
        })
        
    # Bounce user (u2): Dwell < 3s, not a contact event
    data.append({"user_id": "u2", "session_id": "s3", "event_type": "pageview", "timestamp": now, "dwell_time_sec": 1.5, "date": "2026-04-01", "item_id": "i4"})
    
    # Leakage user (u3): Date > 2026-04-09
    data.append({"user_id": "u3", "session_id": "s4", "event_type": "pageview", "timestamp": datetime(2026, 4, 15, 10, 0, 0), "dwell_time_sec": 50.0, "date": "2026-04-15", "item_id": "i5"})
    
    return pl.LazyFrame(data)


@pytest.fixture
def mock_listings() -> pl.LazyFrame:
    """Synthetic listings for testing price outliers and zombie logic."""
    data = [
        # Legit listing
        {
            "item_id": "i1", "category": 1010, "district_name": "D1", "price_vnd": 5_000_000_000.0, 
            "listing_age_days": 10, "views_24h": 50, "contacts_24h": 2, "images_count": 5,
            "area_sqm": 50.0, "bedrooms": 2, "bathrooms": 1, "direction": "East", "legal_status": "Đã có sổ",
            "title": "Chính chủ bán nhà đẹp tại Quận 1", "ad_type": "sell", "furnishing": "Nội thất cơ bản",
            "seller_type": "agent", "city_name": "Hồ Chí Minh"
        },
        # Legitimate expensive listing in D1
        {
            "item_id": "i2", "category": 1010, "district_name": "D1", "price_vnd": 5_500_000_000.0, 
            "listing_age_days": 5, "views_24h": 100, "contacts_24h": 5, "images_count": 8,
            "area_sqm": 60.0, "bedrooms": 2, "bathrooms": 2, "direction": "North", "legal_status": "Đã có sổ",
            "title": "Bán căn hộ cao cấp full nội thất", "ad_type": "sell", "furnishing": "Nội thất đầy đủ",
            "seller_type": "agent", "city_name": "Hồ Chí Minh"
        },
        # Price outlier in D1
        {
            "item_id": "i3", "category": 1010, "district_name": "D1", "price_vnd": 500_000_000_000.0, 
            "listing_age_days": 5, "views_24h": 100, "contacts_24h": 5, "images_count": 8,
            "area_sqm": 1000.0, "bedrooms": 10, "bathrooms": 10, "direction": "South", "legal_status": "Sổ hồng riêng",
            "title": "Biệt thự siêu sang chảnh vạn người mê", "ad_type": "sell", "furnishing": "Nội thất xa hoa",
            "seller_type": "agent", "city_name": "Hồ Chí Minh"
        },
        # Zombie listing
        {
            "item_id": "i4", "category": 1010, "district_name": "D1", "price_vnd": 4_000_000_000.0, 
            "listing_age_days": 100, "views_24h": 2, "contacts_24h": 0, "images_count": 5,
            "area_sqm": 40.0, "bedrooms": 1, "bathrooms": 1, "direction": "West", "legal_status": None,
            "title": "Nhà nát giá rẻ", "ad_type": "sell", "furnishing": None,
            "seller_type": "private", "city_name": "Hồ Chí Minh"
        },
        # Naked listing
        {
            "item_id": "i5", "category": 1010, "district_name": "D1", "price_vnd": 3_000_000_000.0, 
            "listing_age_days": 10, "views_24h": 50, "contacts_24h": 1, "images_count": 0,
            "area_sqm": 35.0, "bedrooms": 1, "bathrooms": 1, "direction": None, "legal_status": None,
            "title": "Cần bán gấp", "ad_type": "sell", "furnishing": None,
            "seller_type": "private", "city_name": "Hồ Chí Minh"
        },
    ]
    return pl.LazyFrame(data)
