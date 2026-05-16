"""
Specific Data Loaders for Datathon 2026.
Handles loading `dim_listing` and `fact_user_events`.
"""
import polars as pl
from typing import Optional, List, Dict, Any
from src.data.loaders.base_loader import ParquetDataLoader

class ListingDataLoader(ParquetDataLoader):
    """Loader for dim_listing"""
    def __init__(self, project_id: str = None, data_path: str = "data/raw/train/dim_listing/"):
        super().__init__(data_path=data_path, table_name="dim_listing", project_id=project_id)

    def get_schema(self) -> dict:
        return {
            'item_id': str, 'seller_id': str, 'category': int, 'title': str,
            'seller_type': str, 'ad_type': str, 'ad_status': str, 'area_sqm': float,
            'bedrooms': float, 'bathrooms': float, 'floors': float, 'width_m': float,
            'direction': str, 'legal_status': str, 'house_type': str, 'furnishing': str,
            'city_name': str, 'district_name': str, 'ward_name': str, 'project_id': str,
            'price_bucket': str, 'images_count': float, 'posted_date': str, 'expected_expired_date': str
        }

class ListingSnapshotLoader(ParquetDataLoader):
    """Loader for fact_listing_snapshot"""
    def __init__(self, project_id: str = None, data_path: str = "data/raw/train/fact_listing_snapshot/"):
        super().__init__(data_path=data_path, table_name="fact_listing_snapshot", project_id=project_id)

    def get_schema(self) -> dict:
        return {
            'item_id': str, 'date': str, 'views_24h': float,
            'contacts_24h': float, 'listing_age_days': float
        }

class PostContactInteractionsLoader(ParquetDataLoader):
    """Loader for fact_post_contact_interactions"""
    def __init__(self, project_id: str = None, data_path: str = "data/raw/train/fact_post_contact_interactions/"):
        super().__init__(data_path=data_path, table_name="fact_post_contact_interactions", project_id=project_id)

    def get_schema(self) -> dict:
        return {
            'user_id': str, 'item_id': str, 'date': str, 'adview_count': float,
            'lead_count': float, 'chat_message_count': float, 'chat_turn_count': float,
            'chat_lead': float, 'purchased': bool, 'category': int
        }

class FactUserEventsLoader(ParquetDataLoader):
    """Loader for fact_user_events"""
    def __init__(self, project_id: str = None, data_path: str = "data/raw/train/fact_user_events/"):
        super().__init__(data_path=data_path, table_name="fact_user_events", project_id=project_id)

    def _load_impl(self, columns: Optional[List[str]], filters: Optional[dict]) -> pl.LazyFrame:
        lf = super()._load_impl(columns, filters)
        # Base validation: drop rows without user_id or item_id
        lf = lf.drop_nulls(subset=['user_id', 'item_id'])
        return lf

    def get_schema(self) -> dict:
        return {
            'is_login': str, 'user_id': str, 'session_id': str, 'event_id': str,
            'item_id': str, 'city_name': str, 'category': int, 'event_type': str,
            'query': str, 'event_ts': str, 'surface': str, 'position': float,
            'device': str, 'dwell_time_sec': float, 'is_contact': int, 'date': str
        }
