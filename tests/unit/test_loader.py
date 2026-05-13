import pytest
import polars as pl
from unittest.mock import patch
from src.data.loader import (
    ListingDataLoader,
    ListingSnapshotLoader,
    PostContactInteractionsLoader,
    FactUserEventsLoader
)

@patch('src.data.loaders.base_loader.ds.dataset')
@patch('src.data.loaders.base_loader.pl.scan_pyarrow_dataset')
def test_loaders(mock_scan, mock_dataset):
    # Mock return value for scan_parquet
    mock_scan.return_value = pl.LazyFrame({"user_id": ["u1"], "item_id": ["i1"], "event_type": ["view"]})
    
    loader_listing = ListingDataLoader(project_id="test", gcs_path="gs://dummy/")
    assert loader_listing.get_schema() is not None
    res = loader_listing.load()
    assert res is not None
    
    loader_snapshot = ListingSnapshotLoader(project_id="test", gcs_path="gs://dummy/")
    assert loader_snapshot.get_schema() is not None
    
    loader_interactions = PostContactInteractionsLoader(project_id="test", gcs_path="gs://dummy/")
    assert loader_interactions.get_schema() is not None
    
    loader_events = FactUserEventsLoader(project_id="test", gcs_path="gs://dummy/")
    assert loader_events.get_schema() is not None
    
    # Check drop_nulls behavior
    mock_scan.return_value = pl.LazyFrame({"user_id": ["u1", None], "item_id": ["i1", "i2"], "event_type": ["view", "click"]})
    res_events = loader_events.load()
    # It should drop the row with null user_id
    assert res_events.collect().height == 1
