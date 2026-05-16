import pytest
import polars as pl
import os
import tempfile
from unittest.mock import patch

from src.data.loader import (
    ListingDataLoader,
    FactUserEventsLoader,
    ListingSnapshotLoader,
    PostContactInteractionsLoader
)

@pytest.fixture
def dummy_parquet_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a dummy parquet file
        df = pl.DataFrame({
            "item_id": ["i1", "i2"],
            "category": [1010, 1020],
            "city_name": ["Hanoi", "HCMC"],
            "user_id": ["u1", None],
            "event_type": ["view", "click"],
            "date": ["2026-04-01", "2026-04-10"]
        })
        df.write_parquet(os.path.join(tmpdirname, "data.parquet"))
        yield tmpdirname

@patch('src.data.loaders.base_loader.GCSDataLoader._authenticate')
def test_loaders_real_file(mock_auth, dummy_parquet_dir):
    # Test ListingDataLoader
    loader = ListingDataLoader(project_id="test", gcs_path=dummy_parquet_dir)
    assert loader.get_schema() is not None
    res = loader.load(use_cache=False)
    df = res.collect()
    assert len(df) == 2
    
    # Test filters
    res_filtered = loader.load(filters={"item_id": "i1"}, use_cache=False)
    assert len(res_filtered.collect()) == 1
    
    # Test list filters
    res_list = loader.load(filters={"item_id": ["i1", "i2"]}, use_cache=False)
    assert len(res_list.collect()) == 2
    
    # Test FactUserEventsLoader with drop_nulls
    events_loader = FactUserEventsLoader(project_id="test", gcs_path=dummy_parquet_dir)
    assert events_loader.get_schema() is not None
    res_events = events_loader.load(use_cache=False)
    df_events = res_events.collect()
    # It drops the row with null user_id
    assert len(df_events) == 1
    
    # Test PostContactInteractionsLoader
    interactions_loader = PostContactInteractionsLoader(project_id="test", gcs_path=dummy_parquet_dir)
    assert interactions_loader.get_schema() is not None
    res_interactions = interactions_loader.load(use_cache=False)
    assert len(res_interactions.collect()) == 2
    
    # Test chunked loading
    snapshot_loader = ListingSnapshotLoader(project_id="test", gcs_path=dummy_parquet_dir)
    assert snapshot_loader.get_schema() is not None
    chunks = list(snapshot_loader.load_chunked(chunk_size=1, date_range=("2026-04-01", "2026-04-05")))
    assert len(chunks) == 1
    assert len(chunks[0].collect()) == 1 # Only 2026-04-01
