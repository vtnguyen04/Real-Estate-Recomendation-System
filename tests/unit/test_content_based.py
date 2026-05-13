import pytest
import polars as pl
from src.models.candidates.content_based import ContentRecommender
from src.core.base import RecommendationContext

def test_content_recommender():
    """Ensure TF-IDF text similarity logic extracts relevant semantic recommendations."""
    items = pl.LazyFrame([
        {"item_id": "i1", "title": "Luxury Apartment in City Center"},
        {"item_id": "i2", "title": "Cheap Room for Rent"},
        {"item_id": "i3", "title": "Luxury Villa near City Center"},
    ])
    
    interactions = pl.LazyFrame([
        {"user_id": "u1", "item_id": "i1"} # user likes luxury city center
    ])
    
    model = ContentRecommender(top_k=2)
    # Train expects items as main arg, interactions as kwargs
    model.fit(items, interactions=interactions)
    
    ctx = RecommendationContext(user_id="u1", timestamp="2026-04-09", num_recommendations=1)
    recs = model.recommend(ctx)
    
    # Assert CF returns results
    assert recs.collect().height == 1
    # i1 is in history, so it must be suppressed (-1.0).
    # i3 shares 'Luxury', 'City', 'Center' -> High Cosine Similarity vs i2
    assert recs.collect()[0, "item_id"] == "i3"

def test_content_recommender_cold_start():
    items = pl.LazyFrame([
        {"item_id": "i1", "title": "Luxury Apartment in City Center"}
    ])
    
    # Empty interactions
    interactions = pl.LazyFrame({"user_id": [], "item_id": []}, schema={"user_id": pl.Utf8, "item_id": pl.Utf8})
    
    model = ContentRecommender(top_k=2)
    model.fit(items, interactions=interactions)
    
    ctx = RecommendationContext(user_id="u_unknown", timestamp="2026-04-09", num_recommendations=1)
    recs = model.recommend(ctx)
    
    # Expect empty fallback trigger
    assert recs.collect().height == 0
