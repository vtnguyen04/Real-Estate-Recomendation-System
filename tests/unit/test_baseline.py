import polars as pl
from src.models.baselines.popularity import PopularityRecommender
from src.core.base import RecommendationContext

def test_popularity_recommender():
    """Ensure baseline extracts the highest weighted interaction metrics accurately."""
    data = pl.LazyFrame([
        {"item_id": "i1", "contacts_24h": 5, "views_24h": 10},   # pop_score = 5*10 + 10 = 60
        {"item_id": "i2", "contacts_24h": 1, "views_24h": 100},  # pop_score = 1*10 + 100 = 110
        {"item_id": "i3", "contacts_24h": 0, "views_24h": 20},   # pop_score = 20
    ])
    
    model = PopularityRecommender(top_k=2)
    model.fit(data)
    
    ctx = RecommendationContext(user_id="u_cold_start", timestamp="2026-04-09", num_recommendations=2)
    recs = model.recommend(ctx)
    
    assert len(recs) == 2
    # i2 has the highest score
    assert recs[0].item_id == "i2"
    assert recs[0].rank == 1
    
    # i1 is second
    assert recs[1].item_id == "i1"
    assert recs[1].rank == 2
