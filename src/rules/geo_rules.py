import polars as pl
from src.core.base import BaseRule, RecommendationContext
from typing import Dict
import numpy as np

class GeoProximityScoreRule(BaseRule):
    """
    Scores an item based on its geographic proximity (district-level co-viewing similarity)
    to the user's historical viewing preferences.
    Uses Jaccard similarity of districts co-viewed within the same session.
    """
    def __init__(self, name: str = "geo_proximity_scorer", district_similarity_df: pl.DataFrame = None):
        super().__init__(name=name, is_hard_filter=False)
        # We expect a Polars DataFrame with columns: ['district_a', 'district_b', 'jaccard_similarity']
        self.similarity_df = district_similarity_df

    @classmethod
    def build_similarity_matrix(cls, fact_user_events: pl.LazyFrame, dim_listing: pl.LazyFrame) -> pl.DataFrame:
        """
        Builds the district co-occurrence Jaccard similarity matrix from historical sessions.
        J(A,B) = |A∩B| / |A∪B|
        """
        # Join events with listing to get district_name per event
        events_with_geo = fact_user_events.join(
            dim_listing.select(['item_id', 'district_name']).drop_nulls('district_name'),
            on='item_id',
            how='inner'
        )
        
        # Get unique districts per session (Session-District pairs)
        session_districts = events_with_geo.select(['session_id', 'district_name']).unique()
        
        # Self join to find co-occurring districts in the same session
        co_occurrence = session_districts.join(
            session_districts, on='session_id', how='inner'
        ).rename({
            "district_name": "district_a", 
            "district_name_right": "district_b"
        })
        
        # Count co-occurrences
        co_counts = co_occurrence.group_by(['district_a', 'district_b']).agg(
            pl.len().alias('co_view_count')
        )
        
        # Get individual district counts (where A == B)
        # This represents |A| and |B|
        dist_counts = co_counts.filter(pl.col('district_a') == pl.col('district_b')) \
            .select([
                pl.col('district_a').alias('district'), 
                pl.col('co_view_count').alias('total_count')
            ])
            
        # Calculate Jaccard Similarity
        # J(A,B) = |A∩B| / (|A| + |B| - |A∩B|)
        jaccard_df = co_counts.join(
            dist_counts, left_on='district_a', right_on='district'
        ).rename({"total_count": "count_a"}).join(
            dist_counts, left_on='district_b', right_on='district'
        ).rename({"total_count": "count_b"})
        
        jaccard_df = jaccard_df.with_columns(
            (pl.col('co_view_count') / (pl.col('count_a') + pl.col('count_b') - pl.col('co_view_count'))).alias('jaccard_similarity')
        )
        
        return jaccard_df.select(['district_a', 'district_b', 'jaccard_similarity']).collect()

    def apply(self, items: pl.LazyFrame, context: RecommendationContext = None) -> pl.LazyFrame:
        """
        Calculates geo_score based on the user's historical district views.
        Requires 'user_district_history' (struct/list of districts and counts) 
        and 'item_district' (district_name) to be present in `items`.
        """
        # If similarity matrix is not provided, return default score 0.5
        if self.similarity_df is None or self.similarity_df.is_empty():
            return items.with_columns(pl.lit(0.5).alias("geo_score"))
            
        schema_names = items.collect_schema().names()
        if "district_name" not in schema_names or "user_viewed_districts" not in schema_names:
            return items.with_columns(pl.lit(0.5).alias("geo_score"))
            
        # Since user_viewed_districts is a complex struct (e.g. list of dicts), 
        # and we need a weighted average similarity, the most robust way in Polars 
        # without falling back to python map_elements is to explode, join, aggregate.
        
        # We will explode the user's viewed districts, join with similarity, compute weighted avg,
        # then join back.
        
        # 1. Add a temporary row index to group back later
        items_with_idx = items.with_row_index("temp_idx")
        
        # 2. Extract and explode user history
        # Assuming user_viewed_districts is a list of structs: [{'district': 'Q1', 'weight': 5}, ...]
        exploded_history = items_with_idx.select(['temp_idx', 'district_name', 'user_viewed_districts']) \
            .explode('user_viewed_districts')
            
        # Extract district and weight
        user_districts = exploded_history.with_columns([
            pl.col('user_viewed_districts').struct.field('district').alias('hist_district'),
            pl.col('user_viewed_districts').struct.field('weight').alias('hist_weight')
        ]).drop_nulls('hist_district')
        
        # 3. Join with similarity matrix
        # district_a = item's district, district_b = historical district
        similarity_lazy = self.similarity_df.lazy()
        
        scored_districts = user_districts.join(
            similarity_lazy, 
            left_on=['district_name', 'hist_district'],
            right_on=['district_a', 'district_b'],
            how='left'
        )
        
        # If no similarity found (or distinct unknown), use a small base similarity like 0.05
        scored_districts = scored_districts.with_columns(
            pl.col('jaccard_similarity').fill_null(0.05)
        )
        
        # Calculate weighted average
        # weighted sum / sum of weights
        weighted_scores = scored_districts.with_columns(
            (pl.col('jaccard_similarity') * pl.col('hist_weight')).alias('weighted_sim')
        ).group_by('temp_idx').agg([
            (pl.col('weighted_sim').sum() / pl.col('hist_weight').sum()).alias('geo_score')
        ])
        
        # Join back
        final_items = items_with_idx.join(weighted_scores, on='temp_idx', how='left') \
            .with_columns(pl.col('geo_score').fill_null(0.5)) \
            .drop('temp_idx')
            
        return final_items
