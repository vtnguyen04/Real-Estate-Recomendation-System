import polars as pl
from src.core.base import BaseFeatureExtractor

class ItemPopularityExtractor(BaseFeatureExtractor):
    """
    Extracts item hotness/popularity features from fact_listing_snapshot 
    and fact_post_contact_interactions.
    """
    def __init__(self, name: str = "item_popularity_extractor", **kwargs):
        super().__init__(name=name, **kwargs)

    def _validate_input(self, data: pl.LazyFrame) -> None:
        pass

    def _compute_features(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Assumes `data` is from fact_listing_snapshot or fact_post_contact_interactions.
        We aggregate over all time (or a time window) to get overall item popularity.
        """
        # If it's the snapshot table, it has views_24h, contacts_24h
        if "views_24h" in data.collect_schema().names():
            return data.group_by("item_id").agg([
                pl.col("views_24h").sum().alias("total_views"),
                pl.col("contacts_24h").sum().alias("total_contacts"),
                pl.col("listing_age_days").max().alias("current_age_days")
            ])
        # If it's the post contact interaction table
        elif "lead_count" in data.collect_schema().names():
            return data.group_by("item_id").agg([
                pl.col("adview_count").sum().alias("total_adviews"),
                pl.col("lead_count").sum().alias("total_leads"),
                pl.col("chat_message_count").sum().alias("total_chat_messages")
            ])
        else:
            # Fallback or raise error
            return data

    def _post_process(self, features: pl.LazyFrame) -> pl.LazyFrame:
        # Calculate conversion rate if fields exist
        schema_names = features.collect_schema().names()
        
        if "total_views" in schema_names and "total_contacts" in schema_names:
            features = features.with_columns([
                (pl.col("total_contacts") / (pl.col("total_views") + 1)).alias("contact_conversion_rate")
            ])
            
        if "total_adviews" in schema_names and "total_leads" in schema_names:
            features = features.with_columns([
                (pl.col("total_leads") / (pl.col("total_adviews") + 1)).alias("lead_conversion_rate")
            ])
            
        return features.fill_null(0.0)
