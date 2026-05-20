"""
Intent-based Candidate Recommender.
Recommends active items that match the user's recent search intent profile.
"""
import polars as pl
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from src.core.base import BaseRecommender
from src.utils.logging import get_logger

logger = get_logger(__name__)

class IntentRecommender(BaseRecommender):
    def __init__(self, name="intent_recommender", max_items_per_intent=200):
        super().__init__(name=name)
        self.max_items = max_items_per_intent
        self._user_intents: Dict[str, Tuple] = {}
        self._user_fallbacks: Dict[str, Tuple] = {}
        self._intent_to_items: Dict[Tuple, List[str]] = defaultdict(list)
        self._fallback_to_items: Dict[Tuple, List[str]] = defaultdict(list)
        
    def fit(self, pvs: pl.DataFrame, dim_listing: pl.DataFrame, valid_items: Optional[Set[str]] = None) -> "IntentRecommender":
        """
        pvs: DataFrame with [user_id, item_id] of recent pageviews
        dim_listing: DataFrame with item attributes
        valid_items: Only include these active items in the index
        """
        logger.info("Fitting IntentRecommender...")
        
        # Sort dim_listing by posted_date descending so we get freshest items first
        if "posted_date" in dim_listing.columns:
            dim_listing = dim_listing.sort("posted_date", descending=True)
            
        # Build item mapping
        item_meta = {}
        item_fallback = {}
        for r in dim_listing.iter_rows(named=True):
            iid = r["item_id"]
            if valid_items and iid not in valid_items:
                continue
            # We use District + Category + Price Bucket as the intent tuple
            profile = (r.get("district_name"), r.get("category"), r.get("price_bucket"))
            fallback = (r.get("city_name"), r.get("category"))
            
            item_meta[iid] = profile
            item_fallback[iid] = fallback
            
            self._intent_to_items[profile].append(iid)
            self._fallback_to_items[fallback].append(iid)
            
        logger.info(f"Built intent index with {len(self._intent_to_items)} unique intents.")
        logger.info(f"Built fallback index with {len(self._fallback_to_items)} unique fallbacks.")

        # Build user intent profiles
        user_profiles = defaultdict(lambda: defaultdict(int))
        user_fallbacks = defaultdict(lambda: defaultdict(int))
        for r in pvs.iter_rows(named=True):
            uid, iid = r["user_id"], r["item_id"]
            if iid in item_meta:
                user_profiles[uid][item_meta[iid]] += 1
            if iid in item_fallback:
                user_fallbacks[uid][item_fallback[iid]] += 1
                
        # Assign best profile per user
        for uid, profiles in user_profiles.items():
            self._user_intents[uid] = max(profiles.items(), key=lambda x: x[1])[0]
        for uid, fallbacks in user_fallbacks.items():
            self._user_fallbacks[uid] = max(fallbacks.items(), key=lambda x: x[1])[0]
            
        logger.info(f"Extracted intents for {len(self._user_intents)} users.")
        return self
        
    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude: Optional[Set[str]] = None,
        **kwargs,
    ) -> List[str]:
        if user_id not in self._user_intents:
            return []
            
        intent = self._user_intents[user_id]
        candidates = self._intent_to_items.get(intent, [])
        exclude = set(exclude) if exclude else set()
        
        recs = []
        # Try primary intent first
        for it in candidates:
            if it not in exclude:
                recs.append(it)
                exclude.add(it)
                if len(recs) >= k:
                    return recs
                    
        # Fill remaining with fallback
        if user_id in self._user_fallbacks:
            fallback = self._user_fallbacks[user_id]
            fallback_cands = self._fallback_to_items.get(fallback, [])
            for it in fallback_cands:
                if it not in exclude:
                    recs.append(it)
                    exclude.add(it)
                    if len(recs) >= k:
                        break
                        
        return recs
        
    def save(self, path: str) -> None:
        pass
        
    def load(self, path: str) -> "IntentRecommender":
        return self
