import numpy as np
from typing import Dict, List
from src.features.base import BaseHeuristicExtractor
from src.features.feature_context import FeatureContext
from src.models.candidates.als_recommender import ALSRecommender

class CollaborativeExtractor(BaseHeuristicExtractor):
    """
    Extracts features from Collaborative Filtering models (ALS & I2I).
    Computes 'score_als', 'als_raw_score', and 'score_i2i'.
    """
    def __init__(self, als_model: ALSRecommender):
        self.als_model = als_model
        
        # State to hold pre-fetched batch predictions
        self.als_recs_cache = {}
        self.item_similar_cache = {}

    def prefetch_batch(self, users: List[str], context: FeatureContext):
        """Pre-fetches matrix recommendations in batch for speed."""
        self.als_recs_cache.clear()
        self.item_similar_cache.clear()
        
        builder = self.als_model.matrix_builder
        if not builder:
            return
            
        # ALS Recommends
        warm_indices = []
        warm_uids = []
        for uid in users:
            idx = builder.get_user_idx(uid)
            if idx != -1:
                warm_indices.append(idx)
                warm_uids.append(uid)
                
        if warm_indices and self.als_model.user_items is not None:
            # Process in chunks to avoid OOM (full batch needs ~67GB intermediate memory)
            chunk_size = 500
            for start in range(0, len(warm_indices), chunk_size):
                chunk_idx = warm_indices[start:start + chunk_size]
                chunk_uids = warm_uids[start:start + chunk_size]
                idx_arr = np.array(chunk_idx, dtype=np.int32)
                ids_mat, scores_mat = self.als_model.model.recommend(
                    idx_arr, self.als_model.user_items[idx_arr], N=300, filter_already_liked_items=False
                )
                for uid, row, srow in zip(chunk_uids, ids_mat, scores_mat):
                    self.als_recs_cache[uid] = [
                        (builder.get_item_id(int(i)), float(s))
                        for i, s in zip(row, srow)
                        if builder.get_item_id(int(i)) in context.valid_items
                    ]
                
        # I2I Similarity (DISABLED FOR SPEED)
        # seed_items = set([it for u in users for it in context.user_prev.get(u, [])[:5]])
        # for it in seed_items:
        #     idx = builder.get_item_idx(it)
        #     if idx != -1:
        #         try:
        #             sim_ids, sim_scores = self.als_model.model.similar_items(idx, N=20)
        #             self.item_similar_cache[it] = [
        #                 (builder.get_item_id(int(i)), float(s)) 
        #                 for i, s in zip(sim_ids[1:], sim_scores[1:])
        #             ]
        #         except: pass

    def extract_scores(self, uid: str, context: FeatureContext, features_dict: Dict[str, Dict[str, float]]):
        pref_city, pref_cat = context.user_prefs.get(uid, (None, None))
        
        # ALS scoring
        if uid in self.als_recs_cache:
            for rank, (it, score) in enumerate(self.als_recs_cache[uid][:300]):
                boost = 3.0 if context.item_city.get(it) == pref_city else 1.0
                als_heuristic = (50.0 - rank) * boost if rank < 30 else max(0.1, 20.0 - (rank - 30) * 0.1) * boost
                features_dict[it]["score_als"] = features_dict[it].get("score_als", 0.0) + als_heuristic
                features_dict[it]["als_raw_score"] = float(score)

        # I2I scoring
        for seed in context.user_prev.get(uid, [])[:5]:
            if seed in self.item_similar_cache:
                for rank, (it, score) in enumerate(self.item_similar_cache[seed][:15]):
                    boost = 2.0 if context.item_city.get(it) == pref_city else 0.5
                    features_dict[it]["score_i2i"] = features_dict[it].get("score_i2i", 0.0) + (20.0 - rank) * boost
