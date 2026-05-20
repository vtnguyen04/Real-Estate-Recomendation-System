from typing import Dict
from src.features.base import BaseHeuristicExtractor
from src.features.feature_context import FeatureContext

class SegmentPopularityExtractor(BaseHeuristicExtractor):
    """
    Extracts global and segment-level popularity features.
    Computes 'score_segpop'.
    """
    def extract_scores(self, uid: str, context: FeatureContext, features_dict: Dict[str, Dict[str, float]]):
        pref_city, pref_cat = context.user_prefs.get(uid, (None, None))
        
        seg_rank = 0
        if pref_city and pref_cat:
            for it in context.cc_lists.get((pref_city, pref_cat), [])[:30]:
                features_dict[it]["score_segpop"] += max(3.0 - seg_rank * 0.01, 0.1)
                seg_rank += 1
                
        if pref_city:
            for it in context.city_lists.get(pref_city, [])[:20]:
                features_dict[it]["score_segpop"] += max(2.0 - seg_rank * 0.01, 0.1)
                seg_rank += 1
                
        for it in context.global_top[:20]:
            features_dict[it]["score_segpop"] += 1.0
