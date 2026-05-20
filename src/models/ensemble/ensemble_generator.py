import polars as pl
from typing import List, Dict, Tuple, Set, Optional

from src.models.candidates.light_als import LightALSRecommender
from src.models.candidates.segment_popularity import SegmentPopularityRecommender

class EnsembleCandidateGenerator:
    """
    SOLID Candidate Generator: Orchestrates multiple recall sources.
    Delegates the responsibility of mixing ALS, View ALS, and SegPop candidates.
    Used by both TrainingPipeline and evaluate.py to ensure DRY.
    """
    def __init__(
        self,
        als: LightALSRecommender,
        als_view: LightALSRecommender,
        segpop: SegmentPopularityRecommender,
        n_cand_als: int = 800,
        n_cand_view_als: int = 600,
        n_cand_segpop: int = 100,
    ):
        self.als = als
        self.als_view = als_view
        self.segpop = segpop
        self.n_cand_als = n_cand_als
        self.n_cand_view_als = n_cand_view_als
        self.n_cand_segpop = n_cand_segpop

    @staticmethod
    def _normalize_scores(pairs: list) -> dict:
        if not pairs:
            return {}
        scores = [s for _, s in pairs]
        mn, mx = min(scores), max(scores)
        if mx == mn:
            return {it: 1.0 for it, _ in pairs}
        return {it: (s - mn) / (mx - mn) for it, s in pairs}

    def generate_batch(
        self,
        users: List[str],
        user_prefs: Dict[str, Tuple[Optional[str], Optional[int]]],
        valid_items: Set[str],
        pos_set: Optional[Dict[str, Set[str]]] = None,
        label_col: bool = False,
    ) -> Tuple[pl.DataFrame, Dict[str, List[str]]]:
        """
        Generate candidates using SoA (Struct of Arrays) for memory efficiency.
        
        Returns:
            df_candidates: pl.DataFrame containing all candidates with metadata
            user_cands_dict: Dict mapping user_id to list of candidate item_ids
        """
        als_recs = self.als.recommend_batch(
            users, n=self.n_cand_als, filter_already_liked=False,
            valid_items=valid_items, return_scores=True,
        )
        view_recs = self.als_view.recommend_batch(
            users, n=self.n_cand_view_als, filter_already_liked=False,
            valid_items=valid_items, return_scores=True,
        )

        uids, item_ids = [], []
        s_als, s_view, s_seg = [], [], []
        is_als, is_view, is_seg = [], [], []
        labels = []
        user_cands_dict = {}

        for uid in users:
            pref_city, pref_cat = user_prefs.get(uid, (None, None))
            als_pairs  = als_recs.get(uid, [])
            view_pairs = view_recs.get(uid, [])
            als_norm   = self._normalize_scores(als_pairs)
            view_norm  = self._normalize_scores(view_pairs)

            seen = set()
            user_pos = pos_set.get(uid, set()) if pos_set else set()

            for it, _ in als_pairs:
                if it not in seen:
                    uids.append(uid); item_ids.append(it)
                    s_als.append(als_norm.get(it, 0.0)); s_view.append(view_norm.get(it, 0.0)); s_seg.append(0.0)
                    is_als.append(1); is_view.append(0); is_seg.append(0)
                    if label_col: labels.append(1 if it in user_pos else 0)
                    seen.add(it)

            for it, _ in view_pairs:
                if it not in seen:
                    uids.append(uid); item_ids.append(it)
                    s_als.append(0.0); s_view.append(view_norm.get(it, 0.0)); s_seg.append(0.0)
                    is_als.append(0); is_view.append(1); is_seg.append(0)
                    if label_col: labels.append(1 if it in user_pos else 0)
                    seen.add(it)

            segpop_items = self.segpop.get_segment_items(
                pref_city=pref_city, pref_cat=pref_cat, k=self.n_cand_segpop
            )
            for rank, it in enumerate(segpop_items):
                if it not in seen and it in valid_items:
                    uids.append(uid); item_ids.append(it)
                    s_als.append(0.0); s_view.append(0.0)
                    s_seg.append(float(self.n_cand_segpop - rank) / self.n_cand_segpop)
                    is_als.append(0); is_view.append(0); is_seg.append(1)
                    if label_col: labels.append(1 if it in user_pos else 0)
                    seen.add(it)

            user_cands_dict[uid] = list(seen)

        if not uids:
            return pl.DataFrame(), {}

        data = {
            "user_id": uids, "item_id": item_ids,
            "score_als": s_als, "score_view_als": s_view, "score_segpop": s_seg,
            "is_from_als": is_als, "is_from_view_als": is_view, "is_from_segpop": is_seg
        }
        if label_col:
            data["label"] = labels

        return pl.DataFrame(data), user_cands_dict
