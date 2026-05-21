"""
CascadeCandidateGenerator — Multi-source cascade with priority-based slot filling.

Single Responsibility: Orchestrate multiple candidate sources in priority order,
filling recommendation slots from the strongest signal first.

Open/Closed Principle: New sources can be added by appending to the sources list
without modifying existing logic.

Architecture (Round 18 — optimized for Recall@10, V6 base + new generators):
    Priority 1: PageviewReplay     — BEST top-10 precision    (Recall@10=0.10)
    Priority 2: IntentRecommender  — district+cat+price match (Recall@200=0.1140)
    Priority 3: CoContact          — co-contact expansion
    Priority 4: LightALS           — wide coverage CF          (Recall@200=0.1749)
    Priority 5: UserKNN            — neighbor-based CF         (Recall@200=0.0862)
    Priority 6: SellerExpansion    — same-seller items         (Recall@200=0.0302)
    Priority 7: RecentCC           — recent segment popular
    Priority 8: SegPop             — all-time segment popular  (Recall@200=0.0079)

This replaces the EnsembleCandidateGenerator for inference but the old
module is preserved for training compatibility.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

import polars as pl

from src.models.candidates.pageview_replay import PageviewReplayRecommender
from src.models.candidates.cocontact import CoContactRecommender
from src.models.candidates.segment_popularity import SegmentPopularityRecommender
from src.models.candidates.light_als import LightALSRecommender
from src.models.candidates.intent_recommender import IntentRecommender
from src.models.candidates.user_knn import UserKNNRecommender
from src.models.candidates.seller_recommender import SellerExpansionRecommender
from src.core.base import RecommendationContext
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CascadeCandidateGenerator:
    """
    Priority-based cascade candidate generator.

    For each user, fills K recommendation slots from the highest-priority
    source that has data, then falls back to lower priorities.

    This design ensures:
    - Users with rich interaction data get ALS CF signal first
    - Users with pageview intent get district+cat+price match
    - Cold-start users get reasonable segment-popular fallback
    """

    def __init__(
        self,
        pv_replay: Optional[PageviewReplayRecommender] = None,
        cocontact: Optional[CoContactRecommender] = None,
        segpop: Optional[SegmentPopularityRecommender] = None,
        als: Optional[LightALSRecommender] = None,
        als_view: Optional[LightALSRecommender] = None,
        recent_cc: Optional[Dict[Tuple[str, int], List[str]]] = None,
        user_histories: Optional[Dict[str, List[str]]] = None,
        intent_rec: Optional[IntentRecommender] = None,
        user_knn: Optional[UserKNNRecommender] = None,
        seller_rec: Optional[SellerExpansionRecommender] = None,
        item_cities: Optional[Dict[str, str]] = None,
        cascade_cfg=None,
    ):
        """
        Args:
            pv_replay: PageviewReplayRecommender.
            cocontact: CoContactRecommender.
            segpop: SegmentPopularityRecommender (last resort fallback).
            als: LightALSRecommender (priority 1 — strongest standalone).
            als_view: LightALSRecommender trained on pageview interactions.
            recent_cc: Pre-computed recent segment popular items.
            user_histories: {user_id: [recent_contact_item_ids]} for co-contact expansion.
            intent_rec: IntentRecommender (priority 2).
            user_knn: UserKNNRecommender (priority 4).
            seller_rec: SellerExpansionRecommender (priority 6).
            item_cities: Mapping from item_id to city_name.
            cascade_cfg: CascadeConfig with budgets and source orders. Falls back to defaults.
        """
        # Lazy import to avoid circular dependency
        if cascade_cfg is None:
            from config.settings import CascadeConfig
            cascade_cfg = CascadeConfig()
        self._cfg = cascade_cfg

        self._pv_replay = pv_replay
        self._cocontact = cocontact
        self._segpop = segpop
        self._als = als
        self._als_view = als_view
        self._recent_cc = recent_cc or {}
        self._user_histories = user_histories or {}
        self._intent_rec = intent_rec
        self._user_knn = user_knn
        self._seller_rec = seller_rec
        self._item_cities = item_cities or {}

    def generate(
        self,
        user_id: str,
        k: int = 200,
        pref_city: Optional[str] = None,
        pref_cat: Optional[int] = None,
    ) -> List[str]:
        """
        Generate candidates using sequential priority with adaptive budgets.
        When k<=10 (direct output): ALS-first for maximum precision.
        When k>10 (reranker pool): balanced budgets for maximum recall.
        """
        seen: Set[str] = set()
        items: List[str] = []

        if k <= 10:
            budgets = self._cfg.budget_top10
            source_order = self._cfg.source_order_top10
        else:
            budgets = self._cfg.budget_pool
            source_order = self._cfg.source_order_pool

        # Collect candidates from each source
        for source in source_order:
            if len(items) >= k:
                break
            budget = budgets.get(source, k - len(items))
            if budget <= 0:
                continue

            if source == "pv":
                cnt = 0
                for it in self._pv_replay.recommend(user_id, k=budget):
                    if it not in seen:
                        items.append(it); seen.add(it); cnt += 1
                        if cnt >= budget: break

            elif source == "intent" and self._intent_rec:
                cnt = 0
                for it in self._intent_rec.recommend(user_id, k=budget, exclude=set(seen)):
                    if it not in seen:
                        items.append(it); seen.add(it); cnt += 1
                        if cnt >= budget: break

            elif source == "cocontact":
                history = self._user_histories.get(user_id, [])
                if history:
                    cnt = 0
                    for it in self._cocontact.recommend(seed_items=history, k=budget, exclude=seen):
                        if it not in seen:
                            items.append(it); seen.add(it); cnt += 1
                            if cnt >= budget: break

            elif source == "als" and self._als is not None:
                als_recs = self._als.recommend_batch([user_id], n=budget, filter_already_liked=False, return_scores=False)
                cnt = 0
                for it in als_recs.get(user_id, []):
                    if isinstance(it, str) and it not in seen:
                        items.append(it); seen.add(it); cnt += 1
                        if cnt >= budget: break

            elif source == "als_view" and self._als_view is not None:
                als_view_recs = self._als_view.recommend_batch([user_id], n=budget, filter_already_liked=False, return_scores=False)
                cnt = 0
                for it in als_view_recs.get(user_id, []):
                    if isinstance(it, str) and it not in seen:
                        items.append(it); seen.add(it); cnt += 1
                        if cnt >= budget: break

            elif source == "user_knn" and self._user_knn:
                ctx = RecommendationContext(user_id=user_id, num_recommendations=budget)
                try:
                    knn_df = self._user_knn.recommend(ctx).collect()
                    if "item_id" in knn_df.columns:
                        cnt = 0
                        for it in knn_df["item_id"].to_list():
                            if it not in seen:
                                items.append(it); seen.add(it); cnt += 1
                                if cnt >= budget: break
                except Exception:
                    pass

            elif source == "seller" and self._seller_rec:
                ctx = RecommendationContext(user_id=user_id, num_recommendations=budget)
                try:
                    seller_df = self._seller_rec.recommend(ctx).collect()
                    if "item_id" in seller_df.columns:
                        cnt = 0
                        for it in seller_df["item_id"].to_list():
                            if it not in seen:
                                items.append(it); seen.add(it); cnt += 1
                                if cnt >= budget: break
                except Exception:
                    pass

            elif source == "recent_cc" and pref_city and pref_cat:
                cc_key = (pref_city, pref_cat)
                cnt = 0
                for it in self._recent_cc.get(cc_key, []):
                    if it not in seen:
                        items.append(it); seen.add(it); cnt += 1
                        if cnt >= budget: break

            elif source == "segpop":
                for it in self._segpop.get_segment_items(pref_city=pref_city, pref_cat=pref_cat, k=k - len(items), user_id=user_id):
                    if it not in seen:
                        items.append(it); seen.add(it)
                        if len(items) >= k: break

        return items[:k]

    def generate_batch(
        self,
        user_ids: List[str],
        user_prefs: Dict[str, Tuple[Optional[str], Optional[int]]],
        k: int = 200,
        valid_items: Optional[Set[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Generate candidates for a batch of users with adaptive budgets.
        k<=10: ALS-first precision mode. k>10: balanced recall mode.
        """
        # Precompute ALS and ALS View for the batch if available
        als_n = 10 if k <= 10 else self._cfg.als_recommend_n
        als_recs_batch = {}
        if self._als is not None:
            als_recs_batch = self._als.recommend_batch(user_ids, n=als_n, filter_already_liked=False, return_scores=False, valid_items=valid_items)
        
        als_view_recs_batch = {}
        if self._als_view is not None and k > 10:
            als_view_recs_batch = self._als_view.recommend_batch(user_ids, n=self._cfg.als_recommend_n, filter_already_liked=False, return_scores=False, valid_items=valid_items)

        results = {}
        stats = {
            "als": 0, "als_view": 0, "intent": 0, "pv": 0, "user_knn": 0, 
            "cocontact": 0, "seller": 0, "recent_cc": 0, "segpop": 0
        }

        if k <= 10:
            budgets = self._cfg.budget_top10
            source_order = self._cfg.source_order_top10
        else:
            budgets = self._cfg.budget_pool
            source_order = self._cfg.source_order_pool

        for uid in user_ids:
            pref_city, pref_cat = user_prefs.get(uid, (None, None))
            seen: Set[str] = set()
            items: List[str] = []

            for source in source_order:
                if len(items) >= k:
                    break
                budget = budgets.get(source, k - len(items))
                if budget <= 0:
                    continue

                if source == "pv":
                    cnt = 0
                    for it in self._pv_replay.recommend(uid, k=budget):
                        if it not in seen and (valid_items is None or it in valid_items):
                            items.append(it); seen.add(it); stats["pv"] += 1; cnt += 1
                            if cnt >= budget: break

                elif source == "intent" and self._intent_rec:
                    cnt = 0
                    for it in self._intent_rec.recommend(uid, k=budget, exclude=set(seen)):
                        if it not in seen:
                            items.append(it); seen.add(it); stats["intent"] += 1; cnt += 1
                            if cnt >= budget: break

                elif source == "cocontact":
                    history = self._user_histories.get(uid, [])
                    if history:
                        cnt = 0
                        for it in self._cocontact.recommend(seed_items=history, k=budget, exclude=set(seen)):
                            if it not in seen and (valid_items is None or it in valid_items):
                                items.append(it); seen.add(it); stats["cocontact"] += 1; cnt += 1
                                if cnt >= budget: break

                elif source == "als" and uid in als_recs_batch:
                    cnt = 0
                    for it in als_recs_batch[uid]:
                        if isinstance(it, str) and it not in seen and (valid_items is None or it in valid_items):
                            items.append(it); seen.add(it); stats["als"] += 1; cnt += 1
                            if cnt >= budget: break

                elif source == "als_view" and uid in als_view_recs_batch:
                    cnt = 0
                    for it in als_view_recs_batch[uid]:
                        if isinstance(it, str) and it not in seen and (valid_items is None or it in valid_items):
                            items.append(it); seen.add(it); stats["als_view"] += 1; cnt += 1
                            if cnt >= budget: break

                elif source == "user_knn" and self._user_knn:
                    ctx = RecommendationContext(user_id=uid, num_recommendations=budget)
                    try:
                        knn_df = self._user_knn.recommend(ctx).collect()
                        if "item_id" in knn_df.columns:
                            cnt = 0
                            for it in knn_df["item_id"].to_list():
                                if it not in seen and (valid_items is None or it in valid_items):
                                    items.append(it); seen.add(it); stats["user_knn"] += 1; cnt += 1
                                    if cnt >= budget: break
                    except Exception:
                        pass

                elif source == "seller" and self._seller_rec:
                    ctx = RecommendationContext(user_id=uid, num_recommendations=budget)
                    try:
                        seller_df = self._seller_rec.recommend(ctx).collect()
                        if "item_id" in seller_df.columns:
                            cnt = 0
                            for it in seller_df["item_id"].to_list():
                                if it not in seen and (valid_items is None or it in valid_items):
                                    items.append(it); seen.add(it); stats["seller"] += 1; cnt += 1
                                    if cnt >= budget: break
                    except Exception:
                        pass

                elif source == "recent_cc" and pref_city and pref_cat:
                    cc_key = (pref_city, pref_cat)
                    cnt = 0
                    for it in self._recent_cc.get(cc_key, []):
                        if it not in seen and (valid_items is None or it in valid_items):
                            items.append(it); seen.add(it); stats["recent_cc"] += 1; cnt += 1
                            if cnt >= budget: break

                elif source == "segpop":
                    for it in self._segpop.get_segment_items(pref_city=pref_city, pref_cat=pref_cat, k=k - len(items), user_id=uid):
                        if it not in seen and (valid_items is None or it in valid_items):
                            items.append(it); seen.add(it); stats["segpop"] += 1
                            if len(items) >= k: break

            results[uid] = items[:k]

        logger.info(
            f"Cascade items: ALS={stats['als']}, ALS_View={stats['als_view']}, Intent={stats['intent']}, PV={stats['pv']}, "
            f"UserKNN={stats['user_knn']}, CoContact={stats['cocontact']}, Seller={stats['seller']}, "
            f"RecentCC={stats['recent_cc']}, SegPop={stats['segpop']}"
        )
        return results

    def generate_batch_with_sources(
        self,
        user_ids: List[str],
        user_prefs: Dict[str, Tuple[Optional[str], Optional[int]]],
        k: int = 200,
        valid_items: Optional[Set[str]] = None,
        pos_set: Optional[Dict[str, set]] = None,
        label_col: bool = True,
    ) -> pl.DataFrame:
        """
        Generate candidates WITH source tracking — for LightGBM training.

        Returns DataFrame with columns:
            user_id, item_id, source,
            is_from_pv, is_from_intent, is_from_cocontact, is_from_als,
            is_from_user_knn, is_from_seller, is_from_recent_cc, is_from_segpop,
            label (if label_col=True and pos_set provided)
        """
        SOURCE_COLS = ["pv", "intent", "cocontact", "als", "als_view",
                       "user_knn", "seller", "recent_cc", "segpop"]

        # Pre-batch ALS
        als_recs_batch = {}
        if self._als is not None:
            als_recs_batch = self._als.recommend_batch(
                user_ids, n=self._cfg.als_recommend_n, filter_already_liked=False,
                return_scores=True, valid_items=valid_items,
            )
        als_view_recs_batch = {}
        als_view_budget = (
            self._cfg.budget_pool.get("als_view", 0)
            + self._cfg.budget_top10.get("als_view", 0)
        )
        if self._als_view is not None and als_view_budget > 0:
            als_view_recs_batch = self._als_view.recommend_batch(
                user_ids, n=self._cfg.als_recommend_n, filter_already_liked=False,
                return_scores=True, valid_items=valid_items,
            )

        budgets = self._cfg.budget_pool
        source_order = self._cfg.source_order_pool

        rows = []
        for uid in user_ids:
            pref_city, pref_cat = user_prefs.get(uid, (None, None))
            seen: Set[str] = set()
            user_rows: List[dict] = []

            for source in source_order:
                if len(user_rows) >= k:
                    break
                budget = budgets.get(source, k - len(user_rows))
                if budget <= 0:
                    continue

                source_items = self._get_source_items(
                    source, uid, budget, seen, valid_items,
                    als_recs_batch, als_view_recs_batch,
                    pref_city, pref_cat,
                )
                for it, score in source_items:
                    if it not in seen and (valid_items is None or it in valid_items):
                        row = {"user_id": uid, "item_id": it, "source": source}
                        for sc in SOURCE_COLS:
                            row[f"is_from_{sc}"] = 1.0 if sc == source else 0.0
                        if source in ("als", "als_view"):
                            row["score_als"] = score if source == "als" else 0.0
                            row["score_view_als"] = score if source == "als_view" else 0.0
                        else:
                            row["score_als"] = 0.0
                            row["score_view_als"] = 0.0
                        row["score_segpop"] = 0.0  # SegPop has no numeric score
                        seen.add(it)
                        user_rows.append(row)
                        if len(user_rows) >= k:
                            break

            # Add labels if requested
            if label_col and pos_set:
                gt = pos_set.get(uid, set())
                for row in user_rows:
                    row["label"] = 1 if row["item_id"] in gt else 0

            rows.extend(user_rows)

        df = pl.DataFrame(rows)
        logger.info(f"Cascade candidates with sources: {len(df):,} pairs, {df['user_id'].n_unique():,} users")
        return df

    def _get_source_items(
        self, source: str, uid: str, budget: int, seen: Set[str],
        valid_items, als_recs_batch, als_view_recs_batch,
        pref_city, pref_cat,
    ) -> List[Tuple[str, float]]:
        """Get (item_id, score) pairs from a single source."""
        results = []
        if source == "pv" and self._pv_replay:
            for it in self._pv_replay.recommend(uid, k=budget):
                results.append((it, 0.0))
        elif source == "intent" and self._intent_rec:
            for it in self._intent_rec.recommend(uid, k=budget, exclude=set(seen)):
                results.append((it, 0.0))
        elif source == "cocontact" and self._cocontact:
            history = self._user_histories.get(uid, [])
            if history:
                for it in self._cocontact.recommend(seed_items=history, k=budget, exclude=set(seen)):
                    results.append((it, 0.0))
        elif source == "als" and uid in als_recs_batch:
            for item in als_recs_batch[uid]:
                if isinstance(item, tuple):
                    results.append((item[0], item[1]))
                elif isinstance(item, str):
                    results.append((item, 0.0))
        elif source == "als_view" and uid in als_view_recs_batch:
            for item in als_view_recs_batch[uid]:
                if isinstance(item, tuple):
                    results.append((item[0], item[1]))
                elif isinstance(item, str):
                    results.append((item, 0.0))
        elif source == "user_knn" and self._user_knn:
            ctx = RecommendationContext(user_id=uid, num_recommendations=budget)
            try:
                knn_df = self._user_knn.recommend(ctx).collect()
                if "item_id" in knn_df.columns:
                    for it in knn_df["item_id"].to_list():
                        results.append((it, 0.0))
            except Exception:
                pass
        elif source == "seller" and self._seller_rec:
            ctx = RecommendationContext(user_id=uid, num_recommendations=budget)
            try:
                seller_df = self._seller_rec.recommend(ctx).collect()
                if "item_id" in seller_df.columns:
                    for it in seller_df["item_id"].to_list():
                        results.append((it, 0.0))
            except Exception:
                pass
        elif source == "recent_cc" and pref_city and pref_cat:
            cc_key = (pref_city, pref_cat)
            for it in self._recent_cc.get(cc_key, []):
                results.append((it, 0.0))
        elif source == "segpop" and self._segpop:
            for it in self._segpop.get_segment_items(
                pref_city=pref_city, pref_cat=pref_cat, k=budget, user_id=uid
            ):
                results.append((it, 0.0))
        return results

    @staticmethod
    def build_recent_cc(
        contact_pairs: pl.DataFrame,
        cutoff_date: object,
        window_days: int = 7,
        max_items_per_segment: int = 200,
    ) -> Dict[Tuple[str, int], List[str]]:
        """
        Build recent segment-popular items from contacts in the last N days.

        Factory method following SRP — this is a static data transformation,
        not part of the cascade logic itself.

        Args:
            contact_pairs: Contact pairs DataFrame with [city_name, category, item_id, last_date].
            cutoff_date: Only use contacts before this date.
            window_days: How many days back to look.
            max_items_per_segment: Max items per (city, category) segment.

        Returns:
            {(city_name, category): [item_ids]} sorted by contact count desc.
        """
        from datetime import timedelta

        cutoff = cutoff_date - timedelta(days=window_days)
        recent = (
            contact_pairs
            .filter(pl.col("last_date") > cutoff)
            .filter(
                pl.col("city_name").is_not_null()
                & pl.col("category").is_not_null()
            )
            .group_by(["city_name", "category", "item_id"])
            .agg(pl.len().alias("c"))
            .sort(["city_name", "category", "c"], descending=[False, False, True])
        )

        cc_map: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        for r in recent.iter_rows(named=True):
            key = (r["city_name"], r["category"])
            if len(cc_map[key]) < max_items_per_segment:
                cc_map[key].append(r["item_id"])

        logger.info(
            f"RecentCC built: {len(cc_map)} segments, "
            f"{window_days}d window, max {max_items_per_segment}/segment"
        )
        return dict(cc_map)

    @staticmethod
    def build_user_histories(
        contact_pairs: pl.DataFrame,
        user_ids: Set[str],
        max_items: int = 20,
    ) -> Dict[str, List[str]]:
        """
        Extract recent contact history for users (seeds for CoContact).

        Factory method following SRP.

        Args:
            contact_pairs: Contact pairs with [user_id, item_id, last_date].
            user_ids: Users to extract history for.
            max_items: Max items per user (ordered by recency).

        Returns:
            {user_id: [item_ids]} ordered by most recent first.
        """
        sorted_contacts = (
            contact_pairs
            .filter(pl.col("user_id").is_in(list(user_ids)))
            .sort("last_date", descending=True)
        )

        histories: Dict[str, List[str]] = defaultdict(list)
        for r in sorted_contacts.iter_rows(named=True):
            uid = r["user_id"]
            if len(histories[uid]) < max_items:
                histories[uid].append(r["item_id"])

        logger.info(f"User histories built: {len(histories):,} users")
        return dict(histories)
