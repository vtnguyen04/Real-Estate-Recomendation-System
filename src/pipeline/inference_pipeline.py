"""
src/pipeline/inference_pipeline.py — Production batch inference pipeline.

Supports 3 modes (via config.inference_mode):
  1. "cascade":  CascadeGen → top-k directly
  2. "hybrid":   CascadeGen k=200 → LightGBM rerank → top-k
  3. "legacy":   EnsembleGen → Features → LightGBM → MultiObjective reranker → top-k

All logic lives here. scripts/inference.py is a thin executor.
"""
from __future__ import annotations

import os
import time
import gc
import glob
from collections import defaultdict
from typing import Optional, Dict, Set, List

import numpy as np
import polars as pl

from config.settings import PipelineConfig
from src.models.candidates.pageview_replay import PageviewReplayRecommender
from src.models.candidates.cocontact import CoContactRecommender
from src.models.candidates.segment_popularity import SegmentPopularityRecommender
from src.models.candidates.intent_recommender import IntentRecommender
from src.models.candidates.light_als import LightALSRecommender
from src.models.candidates.user_knn import UserKNNRecommender
from src.models.candidates.seller_recommender import SellerExpansionRecommender
from src.models.ensemble.cascade_generator import CascadeCandidateGenerator
from src.models.ensemble.ensemble_generator import EnsembleCandidateGenerator
from src.models.rankers.lgbm_ranker import LambdarankLGBMRanker
from src.models.rerankers.multi_objective import MultiObjectiveReranker
from src.features.feature_engineer import FeatureEngineer
from src.features.extractors.recent_history import RecentHistoryExtractor
from src.features.extractors.seller_affinity import SellerAffinityExtractor
from src.features.extractors.preference_match import PreferenceMatchExtractor
from src.features.extractors.item_snapshot import ItemSnapshotExtractor
from src.utils.logging import get_logger

logger = get_logger("inference_pipeline")


class InferencePipeline:
    """
    Production batch inference pipeline for generating submission.csv.

    Usage:
        pipeline = InferencePipeline(config, data_dir, model_dir, cache_dir)
        pipeline.load_data(test_data_dir)
        pipeline.fit_sources()
        df_submission = pipeline.predict()
        df_submission.write_csv("submission.csv")
    """

    def __init__(
        self,
        config: PipelineConfig,
        data_dir: str,
        model_dir: str,
        cache_dir: str,
    ):
        self.cfg = config
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.cache_dir = cache_dir

        # Data
        self.test_users: List[str] = []
        self.test_user_set: Set[str] = set()
        self.contacts: Optional[pl.DataFrame] = None
        self.df_listing: Optional[pl.DataFrame] = None
        self.valid_items: Optional[Set[str]] = None
        self.prefs_dict: Dict[str, tuple] = {}
        self.max_date = None

        # Sources (fitted lazily)
        self.pv_replay: Optional[PageviewReplayRecommender] = None
        self.cocontact: Optional[CoContactRecommender] = None
        self.segpop: Optional[SegmentPopularityRecommender] = None
        self.als: Optional[LightALSRecommender] = None
        self.als_view: Optional[LightALSRecommender] = None
        self.intent_rec: Optional[IntentRecommender] = None
        self.user_knn: Optional[UserKNNRecommender] = None
        self.seller_rec: Optional[SellerExpansionRecommender] = None

        # Generators
        self.cascade: Optional[CascadeCandidateGenerator] = None
        self.ensemble_gen: Optional[EnsembleCandidateGenerator] = None

        # Ranker + features (for hybrid/legacy modes)
        self.ranker: Optional[LambdarankLGBMRanker] = None
        self.reranker: Optional[MultiObjectiveReranker] = None
        self.feature_eng: Optional[FeatureEngineer] = None
        self.user_stats_df: Optional[pl.DataFrame] = None
        self.item_stats_df: Optional[pl.DataFrame] = None
        self.item_meta_df: Optional[pl.DataFrame] = None
        self.snapshot_stats_df: Optional[pl.DataFrame] = None

    # ── Step 1: Load data ──────────────────────────────────────

    def load_data(self, test_data_dir: str) -> "InferencePipeline":
        """Load test users, contacts, listing, preferences."""
        logger.info("[1/5] Loading data...")

        self.test_users = pl.read_parquet(
            os.path.join(test_data_dir, "test_users.parquet")
        )["user_id"].to_list()
        self.test_user_set = set(self.test_users)
        logger.info(f"  Test users: {len(self.test_users):,}")

        self.contacts = pl.read_parquet(os.path.join(self.cache_dir, "contact_pairs.parquet"))
        date_range = pl.read_parquet(os.path.join(self.cache_dir, "date_range.parquet"))
        self.max_date = date_range["max_date"][0]

        # Load dim_listing
        dim_files = glob.glob(os.path.join(self.data_dir, "dim_listing/*.parquet"))
        if dim_files:
            self.df_listing = pl.scan_parquet(dim_files).collect()
            self.valid_items = set(self.df_listing["item_id"].to_list())
            logger.info(f"  Valid items: {len(self.valid_items):,}")

        # User preferences from contacts
        prefs_df = (
            self.contacts.filter(pl.col("user_id").is_in(list(self.test_user_set)))
            .group_by("user_id")
            .agg([
                pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
                pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("pref_cat"),
            ])
        )
        for r in prefs_df.iter_rows(named=True):
            self.prefs_dict[r["user_id"]] = (r.get("pref_city"), r.get("pref_cat"))

        # Merge cold user prefs
        cold_prefs_path = os.path.join(self.cache_dir, "cold_user_prefs.parquet")
        if os.path.exists(cold_prefs_path):
            cold_prefs = pl.read_parquet(cold_prefs_path)
            n_before = len(self.prefs_dict)
            for r in cold_prefs.iter_rows(named=True):
                uid = r["user_id"]
                if uid not in self.prefs_dict and uid in self.test_user_set:
                    self.prefs_dict[uid] = (r.get("pref_city"), r.get("pref_cat"))
            logger.info(f"  Cold prefs merged: {len(self.prefs_dict) - n_before:,}")

        logger.info(f"  Users with prefs: {len(self.prefs_dict):,}/{len(self.test_users):,}")
        return self

    # ── Step 2: Fit candidate sources ──────────────────────────

    def fit_sources(self) -> "InferencePipeline":
        """Fit all candidate sources for the current inference mode."""
        logger.info("[2/5] Fitting candidate sources...")
        mode = self.cfg.inference_mode
        events_path = os.path.join(self.data_dir, "fact_user_events/*.parquet")

        if mode in ("cascade", "hybrid"):
            self._fit_cascade_sources(events_path)
        elif mode == "legacy":
            self._fit_legacy_sources()

        return self

    def _fit_cascade_sources(self, events_path: str) -> None:
        """Fit sources used by CascadeGen: PV, CoContact, ALS, Intent, UserKNN, Seller."""
        cc = self.cfg.cascade

        # PageviewReplay
        self.pv_replay = PageviewReplayRecommender(
            window_days=cc.pv_window_days, max_items_per_user=cc.pv_max_items_per_user)
        self.pv_replay.fit(events_path, user_ids=self.test_user_set, cutoff_date=self.max_date)

        # CoContact
        self.cocontact = CoContactRecommender(window_days=cc.cocontact_window_days)
        self.cocontact.fit(self.contacts, cutoff_date=self.max_date)

        # Recent CC
        recent_cc = CascadeCandidateGenerator.build_recent_cc(
            self.contacts, cutoff_date=self.max_date,
            window_days=cc.recent_cc_window_days,
            max_items_per_segment=cc.recent_cc_max_items_per_segment,
        )

        # SegPop (pre-trained)
        self.segpop = SegmentPopularityRecommender().load(
            os.path.join(self.model_dir, "segpop.pkl")
        )
        snapshot_cache = os.path.join(self.cache_dir, "snapshot_stats.parquet")
        if self.cfg.inference_mode != "cascade" and os.path.exists(snapshot_cache):
            self.segpop.set_blind_global_from_snapshot(
                pl.read_parquet(snapshot_cache),
                valid_items=self.valid_items,
            )

        # ALS
        self.als = LightALSRecommender()
        self.als.load(os.path.join(self.model_dir, "als"))
        if self.als._matrix is None:
            # Rebuild from same data train used (weighted vs standard)
            als_rebuild_file = "als_weighted_contact.parquet" if self.cfg.model.als_use_weighted else "als_contact_pairs.parquet"
            als_rebuild_path = os.path.join(self.cache_dir, als_rebuild_file)
            if not os.path.exists(als_rebuild_path):
                als_rebuild_path = os.path.join(self.cache_dir, "als_contact_pairs.parquet")
            als_contacts = pl.read_parquet(als_rebuild_path)
            self.als.rebuild_matrix(als_contacts)

        # ALS View — only load if budget > 0 (INS-047: als_view disabled)
        has_view_budget = (
            cc.budget_pool.get("als_view", 0) > 0
            or cc.budget_top10.get("als_view", 0) > 0
        )
        als_view_path = os.path.join(self.model_dir, "als_view")
        if has_view_budget and os.path.exists(als_view_path):
            self.als_view = LightALSRecommender()
            self.als_view.load(als_view_path)
            if self.als_view._matrix is None:
                pv_path = os.path.join(self.cache_dir, "als_pageview_pairs.parquet")
                if os.path.exists(pv_path):
                    self.als_view.rebuild_matrix(pl.read_parquet(pv_path))
            logger.info("  ALS View loaded (budget > 0).")
        else:
            logger.info("  ALS View skipped (budget=0 or no artifact).")

        # User histories
        user_histories = CascadeCandidateGenerator.build_user_histories(
            self.contacts, user_ids=self.test_user_set,
            max_items=cc.user_history_max_items,
        )

        # Intent
        self.intent_rec = IntentRecommender(max_items_per_intent=cc.intent_max_items_per_intent)
        pvs = (
            pl.scan_parquet(events_path)
            .filter(
                (pl.col("event_ts") <= self.max_date)
                & (pl.col("event_ts") >= self.max_date - pl.duration(days=cc.pv_window_days))
                & (pl.col("event_type") == "pageview")
            )
            .select(["user_id", "item_id"]).collect()
        )
        if self.df_listing is not None:
            self.intent_rec.fit(pvs=pvs, dim_listing=self.df_listing, valid_items=self.valid_items)

        # UserKNN
        self.user_knn = UserKNNRecommender(max_neighbors_per_item=cc.user_knn_max_neighbors)
        self.user_knn.fit(self.contacts.lazy(), query_user_ids=self.test_user_set, valid_items=self.valid_items)

        # Seller
        self.seller_rec = SellerExpansionRecommender(max_items_per_seller=cc.seller_max_items_per_seller)
        self.seller_rec.fit(self.contacts.lazy(), listing_df=self.df_listing, query_user_ids=self.test_user_set)

        # Item cities
        item_cities = dict(zip(self.df_listing["item_id"], self.df_listing["city_name"])) if self.df_listing is not None else {}

        # Build cascade
        self.cascade = CascadeCandidateGenerator(
            pv_replay=self.pv_replay, cocontact=self.cocontact, segpop=self.segpop,
            als=self.als, als_view=self.als_view,
            recent_cc=recent_cc, user_histories=user_histories,
            intent_rec=self.intent_rec, user_knn=self.user_knn,
            seller_rec=self.seller_rec, item_cities=item_cities,
            cascade_cfg=cc,
        )

        # Load ranker for hybrid mode
        if self.cfg.inference_mode == "hybrid":
            self._load_ranker()

    def _fit_legacy_sources(self) -> None:
        """Fit sources for legacy mode: ALS + ALS_view + SegPop + EnsembleGen + Ranker."""
        self.segpop = SegmentPopularityRecommender().load(os.path.join(self.model_dir, "segpop.pkl"))
        self.als = LightALSRecommender()
        self.als.load(os.path.join(self.model_dir, "als"))
        if self.als._matrix is None:
            als_rebuild_file = "als_weighted_contact.parquet" if self.cfg.model.als_use_weighted else "als_contact_pairs.parquet"
            als_rebuild_path = os.path.join(self.cache_dir, als_rebuild_file)
            if not os.path.exists(als_rebuild_path):
                als_rebuild_path = os.path.join(self.cache_dir, "als_contact_pairs.parquet")
            self.als.rebuild_matrix(pl.read_parquet(als_rebuild_path))
        self.als_view = None
        als_view_path = os.path.join(self.model_dir, "als_view")
        if os.path.isdir(als_view_path):
            self.als_view = LightALSRecommender()
            self.als_view.load(als_view_path)
            if self.als_view._matrix is None:
                pv_path = os.path.join(self.cache_dir, "als_pageview_pairs.parquet")
                if os.path.exists(pv_path):
                    pv = pl.read_parquet(pv_path)
                    self.als_view.rebuild_matrix(pv)
        else:
            logger.info("  als_view artifact not found, skipping.")

        self.ensemble_gen = EnsembleCandidateGenerator(
            als=self.als, als_view=self.als_view, segpop=self.segpop,
            n_cand_als=self.cfg.model.n_cand_als,
            n_cand_view_als=self.cfg.model.n_cand_view_als,
            n_cand_segpop=self.cfg.model.n_cand_segpop,
        )
        self._load_ranker()

        self.reranker = MultiObjectiveReranker(
            alpha=self.cfg.reranker.alpha, beta=self.cfg.reranker.beta,
            gamma=self.cfg.reranker.gamma, delta=self.cfg.reranker.delta,
            epsilon=self.cfg.reranker.epsilon,
        )

    def _load_ranker(self) -> None:
        """Load LightGBM ranker + feature tables + feature engineer."""
        self.ranker = LambdarankLGBMRanker()
        self.ranker.load(self.model_dir)
        logger.info(f"  LightGBM ranker: {len(self.ranker.feature_cols)} features")

        self.user_stats_df = pl.read_parquet(os.path.join(self.model_dir, "user_stats.parquet"))
        self.item_stats_df = pl.read_parquet(os.path.join(self.model_dir, "item_stats.parquet"))
        self.item_meta_df = pl.read_parquet(os.path.join(self.model_dir, "item_meta.parquet"))

        # Load snapshot stats (Fix 2: train/inference feature parity)
        snapshot_path = os.path.join(self.model_dir, "snapshot_stats.parquet")
        if os.path.exists(snapshot_path):
            self.snapshot_stats_df = pl.read_parquet(snapshot_path)
            logger.info(f"  Snapshot stats: {len(self.snapshot_stats_df):,} items")
        else:
            snapshot_cache = os.path.join(self.cache_dir, "snapshot_stats.parquet")
            if os.path.exists(snapshot_cache):
                self.snapshot_stats_df = pl.read_parquet(snapshot_cache)
                logger.info(f"  Snapshot stats (from cache): {len(self.snapshot_stats_df):,} items")

        recent_ext = RecentHistoryExtractor(self.contacts)
        seller_ext = SellerAffinityExtractor(self.contacts, self.df_listing)
        match_ext = PreferenceMatchExtractor()
        snapshot_ext = ItemSnapshotExtractor(snapshot_path)
        self.feature_eng = FeatureEngineer(extractors=[recent_ext, seller_ext, match_ext, snapshot_ext])

    # ── Step 3: Generate predictions ───────────────────────────

    def predict(self) -> pl.DataFrame:
        """Run batch inference and return submission DataFrame."""
        mode = self.cfg.inference_mode
        logger.info(f"[3/5] Generating predictions (mode={mode})...")

        if mode == "cascade":
            return self._predict_cascade(rerank=False)
        elif mode == "hybrid":
            return self._predict_cascade(rerank=True)
        elif mode == "legacy":
            return self._predict_legacy()
        else:
            raise ValueError(f"Unknown inference mode: {mode}")

    def _predict_cascade(self, rerank: bool) -> pl.DataFrame:
        """Cascade inference with optional LightGBM reranking.

        Segmented Inference Policy (INS-069):
        Warm users (exist in contacts) -> hybrid rerank (cascade k=200 -> LightGBM)
        Cold/Blind users (no contacts) -> cascade direct (k=10)
        """
        assert self.cascade is not None, "Cascade Candidate Generator is not initialized"
        assert self.contacts is not None, "Contacts DataFrame is not loaded"

        if not rerank:
            # Pure cascade: direct cascade for all users
            pool_k = self.cfg.top_k
            batch_size = self.cfg.cand_batch
            all_rows: list[dict] = []
            for batch_start in range(0, len(self.test_users), batch_size):
                batch = self.test_users[batch_start:batch_start + batch_size]
                recs = self.cascade.generate_batch(
                    user_ids=batch, user_prefs=self.prefs_dict,
                    k=pool_k, valid_items=self.valid_items,
                )
                batch_rows = self._direct_batch(batch, recs)
                all_rows.extend(batch_rows)
            return self._build_submission(all_rows)

        # Hybrid: Segmented Inference Policy
        warm_user_set = set(self.contacts["user_id"].unique().to_list()) & self.test_user_set
        logger.info(f"Segmented Inference Policy: {len(warm_user_set):,} Warm users (LGBM), {len(self.test_users) - len(warm_user_set):,} Cold/Blind users (Direct)")

        pool_k = self.cfg.cascade.hybrid_pool_size
        batch_size = self.cfg.cand_batch
        all_rows: list[dict] = []

        for batch_start in range(0, len(self.test_users), batch_size):
            batch = self.test_users[batch_start:batch_start + batch_size]
            batch_warm = [u for u in batch if u in warm_user_set]
            batch_cold = [u for u in batch if u not in warm_user_set]

            # 1. Warm users -> Hybrid LGBM reranking
            if batch_warm:
                df_cands_warm = self.cascade.generate_batch_with_sources(
                    user_ids=batch_warm, user_prefs=self.prefs_dict,
                    k=pool_k, valid_items=self.valid_items,
                    label_col=False,
                )
                warm_rows = self._rerank_batch_df(batch_warm, df_cands_warm)
                all_rows.extend(warm_rows)

            # 2. Cold/Blind users -> Direct cascade (k=10)
            if batch_cold:
                recs_cold = self.cascade.generate_batch(
                    user_ids=batch_cold, user_prefs=self.prefs_dict,
                    k=self.cfg.top_k, valid_items=self.valid_items,
                )
                cold_rows = self._direct_batch(batch_cold, recs_cold)
                all_rows.extend(cold_rows)

            if (batch_start // batch_size + 1) % 10 == 0 or batch_start + batch_size >= len(self.test_users):
                logger.info(f"  {min(batch_start + batch_size, len(self.test_users)):,}/{len(self.test_users):,}")

        return self._build_submission(all_rows)

    def _rerank_batch_df(self, batch: list, df_cands: pl.DataFrame) -> list:
        """Rerank cascade candidates with LightGBM.

        df_cands already has source flags (is_from_pv, is_from_als, etc)
        and score_als, score_view_als from generate_batch_with_sources().
        """
        if len(df_cands) == 0:
            return []

        # Ensure score_segpop exists (cascade doesn't produce it as a float score)
        if "score_segpop" not in df_cands.columns:
            df_cands = df_cands.with_columns(pl.lit(0.0).alias("score_segpop"))

        # Feature tables
        df_cands = self.feature_eng.attach_features_inference(
            df_cands, self.user_stats_df, self.item_stats_df, self.item_meta_df
        )

        # Join snapshot features
        if self.snapshot_stats_df is not None:
            df_cands = df_cands.join(self.snapshot_stats_df, on="item_id", how="left")

        # Fill any missing feature cols with 0
        for fc in self.ranker.feature_cols:
            if fc not in df_cands.columns:
                df_cands = df_cands.with_columns(pl.lit(0.0).alias(fc))

        # Score + rank
        scores = self.ranker.predict(df_cands)
        df_cands = df_cands.with_columns(pl.Series("lgbm_score", scores.tolist()))
        df_ranked = df_cands.sort(["user_id", "lgbm_score"], descending=[False, True])

        rows = []
        for uid in batch:
            items = df_ranked.filter(pl.col("user_id") == uid).head(self.cfg.top_k)["item_id"].to_list()
            items = self._pad_items(items)
            for rank, iid in enumerate(items[:self.cfg.top_k], start=1):
                rows.append({"user_id": uid, "rank": rank, "item_id": iid})
        return rows

    def _direct_batch(self, batch: list, recs: dict) -> list:
        """Direct cascade output → top-k with padding."""
        rows = []
        for uid in batch:
            items = list(recs.get(uid, []))
            items = self._pad_items(items)
            for rank, iid in enumerate(items[:self.cfg.top_k], start=1):
                rows.append({"user_id": uid, "rank": rank, "item_id": iid})
        return rows

    def _predict_legacy(self) -> pl.DataFrame:
        """Legacy: EnsembleGen → Features → LightGBM → MultiObjective reranker."""
        batch_size = self.cfg.cand_batch
        all_rows: list[dict] = []
        predicted_users: set = set()

        for batch_start in range(0, len(self.test_users), batch_size):
            batch = self.test_users[batch_start:batch_start + batch_size]
            df_batch, _ = self.ensemble_gen.generate_batch(
                users=batch, user_prefs=self.prefs_dict, valid_items=self.valid_items,
            )
            if len(df_batch) == 0:
                continue
            df_batch = self.feature_eng.attach_features_inference(
                df_batch, self.user_stats_df, self.item_stats_df, self.item_meta_df
            )
            scores = self.ranker.predict(df_batch)
            df_batch = df_batch.with_columns(pl.Series("lgbm_score", scores.tolist()))
            df_reranked = self.reranker.rerank_batch(df_batch, k=self.cfg.top_k)

            user_rank = defaultdict(int)
            for r in df_reranked.select(["user_id", "item_id"]).iter_rows():
                uid, iid = r[0], r[1]
                if user_rank[uid] < self.cfg.top_k:
                    user_rank[uid] += 1
                    all_rows.append({"user_id": uid, "rank": user_rank[uid], "item_id": iid})
                    predicted_users.add(uid)

            if (batch_start // batch_size + 1) % 5 == 0:
                logger.info(f"  {min(batch_start + batch_size, len(self.test_users)):,}/{len(self.test_users):,}")

        # Cold-start fallback
        cold_users = [u for u in self.test_users if u not in predicted_users]
        if cold_users:
            global_top = self.segpop._global[:self.cfg.top_k]
            for uid in cold_users:
                for rank, iid in enumerate(global_top, start=1):
                    all_rows.append({"user_id": uid, "rank": rank, "item_id": iid})

        return self._build_submission(all_rows)

    # ── Helpers ────────────────────────────────────────────────

    def _compute_als_scores(self, df_pairs: pl.DataFrame) -> tuple:
        """Compute ALS dot-product scores for (user, item) pairs."""
        try:
            uf = self.als._model.user_factors
            itf = self.als._model.item_factors
            if hasattr(uf, "to_numpy"):
                uf, itf = uf.to_numpy(), itf.to_numpy()
        except AttributeError:
            uf = itf = None

        als_scores, is_from_als = [], []
        for r in df_pairs.iter_rows(named=True):
            uid_idx = self.als._u2i.get(r["user_id"])
            iid_idx = self.als._i2i.get(r["item_id"])
            if uf is not None and uid_idx is not None and iid_idx is not None:
                als_scores.append(float(np.dot(uf[uid_idx], itf[iid_idx])))
                is_from_als.append(1.0)
            else:
                als_scores.append(0.0)
                is_from_als.append(0.0)
        return als_scores, is_from_als

    def _pad_items(self, items: list) -> list:
        """Pad item list to top_k using global popular items."""
        if len(items) >= self.cfg.top_k:
            return items
        seen = set(items)
        if self.segpop and hasattr(self.segpop, "_global"):
            for pad_item in self.segpop._global:
                if pad_item not in seen and (self.valid_items is None or pad_item in self.valid_items):
                    items.append(pad_item)
                    seen.add(pad_item)
                    if len(items) >= self.cfg.top_k:
                        break
        return items

    def _build_submission(self, all_rows: list) -> pl.DataFrame:
        """Format rows into submission DataFrame."""
        df_sub = pl.DataFrame(all_rows).select(["user_id", "rank", "item_id"])
        df_sub = df_sub.with_columns(pl.Series("ID", range(1, len(df_sub) + 1)))
        df_sub = df_sub.select(["ID", "user_id", "rank", "item_id"])
        logger.info(f"  Submission: {len(df_sub):,} rows, {df_sub['user_id'].n_unique():,} users")
        return df_sub

    @classmethod
    def load(cls, path: str) -> "InferencePipeline":
        """Load serialized pipeline."""
        import joblib
        return joblib.load(path)

    def save(self, path: str) -> None:
        """Save serialized pipeline."""
        import joblib
        joblib.dump(self, path)
