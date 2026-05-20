"""
src/pipeline/training_pipeline.py — Two-stage recommendation training pipeline.

Stage 1 — Candidate retrieval: ContactALS + ViewALS + SegPop fallback
Stage 2 — Ranking: LightGBM lambdarank on engineered features

All hyper-parameters come from PipelineConfig (config/settings.py).
"""
from __future__ import annotations

import gc
import os
from collections import defaultdict
from datetime import timedelta
from typing import Optional

import numpy as np
import polars as pl

from config.settings import PipelineConfig
from src.features.extractors.user_behavior import UserBehaviorExtractor
from src.features.extractors.item_quality import ItemQualityExtractor
from src.features.extractors.item_stats import ItemStatsExtractor
from src.features.extractors.recent_history import RecentHistoryExtractor
from src.features.extractors.seller_affinity import SellerAffinityExtractor
from src.features.extractors.preference_match import PreferenceMatchExtractor
from src.features.feature_engineer import FeatureEngineer
from src.models.candidates.light_als import LightALSRecommender
from src.models.candidates.segment_popularity import SegmentPopularityRecommender
from src.models.ensemble.ensemble_generator import EnsembleCandidateGenerator
from src.models.rankers.lgbm_ranker import LambdarankLGBMRanker
from src.evaluation.metrics import recall_at_k, ndcg_at_k
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Orchestrates ContactALS + ViewALS + SegPop + LightGBM lambdarank training.

    All hyper-parameters read from PipelineConfig — no hardcoded values.

    Usage:
        pipeline = TrainingPipeline(config, use_gpu=True, output_dir="outputs/models/")
        pipeline.run(contacts, als_contacts, als_pageviews, df_listing, split_date)
    """

    def __init__(
        self,
        config: PipelineConfig,
        use_gpu: bool = False,
        output_dir: str = "outputs/models/",
        cache_dir: str = ".cache",
    ):
        self.cfg = config
        self.use_gpu = use_gpu
        self.output_dir = output_dir
        self.cache_dir = cache_dir

        self.segpop: Optional[SegmentPopularityRecommender] = None
        self.als: Optional[LightALSRecommender] = None
        self.als_view: Optional[LightALSRecommender] = None
        self.ensemble_gen: Optional[EnsembleCandidateGenerator] = None
        self.ranker: Optional[LambdarankLGBMRanker] = None

    # ── Candidate generation ────────────────────────────────────

    def generate_candidates(
        self,
        users: list,
        pos_set: dict,
        user_prefs: dict,
        valid_items: set,
        label_col: bool = True,
    ) -> pl.DataFrame:
        """
        Generate (user, item) candidate pairs with actual ALS affinity scores.
        Delegates to EnsembleCandidateGenerator to respect SRP and DRY.
        """
        df, _ = self.ensemble_gen.generate_batch(
            users=users,
            user_prefs=user_prefs,
            valid_items=valid_items,
            pos_set=pos_set,
            label_col=label_col
        )
        return df

    # ── Main entry point ────────────────────────────────────────

    def run(
        self,
        contacts: pl.DataFrame,
        als_contacts: pl.DataFrame,
        als_pageviews: pl.DataFrame,
        df_listing: pl.DataFrame,
        split_date,
        ext_logger=None,
    ) -> "TrainingPipeline":
        """Execute the full training pipeline and save all artefacts."""
        log = ext_logger or logger
        cfg = self.cfg
        os.makedirs(self.output_dir, exist_ok=True)

        pos_cutoff = split_date - timedelta(days=cfg.positive_window_days)

        train_contacts = contacts.filter(pl.col("last_date") <= split_date)
        val_contacts   = contacts.filter(pl.col("last_date") > split_date)

        val_gt: dict[str, set] = defaultdict(set)
        for r in val_contacts.iter_rows(named=True):
            val_gt[r["user_id"]].add(r["item_id"])
        val_users = list(val_gt.keys())

        recent = train_contacts.filter(pl.col("last_date") > pos_cutoff)
        log.info(f"  Recent contacts (last {cfg.positive_window_days}d): {len(recent):,}")
        train_pos: dict[str, set] = defaultdict(set)
        for r in recent.iter_rows(named=True):
            train_pos[r["user_id"]].add(r["item_id"])
        train_users = list(train_pos.keys())
        valid_items = set(df_listing["item_id"].to_list())

        log.info(
            f"  Train contacts: {len(train_contacts):,} | Train users: {len(train_users):,} "
            f"| Val users: {len(val_users):,} | Items: {len(valid_items):,}"
        )

        # ── SegPop ──────────────────────────────────────────────
        log.info("[2/8] Fitting SegmentPopularity (cold-start fallback)...")
        self.segpop = SegmentPopularityRecommender(
            cc_k=cfg.segpop_cc_k,
            segment_k=cfg.segpop_segment_k,
            global_k=cfg.segpop_global_k,
        )
        self.segpop.fit_from_pairs(train_contacts, valid_items=valid_items, listing_df=df_listing)
        # INS-053 fix: save as segpop_trained.pkl to avoid overwriting recency segpop.pkl
        self.segpop.save(os.path.join(self.output_dir, "segpop_trained.pkl"))
        log.info("  SegPop saved as segpop_trained.pkl (NOT overwriting segpop.pkl).")

        # ── ContactALS ──────────────────────────────────────────
        log.info(
            f"[3/8] Fitting ContactALS "
            f"(f={cfg.model.als_factors}, i={cfg.model.als_iterations}, gpu={self.use_gpu})..."
        )
        self.als = LightALSRecommender(
            factors=cfg.model.als_factors,
            iterations=cfg.model.als_iterations,
            use_gpu=self.use_gpu,
        )
        self.als.fit(als_contacts.lazy())
        self.als.save(os.path.join(self.output_dir, "als"))
        log.info("  ContactALS saved.")

        # ── ViewALS ─────────────────────────────────────────────
        log.info(
            f"[4/8] Fitting ViewALS "
            f"(f={cfg.model.als_view_factors}, i={cfg.model.als_view_iterations}, gpu={self.use_gpu})..."
        )
        self.als_view = LightALSRecommender(
            factors=cfg.model.als_view_factors,
            iterations=cfg.model.als_view_iterations,
            use_gpu=self.use_gpu,
        )
        self.als_view.fit(als_pageviews.select(["user_id", "item_id", "view_count"]).lazy())
        self.als_view.save(os.path.join(self.output_dir, "als_view"))
        log.info("  ViewALS saved.")
        del als_pageviews; gc.collect()

        # ── Ensemble Generator ───────────────────────────────────
        self.ensemble_gen = EnsembleCandidateGenerator(
            als=self.als, als_view=self.als_view, segpop=self.segpop,
            n_cand_als=cfg.model.n_cand_als,
            n_cand_view_als=cfg.model.n_cand_view_als,
            n_cand_segpop=cfg.model.n_cand_segpop
        )

        # ── Feature lookup tables via extractors ─────────────────
        log.info("[5/8] Building feature lookup tables via extractors...")
        als_pv_reload = pl.read_parquet(os.path.join(self.cache_dir, "als_pageview_pairs.parquet"))
        user_ext       = UserBehaviorExtractor(train_contacts, df_listing)
        item_stats_ext = ItemStatsExtractor(train_contacts, als_pv_reload, pos_cutoff)
        item_meta_ext  = ItemQualityExtractor(df_listing, split_date)
        recent_ext     = RecentHistoryExtractor(train_contacts)
        seller_ext     = SellerAffinityExtractor(train_contacts, df_listing)
        match_ext      = PreferenceMatchExtractor()
        feature_eng    = FeatureEngineer([user_ext, item_stats_ext, item_meta_ext, recent_ext, seller_ext, match_ext])
        del als_pv_reload; gc.collect()

        user_stats_df = user_ext.build_feature_df()
        user_prefs: dict[str, tuple] = {
            r["user_id"]: (r.get("pref_city"), r.get("pref_cat"))
            for r in user_stats_df.iter_rows(named=True)
        }

        # ── Training candidates ──────────────────────────────────
        log.info("[6/8] Generating training candidates (ALS-first)...")
        rng = np.random.default_rng(42)
        sampled = rng.choice(
            train_users, size=min(cfg.n_train_users, len(train_users)), replace=False
        ).tolist()

        batch_files: list[str] = []
        total_pairs = 0
        for i, start in enumerate(range(0, len(sampled), cfg.cand_batch)):
            chunk = sampled[start:start + cfg.cand_batch]
            chunk_df = self.generate_candidates(chunk, train_pos, user_prefs, valid_items, label_col=True)
            keep = (
                chunk_df.group_by("user_id").agg(pl.col("label").sum().alias("n_pos"))
                .filter(pl.col("n_pos") > 0)["user_id"].to_list()
            )
            chunk_df = chunk_df.filter(pl.col("user_id").is_in(keep))
            if len(chunk_df) == 0:
                continue
            chunk_df = feature_eng.extract_for_training(chunk_df)
            fpath = f"/tmp/train_chunk_{i}.parquet"
            chunk_df.write_parquet(fpath)
            batch_files.append(fpath)
            total_pairs += len(chunk_df)
            log.info(f"  chunk {len(batch_files)}: {len(chunk_df):,} pairs  total: {total_pairs:,}")
            del chunk_df; gc.collect()

        log.info(f"  Total training pairs: {total_pairs:,}")
        df_train = pl.scan_parquet(batch_files).collect()
        available_feats = [c for c in cfg.ranker.feature_cols if c in df_train.columns]
        log.info(f"  Feature cols ({len(available_feats)}): {available_feats}")

        # Val candidates (small sample for early stopping signal)
        val_pairs_df = self.generate_candidates(
            val_users[:cfg.val_sample], val_gt, user_prefs, valid_items, label_col=True
        )
        val_keep = (
            val_pairs_df.group_by("user_id").agg(pl.col("label").sum().alias("n_pos"))
            .filter(pl.col("n_pos") > 0)["user_id"].to_list()
        )
        val_pairs_df = val_pairs_df.filter(pl.col("user_id").is_in(val_keep))
        df_val = (
            feature_eng.extract_for_training(val_pairs_df)
            if len(val_pairs_df) > 0 else None
        )

        # ── LightGBM ─────────────────────────────────────────────
        log.info("[7/8] Training LightGBM lambdarank...")
        r = cfg.ranker
        self.ranker = LambdarankLGBMRanker(
            feature_cols=available_feats,
            use_gpu=self.use_gpu,
            num_leaves=r.num_leaves,
            learning_rate=r.learning_rate,
            num_rounds=r.n_estimators,
            early_stopping=r.early_stopping_rounds,
        )
        self.ranker.fit(df_train, val_df=df_val)

        # ── Save artefacts ────────────────────────────────────────
        log.info("[8/8] Saving artefacts...")
        self.ranker.save(self.output_dir)
        item_stats_df = item_stats_ext.build_feature_df()
        item_meta_df  = item_meta_ext.build_feature_df()
        user_stats_df.write_parquet(os.path.join(self.output_dir, "user_stats.parquet"))
        item_stats_df.write_parquet(os.path.join(self.output_dir, "item_stats.parquet"))
        item_meta_df.write_parquet(os.path.join(self.output_dir, "item_meta.parquet"))
        log.info(f"  All artefacts saved to {self.output_dir}")

        if df_val is not None:
            self._quick_val_eval(df_val, val_gt, val_users[:cfg.val_sample], log)

        return self

    def _quick_val_eval(
        self,
        df_val: pl.DataFrame,
        val_gt: dict,
        val_users: list,
        log,
    ) -> None:
        log.info("── Quick validation eval ──")
        df_scored = df_val.sort("user_id")
        scores = self.ranker.predict(df_scored)
        df_scored = df_scored.with_columns(pl.Series("lgbm_score", scores.tolist()))
        top10 = (
            df_scored.sort(["user_id", "lgbm_score"], descending=[False, True])
            .group_by("user_id", maintain_order=True).head(10)
        )
        user_recs: dict[str, list] = defaultdict(list)
        for r in top10.select(["user_id", "item_id"]).iter_rows():
            user_recs[r[0]].append(r[1])
        recalls, ndcgs = [], []
        for uid in val_users:
            gt_set = val_gt.get(uid, set())
            if not gt_set:
                continue
            preds = user_recs.get(uid, [])
            recalls.append(recall_at_k(preds, gt_set, k=self.cfg.top_k))
            ndcgs.append(ndcg_at_k(preds, gt_set, k=self.cfg.top_k))
        log.info(
            f"  Val Recall@{self.cfg.top_k}: {np.mean(recalls):.4f}  "
            f"NDCG@{self.cfg.top_k}: {np.mean(ndcgs):.4f}  (n={len(recalls)})"
        )
