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
from src.features.extractors.item_snapshot import ItemSnapshotExtractor
from src.features.feature_engineer import FeatureEngineer
from src.models.candidates.light_als import LightALSRecommender
from src.models.candidates.segment_popularity import SegmentPopularityRecommender
from src.models.candidates.pageview_replay import PageviewReplayRecommender
from src.models.candidates.cocontact import CoContactRecommender
from src.models.candidates.intent_recommender import IntentRecommender
from src.models.candidates.user_knn import UserKNNRecommender
from src.models.candidates.seller_recommender import SellerExpansionRecommender
from src.models.ensemble.ensemble_generator import EnsembleCandidateGenerator
from src.models.ensemble.cascade_generator import CascadeCandidateGenerator
from src.models.rankers.lgbm_ranker import LambdarankLGBMRanker
from src.evaluation.metrics import recall_at_k, ndcg_at_k
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Orchestrates ALS + SegPop + LightGBM lambdarank training.
    Supports both CascadeGen (INS-052 fix) and EnsembleGen (legacy) for candidate generation.

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
        self.cascade_gen: Optional[CascadeCandidateGenerator] = None
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
        Generate (user, item) candidate pairs with source tracking.
        Uses CascadeGen (cascade/hybrid) or EnsembleGen (legacy) to match inference distribution.
        """
        if self.cascade_gen is not None:
            # INS-052 FIX: Use SAME cascade generator as inference
            return self.cascade_gen.generate_batch_with_sources(
                user_ids=users,
                user_prefs=user_prefs,
                k=self.cfg.cascade.hybrid_pool_size,
                valid_items=valid_items,
                pos_set=pos_set,
                label_col=label_col,
            )
        elif self.ensemble_gen is not None:
            df, _ = self.ensemble_gen.generate_batch(
                users=users,
                user_prefs=user_prefs,
                valid_items=valid_items,
                pos_set=pos_set,
                label_col=label_col,
            )
            return df
        else:
            raise RuntimeError("No candidate generator available. Check inference_mode config.")

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
        # Save as segpop.pkl — the name inference expects
        self.segpop.save(os.path.join(self.output_dir, "segpop.pkl"))
        log.info("  SegPop saved as segpop.pkl.")

        # ── ContactALS ──────────────────────────────────────────
        # Build clean, non-leaked ALS training pairs <= split_date to prevent future-inventory/val leakage
        log.info(f"[3/8] Building clean, non-leaked ALS training pairs <= {split_date}...")
        events_path = os.path.join(cfg.data.train_path, "fact_user_events/*.parquet")
        pci_path = os.path.join(cfg.data.train_path, "fact_post_contact_interactions/*.parquet")
        
        events_lazy = pl.scan_parquet(events_path).filter(pl.col("date") <= split_date)
        
        if cfg.model.als_use_weighted:
            real_contacts = ["view_phone", "contact_chat", "contact_zalo", "contact_sms"]
            events_pairs = (
                events_lazy
                .filter(pl.col("is_login") == "login")
                .filter(pl.col("is_contact") == 1)
                .with_columns(
                    pl.when(pl.col("event_type").is_in(real_contacts))
                    .then(pl.lit(3.0))
                    .otherwise(pl.lit(1.0))
                    .alias("w")
                )
                .group_by(["user_id", "item_id"])
                .agg(pl.col("w").sum().alias("score"))
                .collect()
            )
        else:
            events_pairs = (
                events_lazy
                .filter(pl.col("is_login") == "login")
                .filter(pl.col("is_contact") == 1)
                .group_by(["user_id", "item_id"])
                .agg(pl.len().cast(pl.Float32).alias("score"))
                .collect()
            )
            
        log.info(f"    Raw event pairs: {len(events_pairs):,}")
        
        if cfg.pci_enabled and pci_path:
            import glob
            pci_files = glob.glob(pci_path)
            if pci_files:
                pci_lazy = pl.scan_parquet(pci_path).filter(
                    (pl.col("date") <= split_date) & 
                    (pl.col("lead_count") >= 1)
                )
                pci_pairs = (
                    pci_lazy
                    .with_columns(
                        pl.when(pl.col("purchased") == True)
                        .then(pl.col("lead_count") * 3.0)
                        .otherwise(pl.col("lead_count").cast(pl.Float64))
                        .alias("score")
                    )
                    .select(["user_id", "item_id", "score"])
                    .collect()
                )
                # Apply existing_only filter
                existing_users = set(events_pairs["user_id"].unique().to_list())
                pci_pairs = pci_pairs.filter(pl.col("user_id").is_in(list(existing_users)))
                
                if len(pci_pairs) > 0:
                    events_pairs = events_pairs.with_columns(pl.col("score").cast(pl.Float64))
                    pci_pairs = pci_pairs.with_columns(pl.col("score").cast(pl.Float64))
                    merged = pl.concat([events_pairs, pci_pairs])
                    events_pairs = (
                        merged.group_by(["user_id", "item_id"])
                        .agg(pl.col("score").sum())
                        .with_columns(pl.col("score").cast(pl.Float32))
                    )
                    log.info(f"    Total merged ALS pairs: {len(events_pairs):,}")
            else:
                log.warning("    No PCI files found.")
                
        als_training_data = events_pairs
        
        # Save this clean data to cache so it can be loaded for rebuilding matrix without leak
        clean_als_path = os.path.join(self.cache_dir, "clean_als_training_data.parquet")
        als_training_data.write_parquet(clean_als_path)
        log.info(f"  Saved clean ALS data to {clean_als_path}")

        log.info(
            f"  ContactALS (f={cfg.model.als_factors}, i={cfg.model.als_iterations}, gpu={self.use_gpu})"
        )
        self.als = LightALSRecommender(
            factors=cfg.model.als_factors,
            regularization=cfg.model.als_regularization,
            iterations=cfg.model.als_iterations,
            use_gpu=self.use_gpu,
        )
        self.als.fit(als_training_data.lazy())
        self.als.save(os.path.join(self.output_dir, "als"))
        log.info("  ContactALS saved.")
        del als_training_data

        # ── ViewALS (only if budget > 0) ──────────────────────────
        has_view_als = (
            cfg.cascade.budget_pool.get("als_view", 0) > 0
            or cfg.cascade.budget_top10.get("als_view", 0) > 0
        )
        if has_view_als:
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
        else:
            log.info("[4/8] ViewALS skipped (budget=0, INS-047).")
            # Clean up stale artifact to prevent inference from loading disabled model
            stale_als_view = os.path.join(self.output_dir, "als_view")
            if os.path.isdir(stale_als_view):
                import shutil
                shutil.rmtree(stale_als_view)
                log.info("  Removed stale als_view artifact.")
        del als_pageviews

        # Keep ContactALS sparse matrix in memory to avoid OOM during rebuild
        # self.als._matrix = None

        # Free raw input DataFrames no longer needed after ALS + split
        del als_contacts, contacts
        gc.collect()
        log.info("  Released ALS matrix + input DataFrames.")
        if cfg.inference_mode == "cascade":
            log.info("  Inference mode is cascade. Skipping LightGBM ranker training as requested.")
            return self

        if cfg.inference_mode in ("hybrid",):
            log.info("  Fitting FULL CascadeCandidateGenerator (INS-052 fix)")
            gc.collect()  # Free any leftover ALS build temporaries

            cc = cfg.cascade
            events_path = os.path.join(cfg.data.train_path, "fact_user_events/*.parquet")

            def _log_ram(label):
                import resource
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
                log.info(f"    RAM({label}): RSS={rss:.0f}MB")

            # PageviewReplay
            log.info("    Fitting PageviewReplay...")
            pv_replay = PageviewReplayRecommender(
                window_days=cc.pv_window_days, max_items_per_user=cc.pv_max_items_per_user)
            pv_replay.fit(events_path, user_ids=set(train_users), cutoff_date=split_date)
            gc.collect(); _log_ram("after_pv")

            # CoContact
            log.info("    Fitting CoContact...")
            cocontact = CoContactRecommender(window_days=cc.cocontact_window_days)
            cocontact.fit(train_contacts, cutoff_date=split_date)

            # RecentCC + User histories (lightweight dict ops)
            log.info("    Building RecentCC + UserHistories...")
            recent_cc = CascadeCandidateGenerator.build_recent_cc(
                train_contacts, cutoff_date=split_date,
                window_days=cc.recent_cc_window_days,
                max_items_per_segment=cc.recent_cc_max_items_per_segment,
            )
            user_histories = CascadeCandidateGenerator.build_user_histories(
                train_contacts, user_ids=set(train_users),
                max_items=cc.user_history_max_items,
            )
            gc.collect(); _log_ram("after_cc_hist")

            # Intent — loads PV events (can be large)
            log.info("    Fitting IntentRecommender...")
            intent_rec = IntentRecommender(max_items_per_intent=cc.intent_max_items_per_intent)
            pvs = (
                pl.scan_parquet(events_path)
                .filter(
                    (pl.col("event_ts") <= split_date)
                    & (pl.col("event_ts") >= split_date - pl.duration(days=cc.pv_window_days))
                    & (pl.col("event_type") == "pageview")
                )
                .select(["user_id", "item_id"]).collect()
            )
            log.info(f"    Intent PV events loaded: {len(pvs):,}")
            intent_rec.fit(pvs=pvs, dim_listing=df_listing, valid_items=valid_items)
            del pvs; gc.collect(); _log_ram("after_intent")

            # UserKNN — builds large Python dicts
            log.info("    Fitting UserKNN...")
            user_knn = UserKNNRecommender(max_neighbors_per_item=cc.user_knn_max_neighbors)
            user_knn.fit(train_contacts.lazy(), query_user_ids=set(train_users), valid_items=valid_items)
            gc.collect(); _log_ram("after_knn")

            # Seller
            log.info("    Fitting SellerExpansion...")
            seller_rec = SellerExpansionRecommender(max_items_per_seller=cc.seller_max_items_per_seller)
            seller_rec.fit(train_contacts.lazy(), listing_df=df_listing, query_user_ids=set(train_users))
            _log_ram("after_seller")

            item_cities = dict(zip(df_listing["item_id"].to_list(), df_listing["city_name"].to_list()))

            self.cascade_gen = CascadeCandidateGenerator(
                pv_replay=pv_replay, cocontact=cocontact, segpop=self.segpop,
                als=self.als, als_view=self.als_view,
                recent_cc=recent_cc, user_histories=user_histories,
                intent_rec=intent_rec, user_knn=user_knn,
                seller_rec=seller_rec, item_cities=item_cities,
                cascade_cfg=cc,
            )
            log.info("  Cascade fitted with ALL sources.")
        else:
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

        # F-012/F-031: Snapshot-based item features
        snapshot_path = os.path.join(self.cache_dir, "snapshot_stats.parquet")
        snapshot_ext = ItemSnapshotExtractor(snapshot_path)

        feature_eng    = FeatureEngineer([
            user_ext, item_stats_ext, item_meta_ext,
            recent_ext, seller_ext, match_ext, snapshot_ext,
        ])
        del als_pv_reload; gc.collect()

        user_stats_df = user_ext.build_feature_df()
        user_prefs: dict[str, tuple] = {
            r["user_id"]: (r.get("pref_city"), r.get("pref_cat"))
            for r in user_stats_df.iter_rows(named=True)
        }

        # ── Training candidates ──────────────────────────────────
        log.info("[6/8] Generating training candidates (ALS-first)...")

        # ContactALS matrix is already in memory, no need to rebuild
        log.info("  ContactALS matrix is already in memory. Rebuild skipped to save memory.")
        gc.collect()
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
        # Build and save feature DFs BEFORE releasing extractors
        log.info("[7/8] Saving feature artefacts + training LightGBM...")
        item_stats_df = item_stats_ext.build_feature_df()
        item_meta_df  = item_meta_ext.build_feature_df()
        user_stats_df.write_parquet(os.path.join(self.output_dir, "user_stats.parquet"))
        item_stats_df.write_parquet(os.path.join(self.output_dir, "item_stats.parquet"))
        item_meta_df.write_parquet(os.path.join(self.output_dir, "item_meta.parquet"))
        if os.path.exists(snapshot_path):
            import shutil
            shutil.copy2(snapshot_path, os.path.join(self.output_dir, "snapshot_stats.parquet"))
            log.info("  snapshot_stats.parquet copied to model dir")

        # Release heavy objects no longer needed for LGBM
        self.cascade_gen = None
        if hasattr(self, 'als') and self.als is not None:
            self.als._matrix = None
        del train_contacts, df_listing, item_stats_df, item_meta_df
        del user_ext, item_stats_ext, item_meta_ext, recent_ext, seller_ext, match_ext, snapshot_ext
        del feature_eng
        gc.collect()
        log.info("  Released cascade + extractors for LGBM.")

        r = cfg.ranker
        self.ranker = LambdarankLGBMRanker(
            feature_cols=available_feats,
            use_gpu=self.use_gpu,
            num_leaves=r.num_leaves,
            learning_rate=r.learning_rate,
            num_rounds=r.n_estimators,
            early_stopping=r.early_stopping_rounds,
            feature_fraction=r.feature_fraction,
            bagging_fraction=r.bagging_fraction,
            bagging_freq=r.bagging_freq,
            min_child_samples=r.min_child_samples,
            lambdarank_truncation_level=r.lambdarank_truncation_level,
        )
        self.ranker.fit(df_train, val_df=df_val)

        # ── Save artefacts ────────────────────────────────────────
        log.info("[8/8] Saving ranker model...")
        self.ranker.save(self.output_dir)
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
