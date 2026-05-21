import sys
import os
import argparse
import time
from datetime import timedelta
from collections import defaultdict

import polars as pl
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import PipelineConfig
from src.evaluation.metrics import recall_at_k, ndcg_at_k
from src.utils.logging import get_logger

logger = get_logger("evaluate")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")


def predict_users_legacy(
    users: list,
    ensemble_gen,
    ranker,
    reranker,
    feature_eng,
    user_stats_df: pl.DataFrame,
    item_stats_df: pl.DataFrame,
    item_meta_df: pl.DataFrame,
    valid_items: set,
    prefs_dict: dict,
    batch_size: int = 2_000,
) -> tuple[dict[str, list], dict[str, list]]:
    """Legacy predict: EnsembleGen → FeatureEng → LGBMRanker → Reranker."""
    from src.models.ensemble.ensemble_generator import EnsembleCandidateGenerator
    user_recs: dict[str, list] = defaultdict(list)
    user_cands_all = {}

    for start_idx in range(0, len(users), batch_size):
        batch = users[start_idx:start_idx + batch_size]
        df_batch, batch_cands = ensemble_gen.generate_batch(
            users=batch, user_prefs=prefs_dict, valid_items=valid_items,
        )
        user_cands_all.update(batch_cands)
        df_batch = feature_eng.attach_features_inference(
            df_batch, user_stats_df, item_stats_df, item_meta_df
        )
        scores = ranker.predict(df_batch)
        df_batch = df_batch.with_columns(pl.Series("lgbm_score", scores.tolist()))
        df_reranked = reranker.rerank_batch(df_batch, k=10)
        for r in df_reranked.select(["user_id", "item_id"]).iter_rows():
            user_recs[r[0]].append(r[1])

    return dict(user_recs), user_cands_all


def predict_users_cascade(
    users: list,
    contacts: pl.DataFrame,
    split_date,
    config: "PipelineConfig",
    data_dir: str,
    model_dir: str,
    prefs_dict: dict,
    df_listing: pl.DataFrame,
    hybrid: bool = False,
) -> dict[str, list]:
    """Cascade predict aligned with production InferencePipeline flow."""
    from src.models.candidates.pageview_replay import PageviewReplayRecommender
    from src.models.candidates.cocontact import CoContactRecommender
    from src.models.candidates.segment_popularity import SegmentPopularityRecommender
    from src.models.candidates.intent_recommender import IntentRecommender
    from src.models.candidates.user_knn import UserKNNRecommender
    from src.models.candidates.seller_recommender import SellerExpansionRecommender
    from src.models.ensemble.cascade_generator import CascadeCandidateGenerator

    cc = config.cascade
    user_set = set(users)
    train_contacts = contacts.filter(pl.col("last_date") <= split_date)
    valid_items = set(df_listing["item_id"].to_list())

    events_path = os.path.join(data_dir, "fact_user_events/*.parquet")

    # PageviewReplay — use config window
    pv_replay = PageviewReplayRecommender(
        window_days=cc.pv_window_days, max_items_per_user=cc.pv_max_items_per_user,
    )
    pv_replay.fit(events_path, user_ids=user_set, cutoff_date=split_date)

    # CoContact
    cocontact = CoContactRecommender(window_days=cc.cocontact_window_days)
    cocontact.fit(train_contacts, cutoff_date=split_date)

    # RecentCC
    recent_cc = CascadeCandidateGenerator.build_recent_cc(
        train_contacts, cutoff_date=split_date, window_days=cc.recent_cc_window_days,
    )

    # SegPop
    segpop = SegmentPopularityRecommender().load(os.path.join(model_dir, "segpop.pkl"))

    # ContactALS
    from src.models.candidates.light_als import LightALSRecommender
    als = LightALSRecommender()
    als.load(os.path.join(model_dir, "als"))
    if als._matrix is None:
        als_rebuild_file = "als_weighted_contact.parquet" if config.model.als_use_weighted else "als_contact_pairs.parquet"
        als_rebuild_path = os.path.join(CACHE_DIR, als_rebuild_file)
        if not os.path.exists(als_rebuild_path):
            als_rebuild_path = os.path.join(CACHE_DIR, "als_contact_pairs.parquet")
        als_contacts = pl.read_parquet(als_rebuild_path)
        als.rebuild_matrix(als_contacts)

    # ViewALS — only if budget > 0
    als_view = None
    has_view_budget = (
        cc.budget_pool.get("als_view", 0) > 0
        or cc.budget_top10.get("als_view", 0) > 0
    )
    als_view_path = os.path.join(model_dir, "als_view")
    if has_view_budget and os.path.isdir(als_view_path):
        als_view = LightALSRecommender()
        als_view.load(als_view_path)
        if als_view._matrix is None:
            pv_data = pl.read_parquet(os.path.join(CACHE_DIR, "als_pageview_pairs.parquet"))
            als_view.rebuild_matrix(pv_data)
        logger.info("  ALS View loaded (budget > 0).")
    else:
        logger.info("  ALS View skipped (budget=0 or no artifact).")

    # User histories
    user_histories = CascadeCandidateGenerator.build_user_histories(
        train_contacts, user_ids=user_set, max_items=cc.user_history_max_items,
    )

    # Intent
    intent_rec = IntentRecommender(max_items_per_intent=cc.intent_max_items_per_intent)
    pvs_lazy = pl.scan_parquet(events_path).filter(
        (pl.col("event_ts") <= split_date) &
        (pl.col("event_ts") >= split_date - pl.duration(days=cc.pv_window_days)) &
        (pl.col("event_type") == "pageview")
    ).select(["user_id", "item_id"]).collect()
    intent_rec.fit(pvs=pvs_lazy, dim_listing=df_listing, valid_items=valid_items)

    # UserKNN
    user_knn = UserKNNRecommender(max_neighbors_per_item=cc.user_knn_max_neighbors)
    user_knn.fit(train_contacts.lazy(), query_user_ids=user_set, valid_items=valid_items)

    # SellerExpansion
    seller_rec = SellerExpansionRecommender(max_items_per_seller=cc.seller_max_items_per_seller)
    seller_rec.fit(train_contacts.lazy(), listing_df=df_listing, query_user_ids=user_set)

    # Item cities
    item_cities = dict(zip(df_listing["item_id"], df_listing["city_name"]))

    cascade = CascadeCandidateGenerator(
        pv_replay=pv_replay, cocontact=cocontact, segpop=segpop,
        als=als, als_view=als_view,
        recent_cc=recent_cc, user_histories=user_histories,
        intent_rec=intent_rec, user_knn=user_knn,
        seller_rec=seller_rec, item_cities=item_cities,
        cascade_cfg=cc,
    )

    if hybrid:
        # Hybrid mode: cascade k=pool → LGBM rerank → top 10
        # Mirrors production _rerank_batch_df() exactly
        from src.models.rankers.lgbm_ranker import LambdarankLGBMRanker
        from src.features.feature_engineer import FeatureEngineer
        from src.features.extractors.preference_match import PreferenceMatchExtractor
        from src.features.extractors.item_snapshot import ItemSnapshotExtractor

        ranker = LambdarankLGBMRanker()
        ranker.load(model_dir)
        user_stats_df = pl.read_parquet(os.path.join(model_dir, "user_stats.parquet"))
        item_stats_df = pl.read_parquet(os.path.join(model_dir, "item_stats.parquet"))
        item_meta_df  = pl.read_parquet(os.path.join(model_dir, "item_meta.parquet"))

        snapshot_path = os.path.join(model_dir, "snapshot_stats.parquet")
        snapshot_stats_df = None
        if os.path.exists(snapshot_path):
            snapshot_stats_df = pl.read_parquet(snapshot_path)

        snapshot_ext = ItemSnapshotExtractor(snapshot_path)
        feature_eng = FeatureEngineer(extractors=[
            PreferenceMatchExtractor(), snapshot_ext,
        ])
        logger.info(f"  Hybrid mode: ranker with {len(ranker.feature_cols)} features")

        user_recs: dict[str, list] = {}
        batch_size = 2_000
        for start_idx in range(0, len(users), batch_size):
            batch = users[start_idx:start_idx + batch_size]
            df_batch = cascade.generate_batch_with_sources(
                user_ids=batch, user_prefs=prefs_dict,
                k=cc.hybrid_pool_size, valid_items=valid_items,
            )
            if len(df_batch) == 0:
                continue

            # Ensure score_segpop exists
            if "score_segpop" not in df_batch.columns:
                df_batch = df_batch.with_columns(pl.lit(0.0).alias("score_segpop"))

            # Feature tables
            df_batch = feature_eng.attach_features_inference(
                df_batch, user_stats_df, item_stats_df, item_meta_df,
            )

            # Join snapshot features (same as production)
            if snapshot_stats_df is not None:
                df_batch = df_batch.join(snapshot_stats_df, on="item_id", how="left")

            # Fill missing feature cols with 0 (same as production)
            for fc in ranker.feature_cols:
                if fc not in df_batch.columns:
                    df_batch = df_batch.with_columns(pl.lit(0.0).alias(fc))

            # Score + rank (use full df, not subset)
            scores = ranker.predict(df_batch)
            df_batch = df_batch.with_columns(pl.Series("lgbm_score", scores.tolist()))
            top10 = (
                df_batch.sort(["user_id", "lgbm_score"], descending=[False, True])
                .group_by("user_id", maintain_order=True).head(10)
            )
            for r in top10.select(["user_id", "item_id"]).iter_rows():
                user_recs.setdefault(r[0], []).append(r[1])
        return user_recs
    else:
        # Direct cascade top-10
        return cascade.generate_batch(
            user_ids=users, user_prefs=prefs_dict, k=10, valid_items=valid_items,
        )


def main():
    config = PipelineConfig()
    parser = argparse.ArgumentParser(description="Evaluate offline Recall@10 / NDCG@10")
    parser.add_argument("--model_dir", default="outputs/models/")
    parser.add_argument("--data_dir", default=config.data.train_path)
    parser.add_argument("--n_users", type=int, default=0,
                        help="Number of val users to sample (0 = all)")
    parser.add_argument("--cascade", action="store_true",
                        help="Use cascade pipeline instead of legacy")
    parser.add_argument("--hybrid", action="store_true",
                        help="Cascade + LGBM rerank (requires --cascade)")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("EVALUATE: Offline Recall@10 / NDCG@10")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────
    contacts   = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
    date_range = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))
    df_listing_path = os.path.join(args.data_dir, "dim_listing")
    if os.path.isdir(df_listing_path):
        df_listing = pl.scan_parquet(os.path.join(df_listing_path, "*.parquet")).collect()
    else:
        df_listing = pl.read_parquet(df_listing_path + ".parquet")

    max_date   = date_range["max_date"][0]
    split_date = max_date - timedelta(days=config.validation_days)
    logger.info(f"  Split date: {split_date}")

    train_contacts = contacts.filter(pl.col("last_date") <= split_date)
    val_contacts   = contacts.filter(pl.col("last_date") > split_date)

    gt: dict[str, set] = defaultdict(set)
    for r in val_contacts.iter_rows(named=True):
        gt[r["user_id"]].add(r["item_id"])
    all_val_users = list(gt.keys())
    logger.info(f"  Val users total: {len(all_val_users):,}")

    n_sample = args.n_users if args.n_users > 0 else len(all_val_users)
    rng = np.random.default_rng(42)
    if n_sample < len(all_val_users):
        val_users = rng.choice(all_val_users, size=n_sample, replace=False).tolist()
    else:
        val_users = all_val_users
    logger.info(f"  Evaluating on {len(val_users):,} users")

    # User prefs from train contacts (no leakage)
    prefs_df = (
        train_contacts.filter(pl.col("user_id").is_in(val_users))
        .group_by("user_id")
        .agg([
            pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
            pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("pref_cat"),
        ])
    )
    prefs_dict: dict[str, tuple] = {}
    for r in prefs_df.iter_rows(named=True):
        prefs_dict[r["user_id"]] = (r.get("pref_city"), r.get("pref_cat"))

    # Merge cold user prefs
    cold_prefs_path = os.path.join(CACHE_DIR, "cold_user_prefs.parquet")
    if os.path.exists(cold_prefs_path):
        cold_prefs = pl.read_parquet(cold_prefs_path)
        val_set = set(val_users)
        n_before = len(prefs_dict)
        for r in cold_prefs.iter_rows(named=True):
            uid = r["user_id"]
            if uid not in prefs_dict and uid in val_set:
                prefs_dict[uid] = (r.get("pref_city"), r.get("pref_cat"))
        logger.info(f"  Cold user prefs merged: {len(prefs_dict) - n_before:,} additional users")
    logger.info(f"  Total users with prefs: {len(prefs_dict):,}/{len(val_users):,}")

    # ── Predict (cascade or legacy) ───────────────────────────
    logger.info("Generating predictions...")
    user_cands = None

    if args.cascade:
        user_recs = predict_users_cascade(
            val_users, contacts, split_date,
            config=config,
            data_dir=args.data_dir, model_dir=args.model_dir,
            prefs_dict=prefs_dict, df_listing=df_listing,
            hybrid=args.hybrid,
        )
    else:
        # Legacy mode: load ALS + LightGBM + Reranker
        from src.models.candidates.light_als import LightALSRecommender
        from src.models.candidates.segment_popularity import SegmentPopularityRecommender
        from src.models.ensemble.ensemble_generator import EnsembleCandidateGenerator
        from src.models.rankers.lgbm_ranker import LambdarankLGBMRanker
        from src.models.rerankers.multi_objective import MultiObjectiveReranker
        from src.features.feature_engineer import FeatureEngineer
        from src.features.extractors.recent_history import RecentHistoryExtractor
        from src.features.extractors.seller_affinity import SellerAffinityExtractor
        from src.features.extractors.preference_match import PreferenceMatchExtractor

        segpop = SegmentPopularityRecommender().load(os.path.join(args.model_dir, "segpop.pkl"))
        als = LightALSRecommender()
        als.load(os.path.join(args.model_dir, "als"))
        if als._matrix is None:
            als_contacts = pl.read_parquet(os.path.join(CACHE_DIR, "als_contact_pairs.parquet"))
            als.rebuild_matrix(als_contacts)
        als_view = None
        als_view_path = os.path.join(args.model_dir, "als_view")
        if os.path.isdir(als_view_path):
            als_view = LightALSRecommender()
            als_view.load(als_view_path)
            if als_view._matrix is None:
                pv_path = os.path.join(CACHE_DIR, "als_pageview_pairs.parquet")
                if os.path.exists(pv_path):
                    pv_data = pl.read_parquet(pv_path)
                    als_view.rebuild_matrix(pv_data)
        else:
            logger.info("  als_view artifact not found, skipping.")
        ranker = LambdarankLGBMRanker()
        ranker.load(args.model_dir)
        logger.info(f"  Loaded ranker with {len(ranker.feature_cols)} feature cols")

        user_stats_df = pl.read_parquet(os.path.join(args.model_dir, "user_stats.parquet"))
        item_stats_df = pl.read_parquet(os.path.join(args.model_dir, "item_stats.parquet"))
        item_meta_df  = pl.read_parquet(os.path.join(args.model_dir, "item_meta.parquet"))
        valid_items   = set(item_meta_df["item_id"].to_list())

        df_listing_path = os.path.join(args.data_dir, "dim_listing")
        if os.path.isdir(df_listing_path):
            df_listing = pl.scan_parquet(os.path.join(df_listing_path, "*.parquet")).collect()
        else:
            df_listing = pl.read_parquet(df_listing_path + ".parquet")

        ensemble_gen = EnsembleCandidateGenerator(
            als=als, als_view=als_view, segpop=segpop,
            n_cand_als=config.model.n_cand_als,
            n_cand_view_als=config.model.n_cand_view_als,
            n_cand_segpop=config.model.n_cand_segpop,
        )
        recent_ext = RecentHistoryExtractor(train_contacts)
        seller_ext = SellerAffinityExtractor(train_contacts, df_listing)
        match_ext  = PreferenceMatchExtractor()
        feature_eng = FeatureEngineer(extractors=[recent_ext, seller_ext, match_ext])
        reranker = MultiObjectiveReranker(
            alpha=config.reranker.alpha, beta=config.reranker.beta,
            gamma=config.reranker.gamma, delta=config.reranker.delta,
            epsilon=config.reranker.epsilon,
        )

        user_recs, user_cands = predict_users_legacy(
            val_users, ensemble_gen, ranker, reranker, feature_eng,
            user_stats_df, item_stats_df, item_meta_df,
            valid_items, prefs_dict,
        )

    # ── Coverage analysis (legacy mode only) ──────────────────
    if user_cands is not None:
        cov_recalls = []
        for uid in val_users:
            actual = gt.get(uid, set())
            if not actual:
                continue
            cands = set(user_cands.get(uid, []))
            cov_recalls.append(len(cands & actual) / len(actual))
        logger.info("=" * 60)
        logger.info(f"Candidate coverage (ceiling Recall): {np.mean(cov_recalls):.4f}")
        logger.info("=" * 60)

    valid_items = set(df_listing["item_id"].to_list())
    recalls, ndcgs = [], []
    active_recalls, active_ndcgs = [], []
    ceiling_recalls, active_ceiling_recalls = [], []
    cold_start_count = 0
    for uid in val_users:
        actual = gt.get(uid, set())
        if not actual:
            continue
            
        actual_active = {item for item in actual if item in valid_items}
        
        preds = user_recs.get(uid, [])
        if not preds:
            cold_start_count += 1
            
        recalls.append(recall_at_k(preds, actual, k=config.top_k))
        ndcgs.append(ndcg_at_k(preds, actual, k=config.top_k))
        ceiling_recalls.append(recall_at_k(preds, actual, k=200))
        
        if actual_active:
            active_recalls.append(recall_at_k(preds, actual_active, k=config.top_k))
            active_ndcgs.append(ndcg_at_k(preds, actual_active, k=config.top_k))
            active_ceiling_recalls.append(recall_at_k(preds, actual_active, k=200))

    elapsed = (time.time() - t0) / 60
    logger.info("=" * 60)
    logger.info(f"Recall@{config.top_k} (Raw GT)    : {np.mean(recalls):.4f}")
    logger.info(f"Recall@200(Raw GT)    : {np.mean(ceiling_recalls):.4f}")
    logger.info(f"NDCG@{config.top_k}   (Raw GT)    : {np.mean(ndcgs):.4f}")
    if active_recalls:
        logger.info(f"Recall@{config.top_k} (Active GT) : {np.mean(active_recalls):.4f}")
        logger.info(f"Recall@200(Active GT) : {np.mean(active_ceiling_recalls):.4f}")
        logger.info(f"NDCG@{config.top_k}   (Active GT) : {np.mean(active_ndcgs):.4f}")
    logger.info(f"Users eval: {len(val_users):,} | Cold-start (no cands): {cold_start_count}")
    logger.info(f"Time       : {elapsed:.1f} min")
    logger.info("=" * 60)

    # Per-segment breakdown by preferred category
    cat_recalls: dict = defaultdict(list)
    for uid in val_users:
        actual = gt.get(uid, set())
        if not actual:
            continue
        pref_cat = prefs_dict.get(uid, (None, None))[1]
        preds = user_recs.get(uid, [])
        cat_recalls[pref_cat].append(recall_at_k(preds, actual, k=config.top_k))
    logger.info(f"Recall@{config.top_k} by preferred category:")
    for cat, scores in sorted(cat_recalls.items(), key=lambda x: str(x[0])):
        logger.info(f"  cat={str(cat):>6}: {np.mean(scores):.4f}  (n={len(scores)})")


if __name__ == "__main__":
    main()
