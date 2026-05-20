"""
scripts/inference.py — Generate submission.csv with format: user_id, rank, item_id

Pipeline V2: Cascade-based candidate generation
  Stage 1: CascadeCandidateGenerator
           Priority 1: PageviewReplay (Recall=0.197 standalone)
           Priority 2: CoContact expansion (Recall=0.107 standalone)
           Priority 3: Recent CC segment popular (last 7d)
           Priority 4: SegPop all-time fallback

Previous pipeline (ALS + LightGBM + Reranker) is preserved in the
EnsembleCandidateGenerator and can be activated via --legacy flag.
"""
import sys
import os
import argparse
import time
from collections import defaultdict

import polars as pl
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import PipelineConfig
from src.utils.logging import get_logger

logger = get_logger("inference")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")


def run_cascade_inference(args, config: PipelineConfig):
    """
    V2 Inference: Cascade-based candidate generation.
    Uses PageviewReplay → CoContact → RecentCC → SegPop cascade.
    """
    from src.models.candidates.pageview_replay import PageviewReplayRecommender
    from src.models.candidates.cocontact import CoContactRecommender
    from src.models.candidates.segment_popularity import SegmentPopularityRecommender
    from src.models.candidates.intent_recommender import IntentRecommender
    from src.models.ensemble.cascade_generator import CascadeCandidateGenerator

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("INFERENCE V2: PV-Replay → CoContact → RecentCC → SegPop")
    logger.info(f"  Hybrid rerank mode: {getattr(args, 'hybrid', False)}")
    logger.info("=" * 60)

    # ── 1. Load test users ────────────────────────────────────
    logger.info("[1/5] Loading test users...")
    test_users = pl.read_parquet(
        os.path.join(args.test_data_dir, "test_users.parquet")
    )["user_id"].to_list()
    test_user_set = set(test_users)
    logger.info(f"  Test users: {len(test_users):,}")

    # ── 2. Load contact history and compute preferences ───────
    logger.info("[2/5] Loading contact history & building preferences...")
    contacts = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))

    prefs_dict: dict[str, tuple] = {}
    prefs_df = (
        contacts.filter(pl.col("user_id").is_in(list(test_user_set)))
        .group_by("user_id")
        .agg([
            pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
            pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("pref_cat"),
        ])
    )
    for r in prefs_df.iter_rows(named=True):
        prefs_dict[r["user_id"]] = (r.get("pref_city"), r.get("pref_cat"))

    # Merge cold user prefs (from pageviews)
    cold_prefs_path = os.path.join(CACHE_DIR, "cold_user_prefs.parquet")
    if os.path.exists(cold_prefs_path):
        cold_prefs = pl.read_parquet(cold_prefs_path)
        n_before = len(prefs_dict)
        for r in cold_prefs.iter_rows(named=True):
            uid = r["user_id"]
            if uid not in prefs_dict and uid in test_user_set:
                prefs_dict[uid] = (r.get("pref_city"), r.get("pref_cat"))
        logger.info(f"  Cold user prefs merged: {len(prefs_dict) - n_before:,} additional")
    logger.info(f"  Total users with prefs: {len(prefs_dict):,}/{len(test_users):,}")

    # ── 3. Fit candidate sources ──────────────────────────────
    logger.info("[3/5] Fitting candidate sources...")

    # Load valid items from dim_listing
    import glob
    dim_files = glob.glob(os.path.join(args.data_dir, "dim_listing/*.parquet"))
    if not dim_files:
        dim_files = glob.glob(os.path.join(args.data_dir, "dim_listing.parquet"))
    valid_items = None
    if dim_files:
        df_listing = pl.scan_parquet(dim_files).collect()
        valid_items = set(df_listing["item_id"].to_list())
        logger.info(f"Loaded {len(valid_items):,} valid items from dim_listing")

    # Determine cutoff date from cached data
    date_range = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))
    max_date = date_range["max_date"][0]

    # Source 1: PageviewReplay (fit on raw events)
    pv_replay = PageviewReplayRecommender(window_days=14, max_items_per_user=50)
    events_path = os.path.join(args.data_dir, "fact_user_events/*.parquet")
    pv_replay.fit(events_path, user_ids=test_user_set, cutoff_date=max_date)

    # Source 2: CoContact graph
    cocontact = CoContactRecommender(window_days=30)
    cocontact.fit(contacts, cutoff_date=max_date)

    # Source 3: Recent CC (last 7 days)
    recent_cc = CascadeCandidateGenerator.build_recent_cc(
        contacts, cutoff_date=max_date, window_days=7, max_items_per_segment=200,
    )

    # Source 4: SegPop (load pre-trained)
    segpop = SegmentPopularityRecommender().load(
        os.path.join(args.model_dir, "segpop.pkl")
    )

    # Source 5: ALS (contact-based CF)
    from src.models.candidates.light_als import LightALSRecommender
    als = LightALSRecommender()
    als.load(os.path.join(args.model_dir, "als"))
    if als._matrix is None:
        als_contacts = pl.read_parquet(os.path.join(CACHE_DIR, "als_contact_pairs.parquet"))
        als.rebuild_matrix(als_contacts)

    # Source 6: ALS View (pageview-based CF — budget=0 but loaded for compatibility)
    als_view = None
    als_view_path = os.path.join(args.model_dir, "als_view")
    if os.path.exists(als_view_path):
        als_view = LightALSRecommender()
        als_view.load(als_view_path)
        if als_view._matrix is None:
            als_view_pairs_path = os.path.join(CACHE_DIR, "als_pageview_pairs.parquet")
            if os.path.exists(als_view_pairs_path):
                als_view_contacts = pl.read_parquet(als_view_pairs_path)
                als_view.rebuild_matrix(als_view_contacts)

    # User histories for CoContact seeding
    user_histories = CascadeCandidateGenerator.build_user_histories(
        contacts, user_ids=test_user_set, max_items=20,
    )

    # Source 7: IntentRecommender
    intent_rec = IntentRecommender(max_items_per_intent=200)
    pvs_lazy = pl.scan_parquet(events_path).filter(
        (pl.col("event_ts") <= max_date) & 
        (pl.col("event_ts") >= max_date - pl.duration(days=14)) &
        (pl.col("event_type") == "pageview")
    ).select(["user_id", "item_id"]).collect()
    
    if dim_files:
        intent_rec.fit(pvs=pvs_lazy, dim_listing=df_listing, valid_items=valid_items)

    # Source 8: UserKNN
    from src.models.candidates.user_knn import UserKNNRecommender
    user_knn = UserKNNRecommender(max_neighbors_per_item=30)
    user_knn.fit(contacts.lazy(), query_user_ids=test_user_set, valid_items=valid_items)

    # Source 9: SellerExpansion
    from src.models.candidates.seller_recommender import SellerExpansionRecommender
    seller_rec = SellerExpansionRecommender(max_items_per_seller=50)
    seller_rec.fit(contacts.lazy(), listing_df=df_listing, query_user_ids=test_user_set)

    # Item-to-city mapping
    item_cities = dict(zip(df_listing["item_id"], df_listing["city_name"])) if dim_files else {}

    # ── 4. Build cascade generator and predict ────────────────
    logger.info("[4/5] Generating predictions...")

    # Hybrid mode: generate k=200 candidates, rerank with LightGBM
    use_hybrid = getattr(args, 'hybrid', False)
    pool_k = 200 if use_hybrid else config.top_k

    if use_hybrid:
        from src.models.rankers.lgbm_ranker import LambdarankLGBMRanker
        from src.features.feature_engineer import FeatureEngineer
        from src.features.extractors.recent_history import RecentHistoryExtractor
        from src.features.extractors.seller_affinity import SellerAffinityExtractor
        from src.features.extractors.preference_match import PreferenceMatchExtractor

        ranker = LambdarankLGBMRanker()
        ranker.load(args.model_dir)
        logger.info(f"  LightGBM ranker loaded: {len(ranker.feature_cols)} features")

        user_stats_df = pl.read_parquet(os.path.join(args.model_dir, "user_stats.parquet"))
        item_stats_df = pl.read_parquet(os.path.join(args.model_dir, "item_stats.parquet"))
        item_meta_df  = pl.read_parquet(os.path.join(args.model_dir, "item_meta.parquet"))
        logger.info(f"  Feature tables loaded: users={len(user_stats_df):,}, items_stats={len(item_stats_df):,}, items_meta={len(item_meta_df):,}")

        recent_ext = RecentHistoryExtractor(contacts)
        seller_ext = SellerAffinityExtractor(contacts, df_listing)
        match_ext  = PreferenceMatchExtractor()
        feature_eng = FeatureEngineer(extractors=[recent_ext, seller_ext, match_ext])

    cascade = CascadeCandidateGenerator(
        pv_replay=pv_replay,
        cocontact=cocontact,
        segpop=segpop,
        als=als,
        als_view=als_view,
        recent_cc=recent_cc,
        user_histories=user_histories,
        intent_rec=intent_rec,
        user_knn=user_knn,
        seller_rec=seller_rec,
        item_cities=item_cities,
    )

    batch_size = config.cand_batch
    all_rows: list[dict] = []

    for batch_start in range(0, len(test_users), batch_size):
        batch = test_users[batch_start:batch_start + batch_size]

        recs = cascade.generate_batch(
            user_ids=batch,
            user_prefs=prefs_dict,
            k=pool_k,
            valid_items=valid_items,
        )

        if use_hybrid:
            # Build (user_id, item_id) pairs DataFrame for feature attachment
            pair_rows = []
            for uid in batch:
                items = recs.get(uid, [])
                for iid in items:
                    pair_rows.append({"user_id": uid, "item_id": iid})
            if not pair_rows:
                continue

            df_pairs = pl.DataFrame(pair_rows)

            # Attach ALS scores (GPU-safe: convert factors to numpy first)
            try:
                uf = als._model.user_factors.to_numpy()
                itf = als._model.item_factors.to_numpy()
            except AttributeError:
                uf = als._model.user_factors
                itf = als._model.item_factors
            als_scores = []
            is_from_als = []
            for r in df_pairs.iter_rows(named=True):
                uid_idx = als._u2i.get(r["user_id"])
                iid_idx = als._i2i.get(r["item_id"])
                if uid_idx is not None and iid_idx is not None:
                    als_scores.append(float(np.dot(uf[uid_idx], itf[iid_idx])))
                    is_from_als.append(1.0)
                else:
                    als_scores.append(0.0)
                    is_from_als.append(0.0)
            df_pairs = df_pairs.with_columns([
                pl.Series("score_als", als_scores),
                pl.lit(0.0).alias("score_view_als"),
                pl.lit(0.0).alias("score_segpop"),
                pl.Series("is_from_als", is_from_als),
                pl.lit(0.0).alias("is_from_view_als"),
                pl.lit(0.0).alias("is_from_segpop"),
            ])

            # Attach feature tables
            df_pairs = feature_eng.attach_features_inference(df_pairs, user_stats_df, item_stats_df, item_meta_df)

            # Ensure all feature columns exist
            for fc in ranker.feature_cols:
                if fc not in df_pairs.columns:
                    df_pairs = df_pairs.with_columns(pl.lit(0.0).alias(fc))

            # Score with LightGBM
            scores = ranker.predict(df_pairs)
            df_pairs = df_pairs.with_columns(pl.Series("lgbm_score", scores.tolist()))

            # Select top-k per user by lgbm_score
            df_ranked = df_pairs.sort(["user_id", "lgbm_score"], descending=[False, True])
            for uid in batch:
                user_df = df_ranked.filter(pl.col("user_id") == uid)
                items = user_df.head(config.top_k)["item_id"].to_list()
                # Pad if needed
                if len(items) < config.top_k:
                    seen = set(items)
                    for pad_item in cascade._segpop._global:
                        if pad_item not in seen and (valid_items is None or pad_item in valid_items):
                            items.append(pad_item)
                            seen.add(pad_item)
                            if len(items) >= config.top_k:
                                break
                for rank, iid in enumerate(items[:config.top_k], start=1):
                    all_rows.append({"user_id": uid, "rank": rank, "item_id": iid})
        else:
            # Direct cascade mode (no reranking)
            for uid in batch:
                items = recs.get(uid, [])
                # Pad with global popular items if needed
                if len(items) < config.top_k:
                    seen = set(items)
                    for pad_item in cascade._segpop._global:
                        if pad_item not in seen and (valid_items is None or pad_item in valid_items):
                            items.append(pad_item)
                            seen.add(pad_item)
                            if len(items) >= config.top_k:
                                break
                for rank, iid in enumerate(items[:config.top_k], start=1):
                    all_rows.append({"user_id": uid, "rank": rank, "item_id": iid})

        if (batch_start // batch_size + 1) % 10 == 0 or batch_start + batch_size >= len(test_users):
            logger.info(f"  Processed {min(batch_start + batch_size, len(test_users)):,}/{len(test_users):,}")

    # ── 5. Write submission ───────────────────────────────────
    logger.info("[5/5] Writing submission.csv...")
    df_sub = pl.DataFrame(all_rows).select(["user_id", "rank", "item_id"])
    df_sub = df_sub.with_columns(pl.Series("ID", range(1, len(df_sub) + 1)))
    df_sub = df_sub.select(["ID", "user_id", "rank", "item_id"])
    df_sub.write_csv(args.output_file)

    elapsed = (time.time() - t0) / 60
    logger.info(f"Submission saved: {args.output_file} ({len(df_sub):,} rows, {elapsed:.1f} min)")
    logger.info(f"Users covered: {df_sub['user_id'].n_unique():,}/{len(test_users):,}")
    logger.info(f"Unique items recommended: {df_sub['item_id'].n_unique():,}")
    logger.info("=" * 60)


def run_legacy_inference(args, config: PipelineConfig):
    """
    V1 Inference: EnsembleGen → FeatureEng → LGBMRanker → Reranker.
    Preserved for backward compatibility and A/B comparison.
    """
    from src.features.feature_engineer import FeatureEngineer
    from src.features.extractors.recent_history import RecentHistoryExtractor
    from src.features.extractors.seller_affinity import SellerAffinityExtractor
    from src.features.extractors.preference_match import PreferenceMatchExtractor
    from src.models.candidates.light_als import LightALSRecommender
    from src.models.candidates.segment_popularity import SegmentPopularityRecommender
    from src.models.ensemble.ensemble_generator import EnsembleCandidateGenerator
    from src.models.rankers.lgbm_ranker import LambdarankLGBMRanker
    from src.models.rerankers.multi_objective import MultiObjectiveReranker

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("INFERENCE LEGACY: EnsembleGen → FeatureEng → LGBMRanker → Reranker")
    logger.info("=" * 60)

    # Load models
    logger.info("[1/6] Loading models...")
    segpop = SegmentPopularityRecommender().load(os.path.join(args.model_dir, "segpop.pkl"))
    als = LightALSRecommender()
    als.load(os.path.join(args.model_dir, "als"))
    als_view = LightALSRecommender()
    als_view.load(os.path.join(args.model_dir, "als_view"))
    ranker = LambdarankLGBMRanker()
    ranker.load(args.model_dir)
    feature_cols = ranker.feature_cols
    logger.info(f"  Ranker loaded: {len(feature_cols)} feature cols")

    # Load lookup tables
    logger.info("[2/6] Loading lookup tables...")
    user_stats_df = pl.read_parquet(os.path.join(args.model_dir, "user_stats.parquet"))
    item_stats_df = pl.read_parquet(os.path.join(args.model_dir, "item_stats.parquet"))
    item_meta_df  = pl.read_parquet(os.path.join(args.model_dir, "item_meta.parquet"))
    valid_items   = set(item_meta_df["item_id"].to_list())

    if als._matrix is None:
        als_contacts = pl.read_parquet(os.path.join(CACHE_DIR, "als_contact_pairs.parquet"))
        als.rebuild_matrix(als_contacts)
    if als_view._matrix is None:
        pv = pl.read_parquet(os.path.join(CACHE_DIR, "als_pageview_pairs.parquet"))
        als_view.rebuild_matrix(pv.select(["user_id", "item_id", "view_count"]).rename({"view_count": "score"}))

    # Load test users + prefs
    logger.info("[3/6] Loading test users...")
    test_users = pl.read_parquet(os.path.join(args.test_data_dir, "test_users.parquet"))["user_id"].to_list()
    contacts = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
    test_user_set = set(test_users)
    prefs_df = (
        contacts.filter(pl.col("user_id").is_in(list(test_user_set)))
        .group_by("user_id")
        .agg([
            pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
            pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("pref_cat"),
        ])
    )
    prefs_dict = {}
    for r in prefs_df.iter_rows(named=True):
        prefs_dict[r["user_id"]] = (r.get("pref_city"), r.get("pref_cat"))
    cold_prefs_path = os.path.join(CACHE_DIR, "cold_user_prefs.parquet")
    if os.path.exists(cold_prefs_path):
        cold_prefs = pl.read_parquet(cold_prefs_path)
        for r in cold_prefs.iter_rows(named=True):
            uid = r["user_id"]
            if uid not in prefs_dict and uid in test_user_set:
                prefs_dict[uid] = (r.get("pref_city"), r.get("pref_cat"))
    logger.info(f"  Users with prefs: {len(prefs_dict):,}/{len(test_users):,}")

    # Build pipeline
    logger.info("[4/6] Building pipeline...")
    ensemble_gen = EnsembleCandidateGenerator(
        als=als, als_view=als_view, segpop=segpop,
        n_cand_als=config.model.n_cand_als,
        n_cand_view_als=config.model.n_cand_view_als,
        n_cand_segpop=config.model.n_cand_segpop,
    )
    recent_ext = RecentHistoryExtractor(contacts)
    seller_ext = SellerAffinityExtractor(contacts, pl.scan_parquet(
        os.path.join(args.data_dir, "dim_listing/*.parquet")).collect())
    match_ext  = PreferenceMatchExtractor()
    feature_eng = FeatureEngineer(extractors=[recent_ext, seller_ext, match_ext])
    reranker = MultiObjectiveReranker(
        alpha=config.reranker.alpha, beta=config.reranker.beta,
        gamma=config.reranker.gamma, delta=config.reranker.delta,
        epsilon=config.reranker.epsilon,
    )

    # Batch inference
    logger.info("[5/6] Generating predictions...")
    batch_size = config.cand_batch
    all_rows = []
    predicted_users = set()
    for batch_start in range(0, len(test_users), batch_size):
        batch = test_users[batch_start:batch_start + batch_size]
        df_batch, _ = ensemble_gen.generate_batch(
            users=batch, user_prefs=prefs_dict, valid_items=valid_items,
        )
        if len(df_batch) == 0:
            continue
        df_batch = feature_eng.attach_features_inference(df_batch, user_stats_df, item_stats_df, item_meta_df)
        scores = ranker.predict(df_batch)
        df_batch = df_batch.with_columns(pl.Series("lgbm_score", scores.tolist()))
        df_reranked = reranker.rerank_batch(df_batch, k=config.top_k)
        user_rank_counter = defaultdict(int)
        for r in df_reranked.select(["user_id", "item_id"]).iter_rows():
            uid, iid = r[0], r[1]
            if user_rank_counter[uid] < config.top_k:
                user_rank_counter[uid] += 1
                all_rows.append({"user_id": uid, "rank": user_rank_counter[uid], "item_id": iid})
                predicted_users.add(uid)
        if (batch_start // batch_size + 1) % 5 == 0:
            logger.info(f"  Processed {min(batch_start + batch_size, len(test_users)):,}/{len(test_users):,}")

    # Cold-start fallback
    logger.info("[6/6] Writing submission.csv...")
    cold_users = [u for u in test_users if u not in predicted_users]
    if cold_users:
        global_top = segpop._global[:config.top_k]
        for uid in cold_users:
            for rank, iid in enumerate(global_top, start=1):
                all_rows.append({"user_id": uid, "rank": rank, "item_id": iid})

    df_sub = pl.DataFrame(all_rows).select(["user_id", "rank", "item_id"])
    df_sub = df_sub.with_columns(pl.Series("ID", range(1, len(df_sub) + 1)))
    df_sub = df_sub.select(["ID", "user_id", "rank", "item_id"])
    df_sub.write_csv(args.output_file)

    elapsed = (time.time() - t0) / 60
    logger.info(f"Submission saved: {args.output_file} ({len(df_sub):,} rows, {elapsed:.1f} min)")
    logger.info(f"Users covered: {df_sub['user_id'].n_unique():,}/{len(test_users):,}")
    logger.info("=" * 60)


def main():
    config = PipelineConfig()
    parser = argparse.ArgumentParser(description="Generate submission.csv")
    parser.add_argument("--test_data_dir", default=config.data.test_path)
    parser.add_argument("--data_dir", default=config.data.train_path)
    parser.add_argument("--model_dir", default="outputs/models/")
    parser.add_argument("--output_file", default="submission.csv")
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use legacy pipeline (ALS + LightGBM + Reranker) instead of cascade."
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Use hybrid mode: cascade k=200 candidates + LightGBM reranker."
    )
    args = parser.parse_args()

    if args.legacy:
        run_legacy_inference(args, config)
    else:
        run_cascade_inference(args, config)


if __name__ == "__main__":
    main()
