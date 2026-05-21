"""
Round 24: Test-aligned evaluation — split-clean offline eval.

Fixes leakage and distribution issues from the first attempt:
1. Classifies users by EVENTS (not just contacts) before split
2. Builds cold prefs from PCI filtered to <= split_date (no future leak)
3. Supports split-clean ALS/SegPop retrain with --retrain_clean
4. Tests both cascade-direct (k=10) and hybrid (k=200 → LGBM → top 10)

Usage:
  bash scripts/run_gpu.sh python scripts/evaluate_aligned.py --n_total 10000
  bash scripts/run_gpu.sh python scripts/evaluate_aligned.py --n_total 10000 --hybrid
"""
import sys, os, argparse, time, gc
from datetime import timedelta, date as dateclass
from collections import defaultdict

import polars as pl
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import PipelineConfig
from src.utils.logging import get_logger

logger = get_logger("eval_aligned")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")


# ═══════════════════════════════════════════════════════════════
# Step 1: Classify users PROPERLY (events, not just contacts)
# ═══════════════════════════════════════════════════════════════
def classify_val_users(
    val_user_ids: list[str],
    contacts: pl.DataFrame,
    split_date,
    data_dir: str,
    pci_path: str,
) -> dict:
    """
    Classify val users into 3 groups based on data BEFORE split_date:
    - warm: has contact history (in contact_pairs with last_date <= split)
    - cold_with_signal: no contacts, but has login event signal or PCI before split
    - truly_blind: zero events, zero PCI before split
    """
    val_set = set(val_user_ids)

    # 1. Users with contact history before split
    train_contacts = contacts.filter(pl.col("last_date") <= split_date)
    contact_users = set(train_contacts["user_id"].unique().to_list()) & val_set
    logger.info(f"    Contact users (before split): {len(contact_users):,}")

    # 2. Users with any login event before split (matching production signal policy)
    remaining = val_set - contact_users
    events_path = os.path.join(data_dir, "fact_user_events", "*.parquet")
    pv_users_df = (
        pl.scan_parquet(events_path)
        .filter(
            (pl.col("date") <= split_date) &
            (pl.col("is_login") == "login") &
            (pl.col("user_id").is_in(remaining))
        )
        .select("user_id")
        .unique()
        .collect()
    )
    pageview_users = set(pv_users_df["user_id"].to_list())
    logger.info(f"    Login-signal/no-contact users (before split): {len(pageview_users):,}")

    # 3. Users with PCI data before split
    remaining2 = remaining - pageview_users
    pci_users_df = (
        pl.scan_parquet(pci_path)
        .filter(
            (pl.col("date") <= split_date) &
            (pl.col("user_id").is_in(remaining2))
        )
        .select("user_id")
        .unique()
        .collect()
    )
    pci_only_users = set(pci_users_df["user_id"].to_list())
    logger.info(f"    PCI-only users (before split): {len(pci_only_users):,}")

    # 4. Truly blind
    cold_with_signal = pageview_users | pci_only_users
    truly_blind = remaining2 - pci_only_users

    return {
        "warm": sorted(contact_users),
        "cold_with_signal": sorted(cold_with_signal),
        "truly_blind": sorted(truly_blind),
    }


# ═══════════════════════════════════════════════════════════════
# Step 2: Build cold prefs from PCI BEFORE split (no leak)
# ═══════════════════════════════════════════════════════════════
def build_split_clean_cold_prefs(
    user_ids: set[str],
    pci_path: str,
    split_date,
    df_listing: pl.DataFrame,
) -> dict[str, tuple]:
    """Build city+category prefs from PCI data BEFORE split_date only."""
    pci = (
        pl.scan_parquet(pci_path)
        .filter(
            (pl.col("date") <= split_date) &
            (pl.col("user_id").is_in(user_ids)) &
            (pl.col("lead_count") > 0)  # Only real leads
        )
        .select(["user_id", "item_id", "category", "lead_count"])
        .collect()
    )

    if len(pci) == 0:
        return {}

    # Get city from dim_listing
    item_city = dict(zip(df_listing["item_id"], df_listing["city_name"]))

    prefs: dict[str, tuple] = {}
    for uid, group in pci.group_by("user_id"):
        uid_str = uid[0] if isinstance(uid, tuple) else uid
        items = group["item_id"].to_list()
        cats = group["category"].to_list()

        # Mode category
        from collections import Counter
        cat_counter = Counter(cats)
        mode_cat = cat_counter.most_common(1)[0][0] if cat_counter else None

        # Mode city (from dim_listing)
        cities = [item_city.get(it) for it in items if item_city.get(it)]
        city_counter = Counter(cities)
        mode_city = city_counter.most_common(1)[0][0] if city_counter else None

        prefs[uid_str] = (mode_city, mode_cat)

    return prefs


def build_split_clean_weighted_als_pairs(
    events_path: str, pci_path: str, split_date, als_use_weighted: bool = True, use_pci: bool = True,
) -> pl.DataFrame:
    """
    Build weighted contact pairs from raw events <= split_date.
    Optionally enrich with PCI pairs <= split_date.
    """
    logger.info("  [RETRAIN_CLEAN] Building weighted ALS pairs from raw events...")
    events_lazy = pl.scan_parquet(events_path).filter(pl.col("date") <= split_date)
    
    if als_use_weighted:
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
        
    logger.info(f"    Raw event pairs: {len(events_pairs):,}")
    
    if use_pci and pci_path:
        logger.info("  [RETRAIN_CLEAN] Integrating PCI supplement pairs <= split_date...")
        import glob
        pci_files = glob.glob(pci_path)
        if pci_files:
            pci_lazy = pl.scan_parquet(pci_path).filter(
                (pl.col("date") <= split_date) & 
                (pl.col("lead_count") >= 1)
            )
            
            # Calculate weight: lead_count + 3.0 boost for purchased
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
            logger.info(f"    PCI raw supplement pairs: {len(pci_pairs):,}")
            
            # Apply INS-058 "existing_only" filter: keep only PCI pairs for users that already exist in ALS
            existing_users = set(events_pairs["user_id"].unique().to_list())
            pci_pairs = pci_pairs.filter(pl.col("user_id").is_in(list(existing_users)))
            logger.info(f"    PCI supplement pairs (existing_only): {len(pci_pairs):,}")
            
            if len(pci_pairs) > 0:
                # Merge pairs: concat and sum scores
                events_pairs = events_pairs.with_columns(pl.col("score").cast(pl.Float64))
                pci_pairs = pci_pairs.with_columns(pl.col("score").cast(pl.Float64))
                merged = pl.concat([events_pairs, pci_pairs])
                events_pairs = (
                    merged.group_by(["user_id", "item_id"])
                    .agg(pl.col("score").sum())
                    .with_columns(pl.col("score").cast(pl.Float32))
                )
                logger.info(f"    Total merged ALS pairs: {len(events_pairs):,}")
        else:
            logger.warning("    No PCI files found at pci_path.")
        
    return events_pairs


# ═══════════════════════════════════════════════════════════════
# Step 3: Cascade builder (same API as evaluate.py)
# ═══════════════════════════════════════════════════════════════
def build_cascade_and_predict(
    users, config, data_dir, model_dir, split_date,
    contacts, df_listing, prefs_dict,
    hybrid=False, k=10, retrain_clean=False,
):
    """Build cascade and generate predictions. Mirrors evaluate.py exactly."""
    from src.models.candidates.pageview_replay import PageviewReplayRecommender
    from src.models.candidates.cocontact import CoContactRecommender
    from src.models.candidates.segment_popularity import SegmentPopularityRecommender
    from src.models.candidates.intent_recommender import IntentRecommender
    from src.models.candidates.user_knn import UserKNNRecommender
    from src.models.candidates.seller_recommender import SellerExpansionRecommender
    from src.models.ensemble.cascade_generator import CascadeCandidateGenerator
    from src.models.candidates.light_als import LightALSRecommender

    cc = config.cascade
    user_set = set(users)
    train_contacts = contacts.filter(pl.col("last_date") <= split_date)
    valid_items = set(df_listing["item_id"].to_list())
    events_path = os.path.join(data_dir, "fact_user_events/*.parquet")

    # ── Fit sources ──
    pv_replay = PageviewReplayRecommender(
        window_days=cc.pv_window_days, max_items_per_user=cc.pv_max_items_per_user,
    )
    pv_replay.fit(events_path, user_ids=user_set, cutoff_date=split_date)

    cocontact = CoContactRecommender(window_days=cc.cocontact_window_days)
    cocontact.fit(train_contacts, cutoff_date=split_date)

    recent_cc = CascadeCandidateGenerator.build_recent_cc(
        train_contacts, cutoff_date=split_date, window_days=cc.recent_cc_window_days,
    )

    # SegPop: retrain on split-clean events or load pre-trained
    if retrain_clean:
        logger.info("  [RETRAIN_CLEAN] Fitting SegPop on events <= split_date ...")
        segpop = SegmentPopularityRecommender()
        events_lazy = pl.scan_parquet(events_path).filter(pl.col("date") <= split_date)
        segpop.fit(events_lazy, valid_items=valid_items, listing_df=df_listing)
    else:
        segpop = SegmentPopularityRecommender().load(os.path.join(model_dir, "segpop.pkl"))

    snapshot_path = os.path.join(data_dir, "fact_listing_snapshot/*.parquet")
    try:
        snapshot_blind = (
            pl.scan_parquet(snapshot_path)
            .filter(
                (pl.col("date") <= split_date)
                & (pl.col("date") >= split_date - pl.duration(days=7))
            )
            .select(["item_id", "views_24h", "contacts_24h"])
            .collect()
        )
        segpop.set_blind_global_from_snapshot(snapshot_blind, valid_items=valid_items)
    except Exception as e:
        logger.warning(f"  Snapshot blind fallback unavailable: {e}")

    # ALS: retrain on split-clean contacts or load pre-trained
    if retrain_clean:
        logger.info("  [RETRAIN_CLEAN] Training ALS on weighted + PCI supplement <= split_date ...")
        als = LightALSRecommender(
            factors=config.model.als_factors,
            iterations=config.model.als_iterations,
            regularization=config.model.als_regularization,
            use_gpu=True,
        )
        pci_path = os.path.join(data_dir, "fact_post_contact_interactions/*.parquet")
        als_pairs = build_split_clean_weighted_als_pairs(
            events_path, pci_path, split_date,
            als_use_weighted=config.model.als_use_weighted,
            use_pci=True,
        )
        als.fit(als_pairs.lazy())
    else:
        als = LightALSRecommender()
        als.load(os.path.join(model_dir, "als"))
        if als._matrix is None:
            als_file = "als_weighted_contact.parquet" if config.model.als_use_weighted else "als_contact_pairs.parquet"
            als_path = os.path.join(CACHE_DIR, als_file)
            if not os.path.exists(als_path):
                als_path = os.path.join(CACHE_DIR, "als_contact_pairs.parquet")
            als.rebuild_matrix(pl.read_parquet(als_path))

    user_histories = CascadeCandidateGenerator.build_user_histories(
        train_contacts, user_ids=user_set, max_items=cc.user_history_max_items,
    )

    intent_rec = IntentRecommender(max_items_per_intent=cc.intent_max_items_per_intent)
    pvs_lazy = pl.scan_parquet(events_path).filter(
        (pl.col("event_ts") <= split_date) &
        (pl.col("event_ts") >= split_date - pl.duration(days=cc.pv_window_days)) &
        (pl.col("event_type") == "pageview")
    ).select(["user_id", "item_id"]).collect()
    intent_rec.fit(pvs=pvs_lazy, dim_listing=df_listing, valid_items=valid_items)

    user_knn = UserKNNRecommender(max_neighbors_per_item=cc.user_knn_max_neighbors)
    user_knn.fit(train_contacts.lazy(), query_user_ids=user_set, valid_items=valid_items)

    seller_rec = SellerExpansionRecommender(max_items_per_seller=cc.seller_max_items_per_seller)
    seller_rec.fit(train_contacts.lazy(), listing_df=df_listing, query_user_ids=user_set)

    item_cities = dict(zip(df_listing["item_id"], df_listing["city_name"]))

    cascade = CascadeCandidateGenerator(
        pv_replay=pv_replay, cocontact=cocontact, segpop=segpop,
        als=als, als_view=None,
        recent_cc=recent_cc, user_histories=user_histories,
        intent_rec=intent_rec, user_knn=user_knn,
        seller_rec=seller_rec, item_cities=item_cities,
        cascade_cfg=cc,
    )

    # ── Generate predictions ──
    if hybrid:
        from src.models.rankers.lgbm_ranker import LambdarankLGBMRanker
        from src.features.feature_engineer import FeatureEngineer
        from src.features.extractors.preference_match import PreferenceMatchExtractor
        from src.features.extractors.item_snapshot import ItemSnapshotExtractor

        ranker = LambdarankLGBMRanker()
        ranker.load(model_dir)
        user_stats_df = pl.read_parquet(os.path.join(model_dir, "user_stats.parquet"))
        item_stats_df = pl.read_parquet(os.path.join(model_dir, "item_stats.parquet"))
        item_meta_df = pl.read_parquet(os.path.join(model_dir, "item_meta.parquet"))

        snapshot_path = os.path.join(model_dir, "snapshot_stats.parquet")
        snapshot_stats_df = None
        if os.path.exists(snapshot_path):
            snapshot_stats_df = pl.read_parquet(snapshot_path)

        snapshot_ext = ItemSnapshotExtractor(snapshot_path)
        feature_eng = FeatureEngineer(extractors=[
            PreferenceMatchExtractor(), snapshot_ext,
        ])
        logger.info(f"  Hybrid mode (Segmented Policy): ranker with {len(ranker.feature_cols)} features")

        # Classify users into warm
        warm_user_set = set(train_contacts["user_id"].unique().to_list()) & user_set

        user_recs: dict[str, list] = {}
        batch_size = 2_000
        for start_idx in range(0, len(users), batch_size):
            batch = users[start_idx:start_idx + batch_size]
            batch_warm = [u for u in batch if u in warm_user_set]
            batch_cold = [u for u in batch if u not in warm_user_set]

            # 1. Warm users -> LGBM reranking
            if batch_warm:
                df_batch = cascade.generate_batch_with_sources(
                    user_ids=batch_warm, user_prefs=prefs_dict,
                    k=cc.hybrid_pool_size, valid_items=valid_items,
                )
                if len(df_batch) > 0:
                    if "score_segpop" not in df_batch.columns:
                        df_batch = df_batch.with_columns(pl.lit(0.0).alias("score_segpop"))

                    df_batch = feature_eng.attach_features_inference(
                        df_batch, user_stats_df, item_stats_df, item_meta_df,
                    )

                    if snapshot_stats_df is not None:
                        df_batch = df_batch.join(snapshot_stats_df, on="item_id", how="left")

                    for fc in ranker.feature_cols:
                        if fc not in df_batch.columns:
                            df_batch = df_batch.with_columns(pl.lit(0.0).alias(fc))

                    scores = ranker.predict(df_batch)
                    df_batch = df_batch.with_columns(pl.Series("lgbm_score", scores.tolist()))
                    top10 = (
                        df_batch.sort(["user_id", "lgbm_score"], descending=[False, True])
                        .group_by("user_id", maintain_order=True).head(10)
                    )
                    for r in top10.select(["user_id", "item_id"]).iter_rows():
                        user_recs.setdefault(r[0], []).append(r[1])

            # 2. Cold/Blind users -> Direct cascade
            if batch_cold:
                recs_cold = cascade.generate_batch(
                    user_ids=batch_cold, user_prefs=prefs_dict,
                    k=10, valid_items=valid_items,
                )
                for uid, items in recs_cold.items():
                    user_recs[uid] = list(items)

        return user_recs
    else:
        # Direct cascade top-10
        return cascade.generate_batch(
            user_ids=users, user_prefs=prefs_dict,
            k=k, valid_items=valid_items,
        )


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    config = PipelineConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=config.data.train_path)
    parser.add_argument("--model_dir", default="outputs/models/")
    parser.add_argument("--pci_path", default=os.path.join(config.data.train_path,
                        "fact_post_contact_interactions/*.parquet"))
    parser.add_argument("--blind_ratio", type=float, default=0.564)
    parser.add_argument("--n_total", type=int, default=10000)
    parser.add_argument("--hybrid", action="store_true",
                        help="Cascade k=200 → LGBM rerank (production mode)")
    parser.add_argument("--retrain_clean", action="store_true",
                        help="Retrain ALS+SegPop on split-clean data (fix INS-066 leak)")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("ROUND 24: TEST-ALIGNED EVALUATION (split-clean)")
    logger.info(f"  Mode: {'hybrid (cascade→LGBM)' if args.hybrid else 'cascade-direct (k=10)'}")
    logger.info("=" * 60)

    # ── Load data ──
    contacts = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
    date_range = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))

    df_listing_path = os.path.join(args.data_dir, "dim_listing")
    if os.path.isdir(df_listing_path):
        df_listing = pl.scan_parquet(os.path.join(df_listing_path, "*.parquet")).collect()
    else:
        df_listing = pl.read_parquet(df_listing_path + ".parquet")

    max_date = date_range["max_date"][0]
    split_date = max_date - timedelta(days=config.validation_days)
    logger.info(f"  Split: {split_date}, Max: {max_date}")

    # FILTER df_listing to posted_date <= split_date (prevent future inventory leak)
    original_listing_len = len(df_listing)
    df_listing = df_listing.filter(pl.col("posted_date") <= split_date)
    logger.info(f"  Cleaned listing inventory (posted_date <= {split_date}): {len(df_listing):,} (removed {original_listing_len - len(df_listing):,} future listings)")

    train_contacts = contacts.filter(pl.col("last_date") <= split_date)
    val_contacts = contacts.filter(pl.col("last_date") > split_date)

    # GT
    gt: dict[str, set] = defaultdict(set)
    for r in val_contacts.iter_rows(named=True):
        gt[r["user_id"]].add(r["item_id"])
    all_gt_users = list(gt.keys())
    logger.info(f"  Val GT users: {len(all_gt_users):,}")

    # ── Classify users properly ──
    logger.info("  Classifying val GT users by pre-split data...")
    groups = classify_val_users(
        all_gt_users, contacts, split_date,
        args.data_dir, args.pci_path,
    )
    warm_pool = groups["warm"]
    cold_pool = groups["cold_with_signal"]
    blind_pool = groups["truly_blind"]

    logger.info(f"  Classification:")
    logger.info(f"    Warm (has contacts): {len(warm_pool):,} ({len(warm_pool)/len(all_gt_users)*100:.1f}%)")
    logger.info(f"    Cold+signal (login/PCI, no contacts): {len(cold_pool):,} ({len(cold_pool)/len(all_gt_users)*100:.1f}%)")
    logger.info(f"    Truly blind: {len(blind_pool):,} ({len(blind_pool)/len(all_gt_users)*100:.1f}%)")

    # ── Sample to match test distribution ──
    rng = np.random.default_rng(42)
    # Test distribution: ~36% warm, ~7.7% cold_with_signal, ~56.4% blind
    # But we only have as many blind-with-GT as exist
    n_warm = min(len(warm_pool), int(args.n_total * 0.360))
    n_cold = min(len(cold_pool), int(args.n_total * 0.077))
    n_blind = min(len(blind_pool), args.n_total - n_warm - n_cold)

    # If not enough blind, take more warm/cold
    actual_total = n_warm + n_cold + n_blind
    if actual_total < args.n_total:
        extra = args.n_total - actual_total
        n_warm = min(len(warm_pool), n_warm + extra)
        actual_total = n_warm + n_cold + n_blind

    sampled_warm = rng.choice(warm_pool, size=n_warm, replace=False).tolist() if n_warm > 0 else []
    sampled_cold = rng.choice(cold_pool, size=n_cold, replace=False).tolist() if n_cold > 0 else []
    sampled_blind = rng.choice(blind_pool, size=n_blind, replace=False).tolist() if n_blind > 0 else []

    all_eval_users = sampled_warm + sampled_cold + sampled_blind
    logger.info(f"  Eval sample: warm={len(sampled_warm)}, cold={len(sampled_cold)}, "
                f"blind={len(sampled_blind)}, total={len(all_eval_users)}")

    # ── Build prefs (split-clean) ──
    # Warm prefs from contacts before split
    prefs_df = (
        train_contacts.filter(pl.col("user_id").is_in(all_eval_users))
        .group_by("user_id")
        .agg([
            pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
            pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("pref_cat"),
        ])
    )
    prefs_dict: dict[str, tuple] = {}
    for r in prefs_df.iter_rows(named=True):
        prefs_dict[r["user_id"]] = (r.get("pref_city"), r.get("pref_cat"))
    logger.info(f"  Contact-based prefs: {len(prefs_dict):,}")

    # Cold prefs from PCI (split-clean, no future leak)
    no_prefs_users = set(all_eval_users) - set(prefs_dict.keys())
    pci_prefs: dict[str, tuple] = {}
    if no_prefs_users:
        pci_prefs = build_split_clean_cold_prefs(
            no_prefs_users, args.pci_path, split_date, df_listing,
        )
        prefs_dict.update(pci_prefs)
        logger.info(f"  PCI prefs (split-clean): {len(pci_prefs):,}")

    # Pageview-based prefs for remaining cold users (split-clean)
    still_no_prefs = no_prefs_users - set(pci_prefs.keys()) if no_prefs_users else set()
    if still_no_prefs:
        events_path = os.path.join(args.data_dir, "fact_user_events/*.parquet")
        pv_prefs_df = (
            pl.scan_parquet(events_path)
            .filter(
                (pl.col("date") <= split_date) &
                (pl.col("is_login") == "login") &
                (pl.col("event_type") == "pageview") &
                (pl.col("user_id").is_in(still_no_prefs))
            )
            .select(["user_id", "city_name", "category"])
            .collect()
        )
        if len(pv_prefs_df) > 0:
            pv_agg = (
                pv_prefs_df.group_by("user_id")
                .agg([
                    pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
                    pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("pref_cat"),
                ])
            )
            n_pv = 0
            for r in pv_agg.iter_rows(named=True):
                if r["user_id"] not in prefs_dict:
                    prefs_dict[r["user_id"]] = (r.get("pref_city"), r.get("pref_cat"))
                    n_pv += 1
            logger.info(f"  Pageview prefs (split-clean): {n_pv:,}")

    logger.info(f"  Total prefs: {len(prefs_dict):,}/{len(all_eval_users):,}")

    # ── Build cascade & predict ──
    logger.info("=" * 60)
    if args.retrain_clean:
        logger.info("✅ RETRAIN_CLEAN mode: ALS+SegPop trained on contacts <= split_date only.")
        logger.info("   No model leak. Metrics are TRUSTWORTHY.")
    else:
        logger.info("⚠️  NOTE: ALS/SegPop models were trained on FULL data (includes val period).")
        logger.info("    This is a known leak. All recall numbers are inflated.")
    logger.info("=" * 60)

    user_recs = build_cascade_and_predict(
        all_eval_users, config, args.data_dir, args.model_dir,
        split_date, contacts, df_listing, prefs_dict,
        hybrid=args.hybrid, k=10, retrain_clean=args.retrain_clean,
    )

    # ── Compute metrics ──
    logger.info("=" * 60)
    logger.info("RESULTS (Round 24)")
    logger.info("=" * 60)

    def compute_recall(users, label):
        recalls = []
        for uid in users:
            actual = gt.get(uid, set())
            if not actual:
                recalls.append(0.0)
                continue
            predicted = user_recs.get(uid, [])
            recalls.append(len(set(predicted) & actual) / min(len(actual), 10))
        mean_recall = np.mean(recalls) if recalls else 0.0
        n_gt = sum(1 for u in users if gt.get(u))
        logger.info(f"  {label:40s}: Recall@10={mean_recall:.4f}  "
                    f"(n={len(users):,}, GT={n_gt:,})")
        return mean_recall

    r_all = compute_recall(all_eval_users, "ALL (simulated LB)")
    r_warm = compute_recall(sampled_warm, "Warm (contact history)")
    r_cold = compute_recall(sampled_cold, "Cold-with-signal (login/PCI)")
    r_blind = compute_recall(sampled_blind, "Truly blind (zero events)")

    # Sub-segments
    logger.info("-" * 60)
    cold_prefs_users = [u for u in sampled_cold if u in prefs_dict]
    cold_no_prefs = [u for u in sampled_cold if u not in prefs_dict]
    if cold_prefs_users:
        compute_recall(cold_prefs_users, "  Cold + prefs")
    if cold_no_prefs:
        compute_recall(cold_no_prefs, "  Cold (no prefs)")

    blind_prefs_users = [u for u in sampled_blind if u in prefs_dict]
    blind_no_prefs = [u for u in sampled_blind if u not in prefs_dict]
    if blind_prefs_users:
        compute_recall(blind_prefs_users, "  Blind + prefs (unusual)")
    if blind_no_prefs:
        compute_recall(blind_no_prefs, "  Blind (truly no prefs)")

    # Warm category breakdown
    logger.info("-" * 60)
    for cat in [1010, 1020, 1030, 1040, 1050]:
        cat_users = [u for u in sampled_warm if prefs_dict.get(u, (None, None))[1] == cat]
        if cat_users:
            compute_recall(cat_users, f"  Warm cat={cat}")

    # Summary
    logger.info("=" * 60)
    w_frac = len(sampled_warm) / len(all_eval_users) if all_eval_users else 0
    c_frac = len(sampled_cold) / len(all_eval_users) if all_eval_users else 0
    b_frac = len(sampled_blind) / len(all_eval_users) if all_eval_users else 0
    logger.info(f"  SIMULATED LB: {r_all:.4f}")
    logger.info(f"  Decomposed: warm={r_warm:.4f}×{w_frac:.3f} "
                f"+ cold={r_cold:.4f}×{c_frac:.3f} "
                f"+ blind={r_blind:.4f}×{b_frac:.3f}")
    logger.info(f"  Mode: {'hybrid' if args.hybrid else 'cascade-direct'}")
    logger.info(f"  Time: {(time.time()-t0)/60:.1f} min")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
