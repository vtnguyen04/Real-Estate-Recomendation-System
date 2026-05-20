"""
Dry-run test: validates the full feature engineering flow end-to-end
on a tiny sample to catch crashes before real training.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import polars as pl
from datetime import timedelta
from config.settings import PipelineConfig

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")

def main():
    config = PipelineConfig()
    print("=" * 60)
    print("DRY-RUN: Feature Engineering Pipeline Validation")
    print("=" * 60)

    # 1. Load data
    print("[1/6] Loading cached data...")
    contacts = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
    date_range = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))
    als_pv = pl.read_parquet(os.path.join(CACHE_DIR, "als_pageview_pairs.parquet"))
    df_listing = pl.scan_parquet(os.path.join(config.data.train_path, "dim_listing/*.parquet")).collect()

    max_date = date_range["max_date"][0]
    split_date = max_date - timedelta(days=config.validation_days)
    pos_cutoff = split_date - timedelta(days=config.positive_window_days)
    train_contacts = contacts.filter(pl.col("last_date") <= split_date)
    print(f"  train_contacts: {len(train_contacts):,} rows")
    print(f"  df_listing: {len(df_listing):,} rows")
    print(f"  als_pv: {len(als_pv):,} rows")

    # 2. Instantiate extractors
    print("[2/6] Instantiating extractors...")
    from src.features.extractors.user_behavior import UserBehaviorExtractor
    from src.features.extractors.item_stats import ItemStatsExtractor
    from src.features.extractors.item_quality import ItemQualityExtractor
    from src.features.extractors.recent_history import RecentHistoryExtractor
    from src.features.extractors.seller_affinity import SellerAffinityExtractor
    from src.features.extractors.preference_match import PreferenceMatchExtractor
    from src.features.feature_engineer import FeatureEngineer

    user_ext = UserBehaviorExtractor(train_contacts, df_listing)
    item_stats_ext = ItemStatsExtractor(train_contacts, als_pv, pos_cutoff)
    item_meta_ext = ItemQualityExtractor(df_listing, split_date)
    recent_ext = RecentHistoryExtractor(train_contacts)
    seller_ext = SellerAffinityExtractor(train_contacts, df_listing)
    match_ext = PreferenceMatchExtractor()
    feature_eng = FeatureEngineer([user_ext, item_stats_ext, item_meta_ext, recent_ext, seller_ext, match_ext])
    print("  OK: 6 extractors instantiated")

    # 3. Build individual lookup tables
    print("[3/6] Building lookup DataFrames...")
    user_stats_df = user_ext.build_feature_df(None)
    item_stats_df = item_stats_ext.build_feature_df(None)
    item_meta_df = item_meta_ext.build_feature_df(None)
    print(f"  user_stats_df: {user_stats_df.shape} cols={user_stats_df.columns}")
    print(f"  item_stats_df: {item_stats_df.shape} cols={item_stats_df.columns}")
    print(f"  item_meta_df:  {item_meta_df.shape} cols={item_meta_df.columns}")
    print(f"  item_meta_df dtypes: {dict(zip(item_meta_df.columns, item_meta_df.dtypes))}")

    # 4. Create fake training pairs (10 users × small candidate set)
    print("[4/6] Creating tiny training pairs...")
    sample_users = train_contacts["user_id"].unique().head(10).to_list()
    sample_items = df_listing["item_id"].head(50).to_list()
    import itertools
    pairs_data = list(itertools.product(sample_users, sample_items))
    fake_pairs = pl.DataFrame({
        "user_id": [p[0] for p in pairs_data],
        "item_id": [p[1] for p in pairs_data],
        "score_als": [0.5] * len(pairs_data),
        "score_view_als": [0.3] * len(pairs_data),
        "score_segpop": [0.2] * len(pairs_data),
        "is_from_als": [1] * len(pairs_data),
        "is_from_view_als": [0] * len(pairs_data),
        "is_from_segpop": [0] * len(pairs_data),
        "label": [0] * len(pairs_data),
    })
    print(f"  fake_pairs: {fake_pairs.shape}")

    # 5. Test extract_for_training
    print("[5/6] Running extract_for_training (Training mode)...")
    result = feature_eng.extract_for_training(fake_pairs)
    print(f"  Result shape: {result.shape}")
    print(f"  Result columns ({len(result.columns)}): {result.columns}")
    print(f"  Result dtypes:")
    for c, d in zip(result.columns, result.dtypes):
        print(f"    {c:30s} -> {d}")

    # 6. Test attach_features_inference (Evaluate mode)
    print("[6/6] Running attach_features_inference (Inference mode)...")
    # Re-create feature_eng with only pairwise extractors (as evaluate.py does)
    feature_eng_eval = FeatureEngineer(extractors=[recent_ext, seller_ext, match_ext])
    infer_pairs = fake_pairs.drop("label")
    result_infer = feature_eng_eval.attach_features_inference(
        infer_pairs, user_stats_df, item_stats_df, item_meta_df
    )
    print(f"  Result shape: {result_infer.shape}")
    print(f"  Result columns ({len(result_infer.columns)}): {result_infer.columns}")

    # 7. Check that required feature_cols are present
    expected = config.ranker.feature_cols
    missing = [c for c in expected if c not in result.columns]
    missing_infer = [c for c in expected if c not in result_infer.columns]
    if missing:
        print(f"\n  ⚠️  MISSING in Training mode: {missing}")
    else:
        print(f"\n  ✅ All {len(expected)} feature_cols present in Training mode")
    if missing_infer:
        print(f"  ⚠️  MISSING in Inference mode: {missing_infer}")
    else:
        print(f"  ✅ All {len(expected)} feature_cols present in Inference mode")

    # 8. Test Categorical conversion (as LightGBM will see it)
    print("\n[BONUS] Testing LightGBM-ready feature preparation...")
    from src.utils.polars_utils import prepare_features_for_lgbm
    available = [c for c in expected if c in result.columns]
    X = prepare_features_for_lgbm(result, available)
    print(f"  X shape: {X.shape}, dtypes: {dict(X.dtypes)}")
    assert X.isnull().sum().sum() == 0, "FAIL: Nulls remain after _prepare_features!"
    print("  ✅ No nulls. LightGBM-ready.")

    print("\n" + "=" * 60)
    print("DRY-RUN PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
