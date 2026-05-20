"""
DIAGNOSTIC 6: Validate Intent Matching Hypothesis with EDA
This script answers: 
1. If we extract a user's intent (District, Category, Price) from their pageviews, 
   does their future Ground Truth (GT) contact item match this intent?
2. If it matches, how often?
"""
import polars as pl
import os
from datetime import timedelta
from collections import defaultdict
from config.settings import PipelineConfig
from src.utils.logging import get_logger

logger = get_logger("diagnostic6")
c = PipelineConfig()
CACHE_DIR = ".cache"

def run_eda():
    logger.info("1. Loading validation contacts (Ground Truth)...")
    date_range = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))
    split_date = date_range["max_date"][0] - timedelta(days=c.validation_days)
    
    contacts = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
    val_contacts = contacts.filter(pl.col("last_date") > split_date)
    
    gt_pairs = {}
    for r in val_contacts.iter_rows(named=True):
        if r["user_id"] not in gt_pairs:
            gt_pairs[r["user_id"]] = set()
        gt_pairs[r["user_id"]].add(r["item_id"])
    
    val_users = list(gt_pairs.keys())
    logger.info(f"Found {len(val_users)} users with validation contacts.")

    logger.info("2. Loading item metadata (dim_listing)...")
    import glob
    dim_files = glob.glob(os.path.join(c.data.train_path, "dim_listing/*.parquet"))
    if not dim_files:
        dim_files = glob.glob(os.path.join(c.data.train_path, "dim_listing.parquet"))
    dim_items = pl.read_parquet(dim_files[0])
    
    item_profiles = {}
    for r in dim_items.select(["item_id", "district_name", "category", "price_bucket"]).iter_rows(named=True):
        item_profiles[r["item_id"]] = (r.get("district_name"), r.get("category"), r.get("price_bucket"))

    logger.info("3. Loading pageviews before the validation period...")
    # Only pageviews BEFORE the split date, to simulate what we know about the user at test time
    pvs_lazy = pl.scan_parquet(os.path.join(c.data.train_path, "fact_user_events/*.parquet")).filter(
        (pl.col("event_type") == "pageview") &
        (pl.col("event_ts") <= split_date) &
        (pl.col("event_ts") >= split_date - timedelta(days=14))
    ).select(["user_id", "item_id"])
    
    pvs = pvs_lazy.collect()
    
    logger.info("4. Building user intent profiles...")
    user_intents = defaultdict(lambda: defaultdict(int))
    for r in pvs.iter_rows(named=True):
        uid, iid = r["user_id"], r["item_id"]
        if uid in gt_pairs and iid in item_profiles:
            profile = item_profiles[iid]
            # Ignore null profiles
            if profile[0] is not None and profile[1] is not None:
                user_intents[uid][profile] += 1
                
    # Get top 3 intents per user (since user might search for multiple things)
    best_intents = {}
    for uid, profiles in user_intents.items():
        sorted_profiles = sorted(profiles.items(), key=lambda x: x[1], reverse=True)
        best_intents[uid] = [p[0] for p in sorted_profiles[:3]]
        
    users_with_intent = len(best_intents)
    logger.info(f"Users with extractable intent: {users_with_intent} / {len(val_users)}")

    logger.info("5. Checking if GT items match the extracted intents...")
    total_gt = 0
    gt_in_dim = 0
    match_top1 = 0
    match_top3 = 0
    
    for uid, gt_items in gt_pairs.items():
        if uid not in best_intents:
            continue
            
        user_top_intents = best_intents[uid]
        
        for iid in gt_items:
            total_gt += 1
            if iid in item_profiles:
                gt_in_dim += 1
                gt_profile = item_profiles[iid]
                if gt_profile == user_top_intents[0]:
                    match_top1 += 1
                if gt_profile in user_top_intents:
                    match_top3 += 1

    logger.info("=== EDA RESULTS ===")
    logger.info(f"Total GT contacts (for users with intent): {total_gt}")
    logger.info(f"GT items present in dim_listing: {gt_in_dim} ({gt_in_dim/max(1, total_gt)*100:.1f}%)")
    logger.info(f"GT items matching Top 1 Intent (District, Category, Price): {match_top1} ({match_top1/max(1, gt_in_dim)*100:.1f}% of active items)")
    logger.info(f"GT items matching Top 3 Intents (District, Category, Price): {match_top3} ({match_top3/max(1, gt_in_dim)*100:.1f}% of active items)")
    
    # Also check City + Category match
    match_city_cat = 0
    item_city_cat = {}
    for r in dim_items.select(["item_id", "city_name", "category"]).iter_rows(named=True):
        item_city_cat[r["item_id"]] = (r.get("city_name"), r.get("category"))
        
    user_city_cat = defaultdict(lambda: defaultdict(int))
    for r in pvs.iter_rows(named=True):
        uid, iid = r["user_id"], r["item_id"]
        if uid in gt_pairs and iid in item_city_cat:
            profile = item_city_cat[iid]
            if profile[0] is not None and profile[1] is not None:
                user_city_cat[uid][profile] += 1
                
    best_cc = {}
    for uid, profiles in user_city_cat.items():
        best_cc[uid] = max(profiles.items(), key=lambda x: x[1])[0]
        
    for uid, gt_items in gt_pairs.items():
        if uid not in best_cc: continue
        user_cc = best_cc[uid]
        for iid in gt_items:
            if iid in item_city_cat and item_city_cat[iid] == user_cc:
                match_city_cat += 1
                
    logger.info(f"GT items matching Top 1 (City, Category): {match_city_cat} ({match_city_cat/max(1, gt_in_dim)*100:.1f}% of active items)")

if __name__ == "__main__":
    run_eda()
