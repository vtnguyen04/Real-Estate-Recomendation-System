"""
DIAGNOSTIC 5: Validate Intent (Ward + Price + Category) matching
"""
import polars as pl
import numpy as np
import os
from datetime import timedelta
from collections import defaultdict
from src.evaluation.metrics import recall_at_k
from config.settings import PipelineConfig

c = PipelineConfig()
CACHE_DIR = ".cache"

# Load GT
date_range = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))
split_date = date_range["max_date"][0] - timedelta(days=c.validation_days)

contacts = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
val_contacts = contacts.filter(pl.col("last_date") > split_date)
gt = defaultdict(set)
for r in val_contacts.iter_rows(named=True):
    gt[r["user_id"]].add(r["item_id"])

val_users = list(gt.keys())[:3000]

# Load dim_listing
import glob
file = glob.glob(os.path.join(c.data.train_path, "dim_listing/*.parquet"))[0]
dim_items = pl.read_parquet(file).select([
    "item_id", "city_name", "district_name", "ward_name", "price_bucket", "category"
]).drop_nulls()

item_dict = {}
for r in dim_items.iter_rows(named=True):
    item_dict[r["item_id"]] = (r["ward_name"], r["price_bucket"], r["category"])

# Load PVs
events_scan = pl.scan_parquet(os.path.join(c.data.train_path, "fact_user_events/*.parquet")).filter(
    (pl.col("user_id").is_in(val_users)) & 
    (pl.col("event_ts") <= split_date) & 
    (pl.col("event_ts") >= split_date - timedelta(days=14)) &
    (pl.col("event_type") == "pageview")
).select(["user_id", "item_id"])
pvs = events_scan.collect()

# Build User Intent Profiles
user_profiles = defaultdict(lambda: defaultdict(int))
for r in pvs.iter_rows(named=True):
    uid, iid = r["user_id"], r["item_id"]
    if iid in item_dict:
        profile = item_dict[iid]
        user_profiles[uid][profile] += 1

# Best profile per user
best_profile = {}
for uid, profiles in user_profiles.items():
    best_profile[uid] = max(profiles.items(), key=lambda x: x[1])[0]

print(f"Users with profiles: {len(best_profile)} / {len(val_users)}")

# How many GT items match the user's best profile?
hits = 0
total_gt = 0
for uid in val_users:
    user_gt = gt[uid]
    total_gt += len(user_gt)
    if uid in best_profile:
        bp = best_profile[uid]
        for gt_iid in user_gt:
            if gt_iid in item_dict and item_dict[gt_iid] == bp:
                hits += 1

print(f"GT items matching exact (Ward, Price, Cat) intent: {hits} / {total_gt} ({hits/max(1,total_gt)*100:.1f}%)")
