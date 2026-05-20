"""
DIAGNOSTIC: Understand what strategies can reach 0.3 Recall@10
==============================================================
Tests multiple candidate generation strategies on val set.
No LightGBM, no feature eng — pure candidate coverage analysis.
"""
import polars as pl
import numpy as np
import os
from datetime import timedelta
from collections import defaultdict
from config.settings import PipelineConfig

c = PipelineConfig()
CACHE_DIR = ".cache"
TRAIN_PATH = c.data.train_path

print("=" * 70)
print("DIAGNOSTIC: What would it take to reach 0.3 Recall@10?")
print("=" * 70)

# ── Load data ────────────────────────────────────────────────
print("\n[1] Loading data...")
contacts = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
date_range = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))
max_date = date_range["max_date"][0]
split_date = max_date - timedelta(days=c.validation_days)

df_listing = pl.scan_parquet(
    os.path.join(TRAIN_PATH, "dim_listing/*.parquet")
).select(["item_id", "city_name", "category", "district_name",
           "price_bucket", "posted_date", "seller_type",
           "images_count", "legal_status", "furnishing"]).collect()

train_contacts = contacts.filter(pl.col("last_date") <= split_date)
val_contacts = contacts.filter(pl.col("last_date") > split_date)

gt = defaultdict(set)
for r in val_contacts.iter_rows(named=True):
    gt[r["user_id"]].add(r["item_id"])
all_val_users = list(gt.keys())

# Sample for speed
rng = np.random.default_rng(42)
val_users = rng.choice(all_val_users, size=min(5000, len(all_val_users)), replace=False).tolist()

# Build user preferences from TRAIN contacts
train_user_set = set(train_contacts["user_id"].to_list())
prefs_df = (
    train_contacts.filter(pl.col("user_id").is_in(val_users))
    .join(df_listing.select(["item_id", "district_name"]), on="item_id", how="left")
    .group_by("user_id")
    .agg([
        pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
        pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("pref_cat"),
        pl.col("district_name").drop_nulls().mode().first().alias("pref_district"),
    ])
)
prefs = {}
for r in prefs_df.iter_rows(named=True):
    prefs[r["user_id"]] = (r["pref_city"], r["pref_cat"], r["pref_district"])

# Also load cold user prefs
cold_prefs = pl.read_parquet(os.path.join(CACHE_DIR, "cold_user_prefs.parquet"))
for r in cold_prefs.filter(pl.col("user_id").is_in(val_users)).iter_rows(named=True):
    if r["user_id"] not in prefs:
        prefs[r["user_id"]] = (r["pref_city"], r["pref_cat"], None)

print(f"  Val users: {len(val_users)}, with prefs: {len(prefs)}")

# ── Analyze GT items ─────────────────────────────────────────
print("\n[2] Analyzing GT items...")
gt_items = set()
for u in val_users:
    gt_items.update(gt.get(u, set()))

gt_in_listing = gt_items & set(df_listing["item_id"].to_list())
print(f"  Total GT items: {len(gt_items)}")
print(f"  GT items in dim_listing: {len(gt_in_listing)} ({100*len(gt_in_listing)/max(1,len(gt_items)):.1f}%)")

gt_listing = df_listing.filter(pl.col("item_id").is_in(list(gt_items)))
print(f"  GT posted_date range: {gt_listing['posted_date'].min()} → {gt_listing['posted_date'].max()}")
gt_ages = (split_date - gt_listing["posted_date"]).dt.total_days()
print(f"  GT item age (days from split): median={gt_ages.median():.0f}, mean={gt_ages.mean():.0f}, p90={gt_ages.quantile(0.9):.0f}")

# ── Analyze contact recency ──────────────────────────────────
print("\n[3] Analyzing recent contacts...")
# How many GT items received contacts in last N days before split?
for window in [7, 14, 30, 60, 90]:
    cutoff = split_date - timedelta(days=window)
    recent = train_contacts.filter(pl.col("last_date") > cutoff)
    recent_items = set(recent["item_id"].to_list())
    overlap = gt_items & recent_items
    print(f"  Items contacted in last {window:2d}d: {len(recent_items):,} | GT overlap: {len(overlap):,} ({100*len(overlap)/max(1,len(gt_items)):.1f}%)")

# ── Strategy tests ───────────────────────────────────────────
def eval_strategy(name, user_recs, val_users, gt):
    hits = 0
    total_gt = 0
    for u in val_users:
        user_gt = gt.get(u, set())
        if not user_gt:
            continue
        total_gt += len(user_gt)
        recs = user_recs.get(u, [])[:10]
        hits += len(set(recs) & user_gt)
    recall = hits / len(val_users)
    cov_items = set()
    for recs in user_recs.values():
        cov_items.update(recs)
    print(f"  {name:50s} | Recall@10: {recall:.4f} | Unique items: {len(cov_items):,}")
    return recall

print("\n[4] Testing candidate strategies...")

# Strategy A: Top popular items in (city, category) from ALL contacts
print("\n--- Strategy A: SegPop CC (all-time contacts, top-K per segment) ---")
cc_pop = (
    train_contacts
    .filter(pl.col("city_name").is_not_null() & pl.col("category").is_not_null())
    .group_by(["city_name", "category", "item_id"])
    .agg(pl.len().alias("c"))
    .sort(["city_name", "category", "c"], descending=[False, False, True])
)
cc_map = defaultdict(list)
for r in cc_pop.iter_rows(named=True):
    key = (r["city_name"], r["category"])
    if len(cc_map[key]) < 1000:
        cc_map[key].append(r["item_id"])

for k in [10, 50, 100, 200, 500]:
    user_recs = {}
    for u in val_users:
        if u in prefs:
            city, cat, dist = prefs[u]
            items = cc_map.get((city, cat), [])[:k]
            user_recs[u] = items
    eval_strategy(f"SegPop CC top-{k}", user_recs, val_users, gt)

# Strategy B: Recently contacted items in (city, category) — last N days
print("\n--- Strategy B: Recent contacts in user's CC segment ---")
for window in [7, 14, 30, 60]:
    cutoff = split_date - timedelta(days=window)
    recent = train_contacts.filter(pl.col("last_date") > cutoff)
    recent_cc = (
        recent.filter(pl.col("city_name").is_not_null() & pl.col("category").is_not_null())
        .group_by(["city_name", "category", "item_id"])
        .agg(pl.len().alias("c"))
        .sort(["city_name", "category", "c"], descending=[False, False, True])
    )
    recent_map = defaultdict(list)
    for r in recent_cc.iter_rows(named=True):
        key = (r["city_name"], r["category"])
        if len(recent_map[key]) < 500:
            recent_map[key].append(r["item_id"])
    
    user_recs = {}
    for u in val_users:
        if u in prefs:
            city, cat, dist = prefs[u]
            user_recs[u] = recent_map.get((city, cat), [])[:10]
    eval_strategy(f"Recent CC ({window}d), top-10", user_recs, val_users, gt)

# Strategy C: Recently posted items in (city, category) from dim_listing
print("\n--- Strategy C: Freshly POSTED items in user's CC segment ---")
for window in [7, 14, 30, 60, 90]:
    cutoff = split_date - timedelta(days=window)
    fresh = df_listing.filter(pl.col("posted_date") > cutoff)
    fresh_cc = (
        fresh.filter(pl.col("city_name").is_not_null() & pl.col("category").is_not_null())
        .sort("posted_date", descending=True)
    )
    fresh_map = defaultdict(list)
    for r in fresh_cc.iter_rows(named=True):
        key = (r["city_name"], r["category"])
        if len(fresh_map[key]) < 500:
            fresh_map[key].append(r["item_id"])
    
    user_recs = {}
    for u in val_users:
        if u in prefs:
            city, cat, dist = prefs[u]
            user_recs[u] = fresh_map.get((city, cat), [])[:10]
    eval_strategy(f"Fresh posted ({window}d), top-10", user_recs, val_users, gt)

# Strategy D: Hybrid — recent contacts + fresh posted
print("\n--- Strategy D: Hybrid (recent contacts + fresh posted) ---")
cutoff_30 = split_date - timedelta(days=30)
recent_30 = train_contacts.filter(pl.col("last_date") > cutoff_30)
recent_cc_30 = (
    recent_30.filter(pl.col("city_name").is_not_null() & pl.col("category").is_not_null())
    .group_by(["city_name", "category", "item_id"])
    .agg(pl.len().alias("c"))
    .sort(["city_name", "category", "c"], descending=[False, False, True])
)
rmap_30 = defaultdict(list)
for r in recent_cc_30.iter_rows(named=True):
    key = (r["city_name"], r["category"])
    if len(rmap_30[key]) < 500:
        rmap_30[key].append(r["item_id"])

fresh_30 = df_listing.filter(pl.col("posted_date") > cutoff_30).sort("posted_date", descending=True)
fmap_30 = defaultdict(list)
for r in fresh_30.iter_rows(named=True):
    key = (r["city_name"], r["category"])
    if len(fmap_30[key]) < 500:
        fmap_30[key].append(r["item_id"])

user_recs = {}
for u in val_users:
    if u in prefs:
        city, cat, dist = prefs[u]
        seen = set()
        items = []
        # First: recently contacted in segment
        for it in rmap_30.get((city, cat), []):
            if it not in seen:
                items.append(it)
                seen.add(it)
                if len(items) >= 10:
                    break
        # Then: freshly posted in segment
        if len(items) < 10:
            for it in fmap_30.get((city, cat), []):
                if it not in seen:
                    items.append(it)
                    seen.add(it)
                    if len(items) >= 10:
                        break
        user_recs[u] = items
eval_strategy("Hybrid: recent-contact(30d) + fresh-posted(30d)", user_recs, val_users, gt)

# Strategy E: User's own recent contacts + segment popular
print("\n--- Strategy E: User's own history replay + segment ---")
user_history = defaultdict(list)
hist_df = train_contacts.sort("last_date", descending=True)
for r in hist_df.filter(pl.col("user_id").is_in(val_users)).iter_rows(named=True):
    if len(user_history[r["user_id"]]) < 50:
        user_history[r["user_id"]].append(r["item_id"])

user_recs = {}
for u in val_users:
    seen = set()
    items = []
    # Own history
    for it in user_history.get(u, []):
        if it not in seen:
            items.append(it)
            seen.add(it)
            if len(items) >= 5:
                break
    # Fill with segment popular
    if u in prefs:
        city, cat, dist = prefs[u]
        for it in cc_map.get((city, cat), []):
            if it not in seen:
                items.append(it)
                seen.add(it)
                if len(items) >= 10:
                    break
    user_recs[u] = items
eval_strategy("History replay(5) + SegPop CC(5)", user_recs, val_users, gt)

# Strategy F: District-level granularity
print("\n--- Strategy F: District-level SegPop ---")
ccd_pop = (
    train_contacts
    .join(df_listing.select(["item_id", "district_name"]), on="item_id", how="left")
    .filter(
        pl.col("city_name").is_not_null()
        & pl.col("category").is_not_null()
        & pl.col("district_name").is_not_null()
    )
    .group_by(["city_name", "category", "district_name", "item_id"])
    .agg(pl.len().alias("c"))
    .sort(["city_name", "category", "district_name", "c"], descending=[False, False, False, True])
)
ccd_map = defaultdict(list)
for r in ccd_pop.iter_rows(named=True):
    key = (r["city_name"], r["category"], r["district_name"])
    if len(ccd_map[key]) < 200:
        ccd_map[key].append(r["item_id"])

user_recs = {}
for u in val_users:
    if u in prefs:
        city, cat, dist = prefs[u]
        seen = set()
        items = []
        if dist:
            for it in ccd_map.get((city, cat, dist), []):
                if it not in seen:
                    items.append(it); seen.add(it)
                    if len(items) >= 10: break
        if len(items) < 10:
            for it in cc_map.get((city, cat), []):
                if it not in seen:
                    items.append(it); seen.add(it)
                    if len(items) >= 10: break
        user_recs[u] = items
eval_strategy("CCD cascade (dist→cc)", user_recs, val_users, gt)

# Strategy G: CEILING — what's the max recall if we had perfect coverage?
print("\n--- Strategy G: Theoretical ceilings ---")
# Ceiling 1: items in same (city, cat) as user pref
user_recs = {}
for u in val_users:
    user_gt = gt.get(u, set())
    if u in prefs:
        city, cat, dist = prefs[u]
        # All GT items that match user's city+cat
        matching = gt_listing.filter(
            (pl.col("city_name") == city) & (pl.col("category") == cat)
        )["item_id"].to_list()
        user_recs[u] = matching[:10]
eval_strategy("CEILING: GT items matching user CC", user_recs, val_users, gt)

# Ceiling 2: items in same city
user_recs = {}
for u in val_users:
    user_gt = gt.get(u, set())
    if u in prefs:
        city, cat, dist = prefs[u]
        matching = gt_listing.filter(pl.col("city_name") == city)["item_id"].to_list()
        user_recs[u] = matching[:10]
eval_strategy("CEILING: GT items matching user city", user_recs, val_users, gt)

print("\n" + "=" * 70)
print("DONE. Use these numbers to decide the best strategy.")
print("=" * 70)
