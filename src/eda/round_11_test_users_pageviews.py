"""
DIAGNOSTIC 2: Analyze pageview data for test users
===================================================
The ceiling from contacts-based prefs is only 0.085. Top 1 has 0.3.
Hypothesis: they use RAW PAGEVIEW data to find what users recently browsed.
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

contacts = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
date_range = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))
max_date = date_range["max_date"][0]
split_date = max_date - timedelta(days=c.validation_days)

train_contacts = contacts.filter(pl.col("last_date") <= split_date)
val_contacts = contacts.filter(pl.col("last_date") > split_date)

gt = defaultdict(set)
for r in val_contacts.iter_rows(named=True):
    gt[r["user_id"]].add(r["item_id"])
all_val_users = list(gt.keys())
rng = np.random.default_rng(42)
val_users = rng.choice(all_val_users, size=min(3000, len(all_val_users)), replace=False).tolist()
val_set = set(val_users)

print(f"Val users: {len(val_users)}")
print(f"Split date: {split_date}")

# ── 1. Load raw pageview events for val users ─────────────────
print("\n[1] Loading pageview data for val users...")
events = pl.scan_parquet(os.path.join(TRAIN_PATH, "fact_user_events/*.parquet"))

# Get pageviews for val users in last 7 days before split
pv_cutoff = split_date - timedelta(days=7)
pageviews = (
    events
    .filter(
        (pl.col("user_id").is_in(list(val_set)))
        & (pl.col("event_type") == "pageview")
        & (pl.col("event_ts") >= pv_cutoff)
        & (pl.col("event_ts") <= split_date)
    )
    .select(["user_id", "item_id", "event_ts", "city_name", "category"])
    .collect()
)
print(f"  Pageviews (last 7d): {len(pageviews):,} rows")
print(f"  Unique users with pageviews: {pageviews['user_id'].n_unique():,}")

# ── 2. How many GT items were pageviewed? ─────────────────────
print("\n[2] GT overlap with pageviews...")
gt_items = set()
for u in val_users:
    gt_items.update(gt.get(u, set()))

pv_items_all = set(pageviews["item_id"].to_list())
print(f"  GT items: {len(gt_items)}")
print(f"  Pageviewed items: {len(pv_items_all)}")
print(f"  Overlap: {len(gt_items & pv_items_all)} ({100*len(gt_items & pv_items_all)/max(1,len(gt_items)):.1f}%)")

# Per-user: how many GT items did the user pageview?
user_gt_in_pv = 0
user_count = 0
for u in val_users:
    user_gt = gt.get(u, set())
    if not user_gt:
        continue
    user_pv = set(pageviews.filter(pl.col("user_id") == u)["item_id"].to_list())
    user_gt_in_pv += len(user_gt & user_pv)
    user_count += len(user_gt)
print(f"  Per-user GT items that were pageviewed: {user_gt_in_pv}/{user_count} ({100*user_gt_in_pv/max(1,user_count):.1f}%)")

# ── 3. Strategy: Recommend user's recent pageviews ────────────
print("\n[3] Strategy: Recommend user's most recently viewed items...")
user_recent_pv = defaultdict(list)
pv_sorted = pageviews.sort("event_ts", descending=True)
for r in pv_sorted.iter_rows(named=True):
    uid = r["user_id"]
    iid = r["item_id"]
    if len(user_recent_pv[uid]) < 50 and iid not in set(user_recent_pv[uid]):
        user_recent_pv[uid].append(iid)

def eval_strategy(name, user_recs, val_users, gt):
    hits = 0
    for u in val_users:
        user_gt = gt.get(u, set())
        recs = user_recs.get(u, [])[:10]
        hits += len(set(recs) & user_gt)
    recall = hits / len(val_users)
    print(f"  {name:50s} | Recall@10: {recall:.4f}")
    return recall

user_recs = {u: user_recent_pv.get(u, [])[:10] for u in val_users}
eval_strategy("User's top-10 recent pageviews (7d)", user_recs, val_users, gt)

# ── 4. Strategy: Pageview-derived prefs + recent contacts ─────
print("\n[4] Strategy: Pageview-derived (city,cat) prefs...")
# Derive preferences from pageviews instead of contacts
pv_prefs = (
    pageviews
    .group_by("user_id")
    .agg([
        pl.col("city_name").drop_nulls().mode().first().alias("pv_city"),
        pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("pv_cat"),
    ])
)
pv_prefs_dict = {}
for r in pv_prefs.iter_rows(named=True):
    pv_prefs_dict[r["user_id"]] = (r["pv_city"], r["pv_cat"])

# Compare with contact-derived prefs
contact_prefs = {}
prefs_df = (
    train_contacts.filter(pl.col("user_id").is_in(val_users))
    .group_by("user_id")
    .agg([
        pl.col("city_name").drop_nulls().mode().first().alias("c_city"),
        pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("c_cat"),
    ])
)
for r in prefs_df.iter_rows(named=True):
    contact_prefs[r["user_id"]] = (r["c_city"], r["c_cat"])

match_count = 0
total = 0
for u in val_users:
    if u in pv_prefs_dict and u in contact_prefs:
        total += 1
        if pv_prefs_dict[u] == contact_prefs[u]:
            match_count += 1
print(f"  PV prefs == Contact prefs: {match_count}/{total} ({100*match_count/max(1,total):.1f}%)")

# Use PV prefs for segment pop
recent_cc_7 = (
    train_contacts.filter(pl.col("last_date") > split_date - timedelta(days=7))
    .filter(pl.col("city_name").is_not_null() & pl.col("category").is_not_null())
    .group_by(["city_name", "category", "item_id"])
    .agg(pl.len().alias("c"))
    .sort(["city_name", "category", "c"], descending=[False, False, True])
)
cc_recent_map = defaultdict(list)
for r in recent_cc_7.iter_rows(named=True):
    key = (r["city_name"], r["category"])
    if len(cc_recent_map[key]) < 500:
        cc_recent_map[key].append(r["item_id"])

# Use PV-derived prefs
user_recs = {}
for u in val_users:
    city, cat = pv_prefs_dict.get(u, (None, None))
    if city is None or cat is None:
        city2, cat2 = contact_prefs.get(u, (None, None))
        city = city or city2
        cat = cat or cat2
    items = cc_recent_map.get((city, cat), [])[:10]
    user_recs[u] = items
eval_strategy("PV-prefs + Recent CC (7d), top-10", user_recs, val_users, gt)

# ── 5. Strategy: User's pageviews + co-contacted items ────────
print("\n[5] Strategy: Item-item co-contact (users who contacted X also contacted Y)...")
# Build co-contact graph from recent contacts
recent_contacts = train_contacts.filter(pl.col("last_date") > split_date - timedelta(days=30))
user_items = defaultdict(set)
for r in recent_contacts.iter_rows(named=True):
    user_items[r["user_id"]].add(r["item_id"])

# For each item, find co-contacted items
item_cocontact = defaultdict(lambda: defaultdict(int))
for uid, items in user_items.items():
    if len(items) > 100: continue  # skip bots
    items_list = list(items)
    for i in range(len(items_list)):
        for j in range(i+1, min(len(items_list), i+20)):
            item_cocontact[items_list[i]][items_list[j]] += 1
            item_cocontact[items_list[j]][items_list[i]] += 1
print(f"  Co-contact graph: {len(item_cocontact)} items")

# For each val user, find their recent contacts and expand via co-contact
user_recent_contacts = defaultdict(list)
for r in train_contacts.filter(pl.col("user_id").is_in(val_users)).sort("last_date", descending=True).iter_rows(named=True):
    if len(user_recent_contacts[r["user_id"]]) < 20:
        user_recent_contacts[r["user_id"]].append(r["item_id"])

user_recs = {}
for u in val_users:
    seed_items = user_recent_contacts.get(u, [])[:10]
    expanded = defaultdict(float)
    for seed in seed_items:
        for co_item, cnt in sorted(item_cocontact.get(seed, {}).items(), key=lambda x: -x[1])[:20]:
            if co_item not in set(seed_items):
                expanded[co_item] += cnt
    top_expanded = sorted(expanded.items(), key=lambda x: -x[1])[:10]
    user_recs[u] = [it for it, _ in top_expanded]
eval_strategy("Co-contact expansion from user history", user_recs, val_users, gt)

# ── 6. MEGA STRATEGY: Combine everything ─────────────────────
print("\n[6] MEGA STRATEGY: PV items + Recent CC + Co-contact...")
user_recs = {}
for u in val_users:
    seen = set()
    items = []
    
    # Source 1: Recent pageviews (direct signal)
    for it in user_recent_pv.get(u, [])[:3]:
        if it not in seen:
            items.append(it); seen.add(it)
    
    # Source 2: Co-contact expansion
    for seed in user_recent_contacts.get(u, [])[:5]:
        for co_item, cnt in sorted(item_cocontact.get(seed, {}).items(), key=lambda x: -x[1])[:3]:
            if co_item not in seen and len(items) < 7:
                items.append(co_item); seen.add(co_item)
    
    # Source 3: Recent CC segment popular
    city, cat = pv_prefs_dict.get(u, contact_prefs.get(u, (None, None)))
    if city and cat:
        for it in cc_recent_map.get((city, cat), []):
            if it not in seen and len(items) < 10:
                items.append(it); seen.add(it)
    
    user_recs[u] = items
eval_strategy("MEGA: PV(3) + CoContact(4) + RecentCC(3)", user_recs, val_users, gt)

print("\nDone.")
