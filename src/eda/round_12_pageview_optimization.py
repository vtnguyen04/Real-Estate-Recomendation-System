"""
DIAGNOSTIC 3: Optimize pageview strategy to maximize Recall
============================================================
Pageview replay = 0.17. Let's push it further.
"""
import polars as pl
import numpy as np
import os
from datetime import timedelta
from collections import defaultdict
from config.settings import PipelineConfig

c = PipelineConfig()
TRAIN_PATH = c.data.train_path
CACHE_DIR = ".cache"

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

GT_EVENTS = ['view_phone', 'contact_chat', 'other_interaction', 'contact_zalo', 'contact_sms']

def eval_strategy(name, user_recs, val_users, gt):
    hits = 0
    for u in val_users:
        user_gt = gt.get(u, set())
        recs = user_recs.get(u, [])[:10]
        hits += len(set(recs) & user_gt)
    recall = hits / len(val_users)
    print(f"  {name:55s} | Recall@10: {recall:.4f}")
    return recall

print("=" * 70)
print("DIAGNOSTIC 3: Push pageview strategy to the limit")
print("=" * 70)

# ── 1. Test different pageview windows ────────────────────────
print("\n[1] Pageview window analysis...")
events = pl.scan_parquet(os.path.join(TRAIN_PATH, "fact_user_events/*.parquet"))

for window in [3, 7, 14, 30]:
    pv_cutoff = split_date - timedelta(days=window)
    pageviews = (
        events
        .filter(
            (pl.col("user_id").is_in(list(val_set)))
            & (pl.col("event_type") == "pageview")
            & (pl.col("event_ts") >= pv_cutoff)
            & (pl.col("event_ts") <= split_date)
        )
        .select(["user_id", "item_id", "event_ts"])
        .collect()
    )
    
    # Count pageviews per (user, item) and sort by count then recency
    pv_ranked = (
        pageviews
        .group_by(["user_id", "item_id"])
        .agg([
            pl.len().alias("pv_count"),
            pl.col("event_ts").max().alias("last_pv"),
        ])
        .sort(["user_id", "pv_count", "last_pv"], descending=[False, True, True])
    )
    
    user_pv = defaultdict(list)
    for r in pv_ranked.iter_rows(named=True):
        if len(user_pv[r["user_id"]]) < 10:
            user_pv[r["user_id"]].append(r["item_id"])
    
    user_recs = {u: user_pv.get(u, []) for u in val_users}
    users_with_recs = sum(1 for u in val_users if len(user_recs.get(u, [])) > 0)
    eval_strategy(f"PV replay ({window}d, by count+recency) [{users_with_recs}/{len(val_users)}]", user_recs, val_users, gt)

# ── 2. Test: PV with GT events included (contact replay) ─────
print("\n[2] Include contact events in replay...")
for window in [7, 14, 30]:
    pv_cutoff = split_date - timedelta(days=window)
    all_events = (
        events
        .filter(
            (pl.col("user_id").is_in(list(val_set)))
            & (pl.col("event_ts") >= pv_cutoff)
            & (pl.col("event_ts") <= split_date)
        )
        .select(["user_id", "item_id", "event_ts", "event_type"])
        .collect()
    )
    
    # Weight: GT events = 10x, pageview = 1x
    all_events = all_events.with_columns(
        pl.when(pl.col("event_type").is_in(GT_EVENTS)).then(10).otherwise(1).alias("weight")
    )
    
    ranked = (
        all_events
        .group_by(["user_id", "item_id"])
        .agg([
            pl.col("weight").sum().alias("total_weight"),
            pl.col("event_ts").max().alias("last_event"),
        ])
        .sort(["user_id", "total_weight", "last_event"], descending=[False, True, True])
    )
    
    user_recs_map = defaultdict(list)
    for r in ranked.iter_rows(named=True):
        if len(user_recs_map[r["user_id"]]) < 10:
            user_recs_map[r["user_id"]].append(r["item_id"])
    
    user_recs = {u: user_recs_map.get(u, []) for u in val_users}
    users_with = sum(1 for u in val_users if len(user_recs.get(u, [])) > 0)
    eval_strategy(f"All events weighted ({window}d) [{users_with}/{len(val_users)}]", user_recs, val_users, gt)

# ── 3. Test: PV replay + Recent CC fallback ───────────────────
print("\n[3] PV replay + fallback for users without pageviews...")
# Best PV window
pv_cutoff = split_date - timedelta(days=7)
pageviews_7 = (
    events
    .filter(
        (pl.col("user_id").is_in(list(val_set)))
        & (pl.col("event_ts") >= pv_cutoff)
        & (pl.col("event_ts") <= split_date)
    )
    .select(["user_id", "item_id", "event_ts", "event_type"])
    .collect()
)
pageviews_7 = pageviews_7.with_columns(
    pl.when(pl.col("event_type").is_in(GT_EVENTS)).then(10).otherwise(1).alias("weight")
)
ranked_7 = (
    pageviews_7.group_by(["user_id", "item_id"])
    .agg([pl.col("weight").sum().alias("w"), pl.col("event_ts").max().alias("last")])
    .sort(["user_id", "w", "last"], descending=[False, True, True])
)
user_pv_7 = defaultdict(list)
for r in ranked_7.iter_rows(named=True):
    if len(user_pv_7[r["user_id"]]) < 10:
        user_pv_7[r["user_id"]].append(r["item_id"])

# Recent CC (7d) fallback
recent_cc = (
    train_contacts.filter(pl.col("last_date") > split_date - timedelta(days=7))
    .filter(pl.col("city_name").is_not_null() & pl.col("category").is_not_null())
    .group_by(["city_name", "category", "item_id"])
    .agg(pl.len().alias("c"))
    .sort(["city_name", "category", "c"], descending=[False, False, True])
)
cc_map = defaultdict(list)
for r in recent_cc.iter_rows(named=True):
    key = (r["city_name"], r["category"])
    if len(cc_map[key]) < 200:
        cc_map[key].append(r["item_id"])

# Contact prefs
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
cold_prefs = pl.read_parquet(os.path.join(CACHE_DIR, "cold_user_prefs.parquet"))
for r in cold_prefs.filter(pl.col("user_id").is_in(val_users)).iter_rows(named=True):
    if r["user_id"] not in contact_prefs:
        contact_prefs[r["user_id"]] = (r["pref_city"], r["pref_cat"])

# Co-contact for users with history but no pageviews
recent_ct = train_contacts.filter(pl.col("last_date") > split_date - timedelta(days=30))
user_items = defaultdict(set)
for r in recent_ct.iter_rows(named=True):
    user_items[r["user_id"]].add(r["item_id"])
item_cocontact = defaultdict(lambda: defaultdict(int))
for uid, items in user_items.items():
    if len(items) > 100: continue
    items_list = list(items)
    for i in range(len(items_list)):
        for j in range(i+1, min(len(items_list), i+20)):
            item_cocontact[items_list[i]][items_list[j]] += 1
            item_cocontact[items_list[j]][items_list[i]] += 1

user_recent_hist = defaultdict(list)
for r in train_contacts.filter(pl.col("user_id").is_in(val_users)).sort("last_date", descending=True).iter_rows(named=True):
    if len(user_recent_hist[r["user_id"]]) < 20:
        user_recent_hist[r["user_id"]].append(r["item_id"])

# Build combined recs
user_recs = {}
pv_users = 0
cocontact_users = 0
cc_users = 0
for u in val_users:
    seen = set()
    items = []
    
    # Priority 1: Recent pageviews (most direct signal)
    pv_items = user_pv_7.get(u, [])
    if pv_items:
        pv_users += 1
        for it in pv_items:
            if it not in seen and len(items) < 10:
                items.append(it); seen.add(it)
    
    # Priority 2: Co-contact expansion from history
    if len(items) < 10:
        seed_items = user_recent_hist.get(u, [])[:10]
        expanded = defaultdict(float)
        for seed in seed_items:
            for co_item, cnt in sorted(item_cocontact.get(seed, {}).items(), key=lambda x: -x[1])[:10]:
                if co_item not in seen and co_item not in set(seed_items):
                    expanded[co_item] += cnt
        if expanded:
            cocontact_users += 1
            for co_item, _ in sorted(expanded.items(), key=lambda x: -x[1]):
                if co_item not in seen and len(items) < 10:
                    items.append(co_item); seen.add(co_item)
    
    # Priority 3: Segment popular (last 7d)
    if len(items) < 10:
        city, cat = contact_prefs.get(u, (None, None))
        if city and cat:
            cc_users += 1
            for it in cc_map.get((city, cat), []):
                if it not in seen and len(items) < 10:
                    items.append(it); seen.add(it)
    
    user_recs[u] = items

print(f"  Users with PV: {pv_users}, CoContact: {cocontact_users}, CC: {cc_users}")
eval_strategy("FULL: PV(7d) → CoContact → RecentCC(7d)", user_recs, val_users, gt)

# ── 4. Longer PV window for more coverage ────────────────────
print("\n[4] Longer PV + same fallback...")
for pv_window in [14, 30]:
    pv_cutoff_w = split_date - timedelta(days=pv_window)
    pv_w = (
        events.filter(
            (pl.col("user_id").is_in(list(val_set)))
            & (pl.col("event_ts") >= pv_cutoff_w)
            & (pl.col("event_ts") <= split_date)
        )
        .select(["user_id", "item_id", "event_ts", "event_type"])
        .collect()
    )
    pv_w = pv_w.with_columns(
        pl.when(pl.col("event_type").is_in(GT_EVENTS)).then(10).otherwise(1).alias("weight")
    )
    ranked_w = (
        pv_w.group_by(["user_id", "item_id"])
        .agg([pl.col("weight").sum().alias("w"), pl.col("event_ts").max().alias("last")])
        .sort(["user_id", "w", "last"], descending=[False, True, True])
    )
    user_pv_w = defaultdict(list)
    for r in ranked_w.iter_rows(named=True):
        if len(user_pv_w[r["user_id"]]) < 10:
            user_pv_w[r["user_id"]].append(r["item_id"])
    
    user_recs = {}
    for u in val_users:
        seen = set()
        items = []
        for it in user_pv_w.get(u, []):
            if it not in seen and len(items) < 10:
                items.append(it); seen.add(it)
        if len(items) < 10:
            seed_items = user_recent_hist.get(u, [])[:10]
            expanded = defaultdict(float)
            for seed in seed_items:
                for co_item, cnt in sorted(item_cocontact.get(seed, {}).items(), key=lambda x: -x[1])[:10]:
                    if co_item not in seen and co_item not in set(seed_items):
                        expanded[co_item] += cnt
            for co_item, _ in sorted(expanded.items(), key=lambda x: -x[1]):
                if co_item not in seen and len(items) < 10:
                    items.append(co_item); seen.add(co_item)
        if len(items) < 10:
            city, cat = contact_prefs.get(u, (None, None))
            if city and cat:
                for it in cc_map.get((city, cat), []):
                    if it not in seen and len(items) < 10:
                        items.append(it); seen.add(it)
        user_recs[u] = items
    
    users_with_pv = sum(1 for u in val_users if len(user_pv_w.get(u, [])) > 0)
    eval_strategy(f"FULL w/ PV({pv_window}d) → CoContact → RecentCC [{users_with_pv}]", user_recs, val_users, gt)

print("\n" + "=" * 70)
print("DONE.")
print("=" * 70)
