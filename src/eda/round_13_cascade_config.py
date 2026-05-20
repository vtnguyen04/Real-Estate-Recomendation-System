"""
DIAGNOSTIC 4: Find optimal cascade config and identify improvement vectors.
"""
import polars as pl
import numpy as np
import os
from datetime import timedelta
from collections import defaultdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PipelineConfig
from src.models.candidates.pageview_replay import PageviewReplayRecommender
from src.models.candidates.cocontact import CoContactRecommender
from src.models.candidates.segment_popularity import SegmentPopularityRecommender
from src.models.ensemble.cascade_generator import CascadeCandidateGenerator
from src.evaluation.metrics import recall_at_k, ndcg_at_k

c = PipelineConfig()
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
val_users = rng.choice(all_val_users, size=min(5000, len(all_val_users)), replace=False).tolist()
val_set = set(val_users)

# Prefs
prefs_dict = {}
prefs_df = (
    train_contacts.filter(pl.col("user_id").is_in(val_users))
    .group_by("user_id")
    .agg([
        pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
        pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("pref_cat"),
    ])
)
for r in prefs_df.iter_rows(named=True):
    prefs_dict[r["user_id"]] = (r.get("pref_city"), r.get("pref_cat"))
cold_prefs = pl.read_parquet(os.path.join(CACHE_DIR, "cold_user_prefs.parquet"))
for r in cold_prefs.filter(pl.col("user_id").is_in(val_users)).iter_rows(named=True):
    if r["user_id"] not in prefs_dict:
        prefs_dict[r["user_id"]] = (r.get("pref_city"), r.get("pref_cat"))

def eval_recall(user_recs, val_users, gt):
    recalls = []
    for u in val_users:
        actual = gt.get(u, set())
        if not actual:
            continue
        preds = user_recs.get(u, [])[:10]
        recalls.append(recall_at_k(preds, actual, k=10))
    return np.mean(recalls)

print("=" * 70)
print("DIAGNOSTIC 4: Optimize cascade hyperparameters")
print("=" * 70)

# ── 1. Build all sources ──────────────────────────────────────
print("\n[1] Building sources...")
events_path = os.path.join(c.data.train_path, "fact_user_events/*.parquet")

# Source A: PV replay (try different windows)
pv_14 = PageviewReplayRecommender(window_days=14, max_items_per_user=50)
pv_14.fit(events_path, user_ids=val_set, cutoff_date=split_date)

pv_30 = PageviewReplayRecommender(window_days=30, max_items_per_user=50)
pv_30.fit(events_path, user_ids=val_set, cutoff_date=split_date)

# Source B: CoContact
cocontact = CoContactRecommender(window_days=30)
cocontact.fit(train_contacts, cutoff_date=split_date)

# Source C: Session co-view graph (from pageviews)
print("  Building session co-view graph...")
pv_events = (
    pl.scan_parquet(events_path)
    .filter(
        (pl.col("event_ts") >= split_date - timedelta(days=30))
        & (pl.col("event_ts") <= split_date)
        & (pl.col("event_type") == "pageview")
    )
    .select(["user_id", "item_id"])
    .collect()
)
# Co-view: items viewed by same user in same 30d window
coview_user_items = defaultdict(set)
for r in pv_events.iter_rows(named=True):
    coview_user_items[r["user_id"]].add(r["item_id"])
coview_graph = defaultdict(lambda: defaultdict(int))
n_used = 0
for uid, items in coview_user_items.items():
    if len(items) > 200: continue  # skip bots
    n_used += 1
    items_list = list(items)
    for i in range(len(items_list)):
        for j in range(i+1, min(len(items_list), i+30)):
            coview_graph[items_list[i]][items_list[j]] += 1
            coview_graph[items_list[j]][items_list[i]] += 1
print(f"  Co-view graph: {len(coview_graph):,} items from {n_used:,} users")

# Source D: Recent CC
recent_cc_7 = CascadeCandidateGenerator.build_recent_cc(
    train_contacts, cutoff_date=split_date, window_days=7,
)
recent_cc_14 = CascadeCandidateGenerator.build_recent_cc(
    train_contacts, cutoff_date=split_date, window_days=14,
)

# Source E: SegPop
segpop = SegmentPopularityRecommender().load("outputs/models/segpop.pkl")

# User histories
user_histories = CascadeCandidateGenerator.build_user_histories(
    train_contacts, user_ids=val_set, max_items=20,
)

# User pageview histories (for co-view expansion)
user_pv_histories = defaultdict(list)
pv_sorted = pv_events.group_by(["user_id", "item_id"]).agg(pl.len().alias("c")).sort("c", descending=True)
for r in pv_sorted.iter_rows(named=True):
    if len(user_pv_histories[r["user_id"]]) < 30:
        user_pv_histories[r["user_id"]].append(r["item_id"])

# ALS recommendations
from src.models.candidates.light_als import LightALSRecommender
als = LightALSRecommender()
als.load("outputs/models/als")
if als._matrix is None:
    als_contacts = pl.read_parquet(os.path.join(CACHE_DIR, "als_contact_pairs.parquet"))
    als.rebuild_matrix(als_contacts)
als_recs_cache = als.recommend_batch(val_users, n=20, return_scores=False)
print(f"  ALS recs for {len(als_recs_cache):,} users")

# ── 2. Per-source analysis ────────────────────────────────────
print("\n[2] Per-source recall analysis...")

# A1: PV only (14d)
recs = pv_14.recommend_batch(val_users, k=10)
print(f"  PV(14d) only:  Recall={eval_recall(recs, val_users, gt):.4f}, covers {sum(1 for u in val_users if recs.get(u)):,}")

# A2: PV only (30d)
recs = pv_30.recommend_batch(val_users, k=10)
print(f"  PV(30d) only:  Recall={eval_recall(recs, val_users, gt):.4f}, covers {sum(1 for u in val_users if recs.get(u)):,}")

# B: CoContact only
recs = {}
for u in val_users:
    hist = user_histories.get(u, [])
    if hist:
        recs[u] = cocontact.recommend(hist, k=10, exclude=set(hist))
print(f"  CoContact:     Recall={eval_recall(recs, val_users, gt):.4f}, covers {sum(1 for u in val_users if recs.get(u)):,}")

# C: Co-view expansion
recs = {}
for u in val_users:
    pv_hist = user_pv_histories.get(u, [])
    if pv_hist:
        expanded = defaultdict(float)
        for seed in pv_hist[:10]:
            for co_item, cnt in sorted(coview_graph.get(seed, {}).items(), key=lambda x: -x[1])[:10]:
                if co_item not in set(pv_hist):
                    expanded[co_item] += cnt
        recs[u] = [it for it, _ in sorted(expanded.items(), key=lambda x: -x[1])[:10]]
print(f"  Co-view:       Recall={eval_recall(recs, val_users, gt):.4f}, covers {sum(1 for u in val_users if recs.get(u)):,}")

# D: ALS only
recs = {u: items[:10] for u, items in als_recs_cache.items()}
print(f"  ALS:           Recall={eval_recall(recs, val_users, gt):.4f}, covers {sum(1 for u in val_users if recs.get(u)):,}")

# E: Recent CC (7d)
recs = {}
for u in val_users:
    city, cat = prefs_dict.get(u, (None, None))
    if city and cat:
        recs[u] = recent_cc_7.get((city, cat), [])[:10]
print(f"  RecentCC(7d):  Recall={eval_recall(recs, val_users, gt):.4f}, covers {sum(1 for u in val_users if recs.get(u)):,}")

# ── 3. Test cascade combinations ─────────────────────────────
print("\n[3] Testing cascade combinations...")

def build_cascade_recs(val_users, sources_config):
    """Generic cascade builder.
    sources_config: list of (name, recommend_fn) in priority order.
    """
    recs = {}
    for u in val_users:
        seen = set()
        items = []
        for name, rec_fn in sources_config:
            for it in rec_fn(u):
                if it not in seen and len(items) < 10:
                    items.append(it); seen.add(it)
            if len(items) >= 10:
                break
        recs[u] = items
    return recs

def pv14_fn(u): return pv_14.recommend(u, k=10)
def pv30_fn(u): return pv_30.recommend(u, k=10)
def cocontact_fn(u):
    hist = user_histories.get(u, [])
    return cocontact.recommend(hist, k=10, exclude=set(hist)) if hist else []
def coview_fn(u):
    pv_hist = user_pv_histories.get(u, [])
    if not pv_hist: return []
    expanded = defaultdict(float)
    for seed in pv_hist[:10]:
        for co_item, cnt in sorted(coview_graph.get(seed, {}).items(), key=lambda x: -x[1])[:10]:
            if co_item not in set(pv_hist):
                expanded[co_item] += cnt
    return [it for it, _ in sorted(expanded.items(), key=lambda x: -x[1])[:10]]
def als_fn(u): return als_recs_cache.get(u, [])[:10]
def rcc7_fn(u):
    city, cat = prefs_dict.get(u, (None, None))
    return recent_cc_7.get((city, cat), [])[:10] if city and cat else []
def rcc14_fn(u):
    city, cat = prefs_dict.get(u, (None, None))
    return recent_cc_14.get((city, cat), [])[:10] if city and cat else []
def segpop_fn(u):
    city, cat = prefs_dict.get(u, (None, None))
    return segpop.get_segment_items(pref_city=city, pref_cat=cat, k=10)

configs = [
    ("PV14 → CoContact → RCC7 → SegPop", [("pv14", pv14_fn), ("cc", cocontact_fn), ("rcc7", rcc7_fn), ("seg", segpop_fn)]),
    ("PV30 → CoContact → RCC7 → SegPop", [("pv30", pv30_fn), ("cc", cocontact_fn), ("rcc7", rcc7_fn), ("seg", segpop_fn)]),
    ("PV14 → CoView → CoContact → RCC7 → SegPop", [("pv14", pv14_fn), ("cov", coview_fn), ("cc", cocontact_fn), ("rcc7", rcc7_fn), ("seg", segpop_fn)]),
    ("PV14 → CoContact → ALS → RCC7 → SegPop", [("pv14", pv14_fn), ("cc", cocontact_fn), ("als", als_fn), ("rcc7", rcc7_fn), ("seg", segpop_fn)]),
    ("PV14 → CoView → CoContact → ALS → RCC7 → SegPop", [("pv14", pv14_fn), ("cov", coview_fn), ("cc", cocontact_fn), ("als", als_fn), ("rcc7", rcc7_fn), ("seg", segpop_fn)]),
    ("PV30 → CoView → CoContact → ALS → RCC7 → SegPop", [("pv30", pv30_fn), ("cov", coview_fn), ("cc", cocontact_fn), ("als", als_fn), ("rcc7", rcc7_fn), ("seg", segpop_fn)]),
    ("PV30 → CoView → CoContact → ALS → RCC14 → SegPop", [("pv30", pv30_fn), ("cov", coview_fn), ("cc", cocontact_fn), ("als", als_fn), ("rcc14", rcc14_fn), ("seg", segpop_fn)]),
    ("PV14 → CoContact → CoView → ALS → RCC7 → SegPop", [("pv14", pv14_fn), ("cc", cocontact_fn), ("cov", coview_fn), ("als", als_fn), ("rcc7", rcc7_fn), ("seg", segpop_fn)]),
]

for name, cfg in configs:
    recs = build_cascade_recs(val_users, cfg)
    r = eval_recall(recs, val_users, gt)
    print(f"  {name:60s} | Recall@10: {r:.4f}")

# ── 4. Per-source contribution analysis ──────────────────────
print("\n[4] Source contribution in best cascade...")
best_cfg = [("pv14", pv14_fn), ("cov", coview_fn), ("cc", cocontact_fn), ("als", als_fn), ("rcc7", rcc7_fn), ("seg", segpop_fn)]
source_hits = defaultdict(int)
source_count = defaultdict(int)
for u in val_users:
    actual = gt.get(u, set())
    if not actual: continue
    seen = set()
    items = []
    for name, rec_fn in best_cfg:
        for it in rec_fn(u):
            if it not in seen and len(items) < 10:
                items.append(it)
                seen.add(it)
                if it in actual:
                    source_hits[name] += 1
                source_count[name] += 1
        if len(items) >= 10: break
for name, _ in best_cfg:
    print(f"  {name:8s}: {source_count[name]:6,} items placed, {source_hits[name]:4} hits")

print("\nDone.")
