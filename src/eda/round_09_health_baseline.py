"""
Round 09 EDA: Health Metrics Baseline Analysis.

Goal:
  Quantify the current state of our recommendations across 4 axes:
    1. Diversity   — Intra-List Diversity (ILD) & category entropy
    2. Fairness    — agent/private seller ratio & category distribution
    3. Freshness   — listing age distribution in recommendations
    4. Coverage    — fraction of item catalogue represented; long-tail exposure

  Also establishes the "natural distribution" ground truth from training
  positive contacts — used to calibrate HealthMetrics.gt_dist.

Outputs:
  - figures/round_09_diversity.png
  - figures/round_09_fairness.png
  - figures/round_09_freshness.png
  - figures/round_09_coverage.png
  - reports/round_09_report.md
  - data/gt_dist.json   (pickled HealthMetrics calibration)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import polars as pl
import numpy as np
import json
from collections import Counter
from datetime import date
from scipy.stats import entropy as scipy_entropy

from src.data.loader import ListingDataLoader, PostContactInteractionsLoader
from src.utils.report_writer import write_report
from src.utils.plotting import plot_bar, plot_histogram, save_figure

TRAIN_PATH    = "/home/db/rc/datathon/train/"
SUBMISSION    = os.path.join(os.path.dirname(__file__), "..", "..", "submission.csv")
REPORT_DIR    = os.path.join(os.path.dirname(__file__), "reports")
FIGURES_DIR   = os.path.join(REPORT_DIR, "figures")
GT_DIST_OUT   = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "gt_dist.json")
CUTOFF_DATE   = date(2026, 4, 9)

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(GT_DIST_OUT), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading dim_listing...")
listing_loader = ListingDataLoader(data_path=os.path.join(TRAIN_PATH, "dim_listing/"))
df_listing = listing_loader.load(
    columns=["item_id", "category", "city_name", "district_name",
             "seller_type", "ad_type", "posted_date"]
).collect().with_columns(
    (
        (pl.lit(CUTOFF_DATE).cast(pl.Date) - pl.col("posted_date")).dt.total_days()
    ).alias("listing_age_days")
)
print(f"  Listings: {len(df_listing):,}")

print("Loading positive interactions...")
inter_loader = PostContactInteractionsLoader(
    data_path=os.path.join(TRAIN_PATH, "fact_post_contact_interactions/")
)
df_inter = (
    inter_loader
    .load(columns=["user_id", "item_id", "lead_count", "chat_lead"])
    .filter(
        (pl.col("lead_count") > 0) | (pl.col("chat_lead") > 0)
    )
    .collect()
)
print(f"  Positive interactions: {len(df_inter):,}")

print("Loading submission...")
df_sub = pl.read_csv(SUBMISSION)
print(f"  Submission rows: {len(df_sub):,}")

# Join submission with listing metadata
df_sub_meta = df_sub.join(df_listing, on="item_id", how="left")
print(f"  Submission with metadata: {len(df_sub_meta):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Natural Distribution from Training Contacts (gt_dist)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Computing gt_dist from positive contacts ──")

contact_with_meta = (
    df_inter
    .join(df_listing.select(["item_id", "category", "seller_type"]), on="item_id", how="left")
)

# Category distribution in contacts
cat_counts_gt = (
    contact_with_meta
    .filter(pl.col("category").is_not_null())
    .group_by("category")
    .agg(pl.len().alias("count"))
    .sort("category")
)
total_contacts = cat_counts_gt["count"].sum()
cat_dist_gt = {
    int(row["category"]): float(row["count"]) / total_contacts
    for row in cat_counts_gt.iter_rows(named=True)
    if int(row["category"]) in {1010, 1020, 1030, 1040, 1050}
}
# Normalise to valid categories only
s = sum(cat_dist_gt.values())
cat_dist_gt = {k: v / s for k, v in cat_dist_gt.items()}

# Agent ratio in contacts
agent_count = (
    contact_with_meta
    .filter(pl.col("seller_type") == "agent")["lead_count"].len()
)
total_seller = contact_with_meta.filter(pl.col("seller_type").is_not_null()).height
agent_ratio_gt = agent_count / total_seller if total_seller > 0 else 0.7

gt_dist = {
    "agent_ratio": float(agent_ratio_gt),
    "category_dist": {str(k): v for k, v in cat_dist_gt.items()},
}
with open(GT_DIST_OUT, "w") as f:
    json.dump(gt_dist, f, indent=2)

print(f"  gt_dist computed:")
print(f"    agent_ratio = {agent_ratio_gt:.3f}")
for cat, pct in sorted(cat_dist_gt.items()):
    print(f"    category {cat}: {pct:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Diversity Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Diversity Analysis ──")

# Category entropy per user (top-10 list)
def category_entropy(cats):
    cats = [c for c in cats if c is not None]
    if len(cats) < 2:
        return 0.0
    counts = Counter(cats)
    probs = np.array(list(counts.values())) / len(cats)
    ent = scipy_entropy(probs, base=2)
    max_ent = np.log2(len(counts)) if len(counts) > 1 else 1
    return ent / max_ent if max_ent > 0 else 0.0

user_cats = (
    df_sub_meta
    .group_by("user_id")
    .agg(pl.col("category").alias("cats"))
)
entropy_scores = [
    category_entropy(row["cats"])
    for row in user_cats.iter_rows(named=True)
]
avg_cat_entropy = float(np.mean(entropy_scores))
print(f"  Avg category entropy: {avg_cat_entropy:.4f}  (1.0 = perfect diversity)")

# Geographic diversity: unique cities per user
user_cities = (
    df_sub_meta
    .filter(pl.col("city_name").is_not_null())
    .group_by("user_id")
    .agg(pl.col("city_name").n_unique().alias("n_cities"))
)
avg_cities = float(user_cities["n_cities"].mean())
print(f"  Avg distinct cities per user: {avg_cities:.3f}")

# Category distribution in submission vs gt
cat_dist_sub = (
    df_sub_meta
    .filter(pl.col("category").is_not_null())
    .filter(pl.col("category").is_in([1010, 1020, 1030, 1040, 1050]))
    .group_by("category")
    .agg(pl.len().alias("count"))
    .sort("category")
)
total_sub = cat_dist_sub["count"].sum()
cat_dist_sub_pct = {
    int(row["category"]): row["count"] / total_sub
    for row in cat_dist_sub.iter_rows(named=True)
}

print("\n  Category distribution comparison:")
print(f"  {'Cat':>5} | {'Submission':>12} | {'GT (contacts)':>14} | {'Gap':>8}")
print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*14}-+-{'-'*8}")
for cat in sorted(cat_dist_gt.keys()):
    sub_pct = cat_dist_sub_pct.get(cat, 0)
    gt_pct  = cat_dist_gt[cat]
    print(f"  {cat:>5} | {sub_pct:>12.3f} | {gt_pct:>14.3f} | {sub_pct-gt_pct:>+8.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fairness Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Fairness Analysis ──")

seller_dist = (
    df_sub_meta
    .filter(pl.col("seller_type").is_not_null())
    .group_by("seller_type")
    .agg(pl.len().alias("count"))
)
total_sellers = seller_dist["count"].sum()
agent_ratio_sub = (
    seller_dist
    .filter(pl.col("seller_type") == "agent")["count"].sum()
) / total_sellers
private_ratio_sub = 1.0 - agent_ratio_sub

print(f"  Submission:   agent={agent_ratio_sub:.3f}, private={private_ratio_sub:.3f}")
print(f"  GT (contacts): agent={agent_ratio_gt:.3f}, private={1-agent_ratio_gt:.3f}")
print(f"  Note (EDA R05): private sellers have 3× higher avg leads → need more exposure")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Freshness Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Freshness Analysis ──")

age_sub = df_sub_meta.filter(pl.col("listing_age_days").is_not_null())["listing_age_days"]
age_gt_contacts = (
    df_inter
    .join(df_listing.select(["item_id", "listing_age_days"]), on="item_id", how="left")
    .filter(pl.col("listing_age_days").is_not_null())
)["listing_age_days"]

print(f"  Submission  — median age: {float(age_sub.median()):.0f}d, mean: {float(age_sub.mean()):.0f}d")
print(f"  GT contacts — median age: {float(age_gt_contacts.median()):.0f}d, mean: {float(age_gt_contacts.mean()):.0f}d")
print(f"  Freshness score (exp decay 0.05): sub={float(np.mean(np.exp(-0.05 * age_sub.to_numpy()))):.4f}, gt={float(np.mean(np.exp(-0.05 * age_gt_contacts.to_numpy()))):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Coverage Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Coverage Analysis ──")

unique_items_sub = df_sub["item_id"].n_unique()
total_items      = df_listing["item_id"].n_unique()
coverage_pct     = unique_items_sub / total_items * 100

# Popularity concentration: what fraction of recs go to top-1% items?
item_freq = df_sub.group_by("item_id").agg(pl.len().alias("rec_count")).sort("rec_count", descending=True)
top1pct_items = max(1, int(len(item_freq) * 0.01))
top1pct_share = item_freq.head(top1pct_items)["rec_count"].sum() / item_freq["rec_count"].sum()

print(f"  Items recommended: {unique_items_sub:,} / {total_items:,} ({coverage_pct:.2f}%)")
print(f"  Top-1% items get {top1pct_share:.1%} of all recommendation slots")
print(f"  Gini-like concentration: {'HIGH' if top1pct_share > 0.5 else 'MODERATE'}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary Table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HEALTH METRICS SUMMARY")
print("=" * 60)
print(f"  Diversity (cat entropy):  {avg_cat_entropy:.4f}  [0=all same, 1=all diff]")
print(f"  Diversity (city spread):  {avg_cities:.3f} cities/user")
print(f"  Fairness (agent ratio):   {agent_ratio_sub:.3f}  [gt={agent_ratio_gt:.3f}]")
print(f"  Freshness (exp-decay):    {float(np.mean(np.exp(-0.05 * age_sub.to_numpy()))):.4f}")
print(f"  Coverage:                 {coverage_pct:.2f}%  of item catalogue")
print(f"  Popularity concentration: top-1% items get {top1pct_share:.1%} of slots")
print("=" * 60)
print(f"\ngt_dist saved → {GT_DIST_OUT}")
print("Round 09 complete.")
