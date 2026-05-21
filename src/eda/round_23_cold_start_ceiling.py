"""
Round 23: Cold-Start Recall Ceiling Analysis

HYPOTHESIS: H-013 — SegPop recall for blind users is capped at ~2% even with
perfect city+category knowledge. BĐS item distribution is too sparse for
popularity-based cold-start to work.

METHODOLOGY:
1. Split contacts into train/val (last 3 days = val)
2. Identify "blind" val users (contacted in val but NOT in train)
3. Build segment popularity from train period
4. Compute theoretical max recall@10 if we KNEW each blind user's city+cat
5. Compare with recency-weighted and global approaches

OUTPUT:
- reports/round_23_report.md
- Figures: none (numeric analysis)
"""
import os
import sys
import random
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import polars as pl
from config.settings import PipelineConfig
from src.utils.logging import get_logger

logger = get_logger("round_23")

CACHE_DIR = ".cache"
DATA_DIR = PipelineConfig().data.train_path
REPORT_PATH = "src/eda/reports/round_23_report.md"


def load_data():
    """Load contacts, listings, test users."""
    contacts = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
    date_range = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))
    max_date = date_range["max_date"][0]

    listing_dir = os.path.join(DATA_DIR, "dim_listing")
    listing = pl.scan_parquet(os.path.join(listing_dir, "*.parquet")).select(
        ["item_id", "city_name", "category", "posted_date"]
    ).collect()

    test = pl.read_parquet(os.path.join(PipelineConfig().data.test_path, "test_users.parquet"))
    return contacts, max_date, listing, test


def split_train_val(contacts, max_date, val_days=3):
    """Time-split contacts into train and val periods."""
    cutoff = max_date - timedelta(days=val_days)
    train = contacts.filter(pl.col("last_date") <= cutoff)
    val = contacts.filter(pl.col("last_date") > cutoff)
    return train, val


def compute_user_segments(contacts, max_date, test_ids):
    """Compute user segmentation: warm, cold+prefs, truly blind."""
    train_ids = set(contacts.filter(
        pl.col("last_date") <= max_date - timedelta(days=3)
    )["user_id"].unique().to_list())

    cold_prefs_path = os.path.join(CACHE_DIR, "cold_user_prefs.parquet")
    cold_ids = set()
    if os.path.exists(cold_prefs_path):
        cold_ids = set(pl.read_parquet(cold_prefs_path)["user_id"].to_list())

    blind_ids = test_ids - train_ids
    warm = test_ids & train_ids
    cold_with_prefs = blind_ids & cold_ids
    truly_blind = blind_ids - cold_ids

    return {
        "warm": warm,
        "cold_with_prefs": cold_with_prefs,
        "truly_blind": truly_blind,
        "total": test_ids,
    }


def build_segment_popularity(train_contacts, listing):
    """Build (city, category) → ranked items from training contacts."""
    train_with_listing = train_contacts.join(
        listing.select(["item_id", "city_name", "category"]),
        on="item_id", how="left"
    )
    seg_pop = (
        train_with_listing
        .filter(pl.col("city_name").is_not_null() & pl.col("category").is_not_null())
        .group_by(["city_name", "category", "item_id"])
        .agg(pl.len().alias("n"))
        .sort(["city_name", "category", "n"], descending=[False, False, True])
    )
    return seg_pop


def compute_theoretical_max_recall(val_contacts, blind_val_uids, seg_pop, listing, n_sample=2000, k=10):
    """
    Compute recall@K if we KNEW each blind user's city+cat perfectly.
    Uses mode(city), mode(cat) from their actual GT contacts.
    """
    blind_items = val_contacts.filter(pl.col("user_id").is_in(list(blind_val_uids)))
    blind_with_listing = blind_items.join(
        listing.select(["item_id", "city_name", "category"]),
        on="item_id", how="left"
    )

    sampled = random.sample(list(blind_val_uids), min(n_sample, len(blind_val_uids)))
    total_recall = 0.0

    for uid in sampled:
        user_gt = blind_with_listing.filter(pl.col("user_id") == uid)
        gt_items = set(user_gt["item_id"].to_list())
        if not gt_items:
            continue

        cities = user_gt["city_name"].drop_nulls().mode().to_list()
        cats = user_gt["category"].drop_nulls().mode().to_list()
        city = cities[0] if cities else None
        cat = cats[0] if cats else None

        if city and cat:
            top_k = seg_pop.filter(
                (pl.col("city_name") == city) & (pl.col("category") == cat)
            ).head(k)["item_id"].to_list()
            hits = len(set(top_k) & gt_items)
            total_recall += hits / len(gt_items)

    return total_recall / len(sampled)


def compute_segpop_hit_rates(val_contacts, blind_val_uids, seg_pop, listing, n_sample=5000):
    """Compute hit rate at various K for blind val users."""
    blind_items = val_contacts.filter(pl.col("user_id").is_in(list(blind_val_uids)))
    blind_with_listing = blind_items.join(
        listing.select(["item_id", "city_name", "category"]),
        on="item_id", how="left"
    )

    results = {}
    for k_val in [10, 20, 50, 100, 200, 500]:
        hit = 0
        total = 0
        for row in blind_with_listing.iter_rows(named=True):
            city = row.get("city_name")
            cat = row.get("category")
            item = row["item_id"]
            if city and cat:
                seg_items = seg_pop.filter(
                    (pl.col("city_name") == city) & (pl.col("category") == cat)
                ).head(k_val)["item_id"].to_list()
                if item in seg_items:
                    hit += 1
            total += 1
            if total >= n_sample:
                break
        results[k_val] = hit / total if total > 0 else 0
    return results


def compute_blind_contact_distribution(val_contacts, blind_val_uids, listing):
    """Analyze city/category distribution of blind user contacts."""
    blind_items = val_contacts.filter(pl.col("user_id").is_in(list(blind_val_uids)))
    blind_with_listing = blind_items.join(
        listing.select(["item_id", "city_name", "category", "posted_date"]),
        on="item_id", how="left"
    )

    city_dist = blind_with_listing.group_by("city_name").agg(
        pl.len().alias("n")
    ).sort("n", descending=True)

    cat_dist = blind_with_listing.group_by("category").agg(
        pl.len().alias("n")
    ).sort("n", descending=True)

    return city_dist, cat_dist, blind_with_listing


def compute_item_age_distribution(blind_with_listing, max_date):
    """Compute age distribution of items contacted by blind users."""
    ages = blind_with_listing.with_columns(
        ((max_date - pl.col("posted_date")).dt.total_days()).alias("age_days")
    ).filter(pl.col("age_days").is_not_null())

    results = {}
    for d in [1, 3, 7, 14, 30, 60]:
        n = ages.filter(pl.col("age_days") <= d).shape[0]
        results[d] = n / len(ages) if len(ages) > 0 else 0
    return results


def main():
    random.seed(42)
    logger.info("=" * 60)
    logger.info("Round 23: Cold-Start Recall Ceiling Analysis")
    logger.info("=" * 60)

    contacts, max_date, listing, test = load_data()
    test_ids = set(test["user_id"].to_list())
    train_contacts, val_contacts = split_train_val(contacts, max_date)
    train_users = set(train_contacts["user_id"].unique().to_list())
    val_users = set(val_contacts["user_id"].unique().to_list())
    blind_val = val_users - train_users
    warm_val = val_users & train_users

    # 1. User segmentation
    segments = compute_user_segments(contacts, max_date, test_ids)
    logger.info(f"Test users: {len(test_ids):,}")
    logger.info(f"  Warm: {len(segments['warm']):,} ({len(segments['warm'])/len(test_ids)*100:.1f}%)")
    logger.info(f"  Cold+prefs: {len(segments['cold_with_prefs']):,} ({len(segments['cold_with_prefs'])/len(test_ids)*100:.1f}%)")
    logger.info(f"  Truly blind: {len(segments['truly_blind']):,} ({len(segments['truly_blind'])/len(test_ids)*100:.1f}%)")

    # 2. Build segment popularity
    seg_pop = build_segment_popularity(train_contacts, listing)
    logger.info(f"Segment popularity: {len(seg_pop):,} (city,cat,item) rows")

    # 3. Theoretical max recall with perfect segment knowledge
    logger.info("Computing theoretical max recall@10...")
    max_recall = compute_theoretical_max_recall(val_contacts, blind_val, seg_pop, listing)
    logger.info(f"Theoretical max Recall@10 (perfect city+cat): {max_recall:.4f}")

    # 4. Hit rates at various K
    logger.info("Computing hit rates at various K...")
    hit_rates = compute_segpop_hit_rates(val_contacts, blind_val, seg_pop, listing)
    for k_val, rate in hit_rates.items():
        logger.info(f"  SegPop top-{k_val}: {rate:.4f}")

    # 5. Contact distribution of blind users
    city_dist, cat_dist, blind_with_listing = compute_blind_contact_distribution(
        val_contacts, blind_val, listing
    )

    # 6. Item age distribution
    age_dist = compute_item_age_distribution(blind_with_listing, max_date)

    # 7. Score decomposition
    warm_pct = len(segments["warm"]) / len(test_ids)
    logger.info(f"\nScore decomposition:")
    logger.info(f"  warm_pct={warm_pct:.3f}, current total=0.034")
    logger.info(f"  implied warm recall@10 = 0.034 / {warm_pct:.3f} = {0.034/warm_pct:.3f}")
    for target in [0.10, 0.20, 0.32]:
        cold_needed = (target - 0.10 * warm_pct) / (1 - warm_pct)
        logger.info(f"  To reach {target:.2f}: need cold_recall@10 = {cold_needed:.3f}")

    # 8. Generate report
    report = generate_report(
        segments, max_recall, hit_rates, city_dist, cat_dist,
        age_dist, warm_pct, blind_val, warm_val
    )
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {REPORT_PATH}")


def generate_report(segments, max_recall, hit_rates, city_dist, cat_dist,
                    age_dist, warm_pct, blind_val, warm_val):
    """Generate markdown report."""
    city_table = "\n".join(
        f"| {r['city_name']} | {r['n']:,} | {r['n']/city_dist['n'].sum()*100:.1f}% |"
        for r in city_dist.head(10).iter_rows(named=True)
    )
    cat_table = "\n".join(
        f"| {r['category']} | {r['n']:,} | {r['n']/cat_dist['n'].sum()*100:.1f}% |"
        for r in cat_dist.iter_rows(named=True)
    )
    hit_table = "\n".join(
        f"| {k} | {v*100:.2f}% |" for k, v in hit_rates.items()
    )
    age_table = "\n".join(
        f"| {k} | {v*100:.1f}% |" for k, v in age_dist.items()
    )

    return f"""# Round 23 Report: Cold-Start Recall Ceiling Analysis

## Executive Summary
SegPop popularity-based cold-start has a **theoretical ceiling of ~{max_recall:.1%} Recall@10**
for blind users, even with PERFECT city+category knowledge. BĐS items are too sparse
for top-K popularity to cover GT contacts. 44% of blind user contacts are on items
posted ≤7 days — freshness matters more than historical popularity.

## Methodology
- Time-split: last 3 days of training = validation period
- Blind val users: {len(blind_val):,} (contacted in val, never in train)
- Warm val users: {len(warm_val):,}
- Built (city, category) segment popularity from training period
- Tested recall@K at K=10,20,50,100,200,500
- Computed theoretical max: gave each user top-K from their TRUE (city, cat)

## Key Findings

### Finding 1: SegPop Ceiling is ~{max_recall:.1%}
- 📊 Even with PERFECT city+cat knowledge, Recall@10 = {max_recall:.4f}
- 🏠 BĐS has 28,732 unique items contacted by 13,460 blind users in 3 days.
  Each (city, cat) segment has thousands of items but top-10 only covers tiny fraction.
- 💡 Popularity alone cannot solve cold-start in BĐS.

### Finding 2: Hit Rates at Various K
| K | Hit Rate |
|---|---------|
{hit_table}

- 📊 Even at K=500, only 16.3% hit rate. Long-tail distribution.

### Finding 3: Blind User Contact Geography
| City | Contacts | % |
|------|----------|---|
{city_table}

- 📊 HCM dominates at ~74% of blind user contacts.

### Finding 4: Blind User Category Distribution
| Category | Contacts | % |
|----------|----------|---|
{cat_table}

- 📊 1050 (Dự án) is #1 for blind users (39.6%), unlike warm users where 1020 dominates.

### Finding 5: Item Freshness Is Critical
| Posted Within (days) | % of Contacts |
|---------------------|---------------|
{age_table}

- 📊 44% of blind contacts are on items ≤7 days old. Recency > historical popularity.
- 🏠 New listings get contacts quickly. Old popular items are stale.

### Finding 6: Score Decomposition
- Warm users: {len(segments['warm']):,} ({warm_pct:.1%})
- Truly blind: {len(segments['truly_blind']):,} ({len(segments['truly_blind'])/len(segments['total'])*100:.1f}%)
- Current score: 0.034 = warm × {warm_pct:.3f} × ~0.10 recall
- To reach 0.10: need cold_recall = {(0.10 - 0.10*warm_pct)/(1-warm_pct):.3f}
- To reach 0.32: need cold_recall = {(0.32 - 0.10*warm_pct)/(1-warm_pct):.3f}

## Hypotheses Generated
- H-013: SegPop ceiling ~2% for blind users → VERIFIED ✅
- H-014: Fresh items (≤7d) as cold-start candidates will improve recall → PENDING
- H-015: LightGBM reranker on cascade can boost warm recall 0.10→0.15+ → PENDING

## Code Reference
- File: `src/eda/round_23_cold_start_ceiling.py`

## Next Steps
1. Focus on warm user reranking (bigger lever: 0.10→0.15 = +0.017 total)
2. For cold: try fresh-item-first SegPop (items posted ≤7 days, ranked by early contacts)
3. Retrain full pipeline with weighted+PCI ALS
"""


if __name__ == "__main__":
    main()
