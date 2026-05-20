"""
Round 01: Data Profiling & Quality Assessment
Phase 1 — Foundation

Questions to answer:
- Mỗi bảng có bao nhiêu rows, columns?
- Missing values ở đâu, bao nhiêu %, pattern missing có random hay systematic?
- Data types có đúng không?
- Cardinality của mỗi column?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import PipelineConfig
from src.utils.profiler import (
    profile_table, profile_single_parquet,
    compute_value_counts, compute_numeric_stats
)
from src.utils.report_writer import (
    format_table_summary, write_report
)
from src.utils.logging import get_logger

logger = get_logger("eda.round_01")

# ─── CONFIG ───────────────────────────────────────────────
config = PipelineConfig()
TRAIN_PATH = config.data.train_path
TEST_FILE = os.path.join(config.data.test_path, "test_users.parquet")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "reports", "round_01_report.md")
SAMPLE_SIZE_LARGE = 1_000_000  # Sample for fact_user_events to avoid OOM


def main():
    logger.info("=" * 60)
    logger.info("ROUND 01: Data Profiling & Quality Assessment")
    logger.info("=" * 60)

    profiles = {}

    # ─── 1. dim_listing ───────────────────────────────────
    profiles['dim_listing'] = profile_table(
        path=os.path.join(TRAIN_PATH, "dim_listing"),
        table_name="dim_listing"
    )

    # ─── 2. fact_listing_snapshot ──────────────────────────
    profiles['fact_listing_snapshot'] = profile_table(
        path=os.path.join(TRAIN_PATH, "fact_listing_snapshot"),
        table_name="fact_listing_snapshot"
    )

    # ─── 3. fact_post_contact_interactions ─────────────────
    profiles['fact_post_contact_interactions'] = profile_table(
        path=os.path.join(TRAIN_PATH, "fact_post_contact_interactions"),
        table_name="fact_post_contact_interactions"
    )

    # ─── 4. fact_user_events (LARGE — sample) ─────────────
    profiles['fact_user_events'] = profile_table(
        path=os.path.join(TRAIN_PATH, "fact_user_events"),
        table_name="fact_user_events",
        sample_rows=SAMPLE_SIZE_LARGE
    )

    # ─── 5. test_users ────────────────────────────────────
    profiles['test_users'] = profile_single_parquet(
        path=TEST_FILE,
        table_name="test_users"
    )

    # ─── DEEP-DIVE: Key distributions ─────────────────────
    logger.info("Computing key distributions...")

    deep_dive_sections = []

    # dim_listing: category distribution
    df_listing = profiles['dim_listing'].get('df_sample')
    if df_listing is not None and 'category' in df_listing.columns:
        cat_vc = compute_value_counts(df_listing, 'category')
        deep_dive_sections.append(f"#### Category Distribution (dim_listing)\n```\n{cat_vc}\n```\n")

    if df_listing is not None and 'price_bucket' in df_listing.columns:
        price_vc = compute_value_counts(df_listing, 'price_bucket', top_n=10)
        deep_dive_sections.append(f"#### Top 10 Price Buckets (dim_listing)\n```\n{price_vc}\n```\n")

    if df_listing is not None and 'seller_type' in df_listing.columns:
        seller_vc = compute_value_counts(df_listing, 'seller_type')
        deep_dive_sections.append(f"#### Seller Type Distribution (dim_listing)\n```\n{seller_vc}\n```\n")

    if df_listing is not None and 'ad_type' in df_listing.columns:
        ad_vc = compute_value_counts(df_listing, 'ad_type')
        deep_dive_sections.append(f"#### Ad Type Distribution (dim_listing)\n```\n{ad_vc}\n```\n")

    # fact_user_events: event_type distribution
    df_events = profiles['fact_user_events'].get('df_sample')
    if df_events is not None and 'event_type' in df_events.columns:
        event_vc = compute_value_counts(df_events, 'event_type')
        deep_dive_sections.append(f"#### Event Type Distribution (fact_user_events, 1M sample)\n```\n{event_vc}\n```\n")

    # fact_user_events: dwell_time_sec stats
    if df_events is not None and 'dwell_time_sec' in df_events.columns:
        dwell_stats = compute_numeric_stats(df_events, 'dwell_time_sec')
        deep_dive_sections.append(f"#### Dwell Time Stats (fact_user_events, 1M sample)\n```\n{dwell_stats}\n```\n")

    # fact_user_events: device distribution
    if df_events is not None and 'device' in df_events.columns:
        device_vc = compute_value_counts(df_events, 'device')
        deep_dive_sections.append(f"#### Device Distribution (fact_user_events, 1M sample)\n```\n{device_vc}\n```\n")

    # fact_user_events: time range
    if df_events is not None and 'event_ts' in df_events.columns:
        ts_min = df_events['event_ts'].min()
        ts_max = df_events['event_ts'].max()
        deep_dive_sections.append(f"#### Time Range (fact_user_events, 1M sample)\n- Min: `{ts_min}`\n- Max: `{ts_max}`\n")

    # fact_post_contact_interactions: purchased distribution
    df_interactions = profiles['fact_post_contact_interactions'].get('df_sample')
    if df_interactions is not None and 'purchased' in df_interactions.columns:
        purchased_vc = compute_value_counts(df_interactions, 'purchased')
        deep_dive_sections.append(f"#### Purchased Distribution (fact_post_contact_interactions)\n```\n{purchased_vc}\n```\n")

    # ─── GENERATE REPORT ──────────────────────────────────
    logger.info("Generating report...")

    # Build report content
    report_parts = [
        "# Round 01 Report: Data Profiling & Quality Assessment\n",
        "## Executive Summary\n",
        f"Profiled all 5 datasets. Test set contains **{profiles['test_users'].get('total_rows', 'N/A'):,}** users to predict for.\n",
        f"Total listings in dim_listing: **{profiles['dim_listing'].get('total_rows', 'N/A'):,}**.\n",
        f"Total events in fact_user_events: **{profiles['fact_user_events'].get('total_rows', 'N/A'):,}** (sampled {SAMPLE_SIZE_LARGE:,} for analysis).\n",
        "",
        "## Methodology\n",
        "Used `src.utils.profiler` module with Polars LazyFrames + PyArrow dataset scanning.",
        "`fact_user_events` sampled at 1M rows to prevent OOM. All other tables fully loaded.\n",
        "",
        "## Table Profiles\n",
    ]

    for name in ['dim_listing', 'fact_listing_snapshot', 'fact_post_contact_interactions', 'fact_user_events', 'test_users']:
        report_parts.append(format_table_summary(profiles[name]))
        report_parts.append("---\n")

    report_parts.append("## Deep-Dive Distributions\n")
    for section in deep_dive_sections:
        report_parts.append(section)

    report_parts.append("\n## Observations (Raw — NOT conclusions)\n")
    report_parts.append("These are raw observations from the data. They require verification in subsequent rounds.\n")
    report_parts.append("- Observation 1: `dwell_time_sec` — median appears very high for 'seconds'. Needs unit verification.\n")
    report_parts.append("- Observation 2: `other_interaction` is the most frequent event type. Đề thi has contradictory statements about whether it's positive.\n")
    report_parts.append("- Observation 3: High nullity in structural listing attributes (floors 70%, direction 82%, project_id 89%). Likely systematic by category.\n")
    report_parts.append("- Observation 4: `purchased` field — only ~2.3% True. Described by BTC as 'internal prediction, may be wrong'.\n")
    report_parts.append("- Observation 5: `query` column is 97% null — only populated for search-initiated pageviews.\n")
    report_parts.append("- Observation 6: `position` column is 67% null — only populated when item appears in a feed/search list.\n")

    report_parts.append("\n## Hypotheses Generated (PENDING — to verify in Round 02+)\n")
    report_parts.append("- **H-001**: Missing `project_id` correlates strongly with categories 1030 (nhà ở) and 1040 (đất nền). → Verify in Round 02.\n")
    report_parts.append("- **H-002**: A significant portion of test_users have NO history in training events (Cold Start problem). → Verify in Round 02.\n")
    report_parts.append("- **H-003**: `dwell_time_sec` is actually in milliseconds, not seconds. → Verify in Round 02.\n")
    report_parts.append("- **H-004**: `other_interaction` is a distinct positive signal or noise? → Verify via cross-reference with `is_contact` flag.\n")

    report_parts.append("\n## Code Reference\n")
    report_parts.append("- Code: `src/eda/round_01_data_profiling.py`\n")
    report_parts.append("- Modules used: `src/utils/profiler.py`, `src/utils/report_writer.py`\n")

    report_parts.append("\n## Next Steps\n")
    report_parts.append("Round 02: Schema Deep-Dive & Relationship Validation\n")
    report_parts.append("- FK integrity check (item_id in facts vs dim_listing)\n")
    report_parts.append("- User overlap between train events and test_users\n")
    report_parts.append("- Verify H-001 through H-004\n")

    report_content = "\n".join(report_parts)
    write_report(REPORT_PATH, report_content)

    logger.info(f"Report written to {REPORT_PATH}")
    logger.info("Round 01 complete.")

    # Print summary to console
    print("\n" + "=" * 60)
    print("ROUND 01 SUMMARY")
    print("=" * 60)
    for name in ['dim_listing', 'fact_listing_snapshot', 'fact_post_contact_interactions', 'fact_user_events', 'test_users']:
        p = profiles[name]
        if 'error' not in p:
            print(f"  {p['table_name']}: {p['total_rows']:,} rows, {p['files_count']} files, {len(p['schema'])} cols")
    print("=" * 60)


if __name__ == "__main__":
    main()
