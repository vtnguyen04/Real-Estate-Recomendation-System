"""
Round 02: Schema Deep-Dive & Relationship Validation
Phase 1 — Foundation

Hypotheses to verify:
- H-001: project_id nullity correlates with non-apartment categories
- H-002: Cold-start problem — test users without train history
- H-003: dwell_time_sec is in milliseconds, not seconds
- H-004: other_interaction — positive or noise? (cross-ref with is_contact)

Additional questions:
- FK integrity: item_ids in facts exist in dim_listing?
- Orphan records?
- Non-login user prevalence?

"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import polars as pl
from config.settings import PipelineConfig
from src.utils.profiler import scan_table, get_row_count
from src.utils.report_writer import write_report
from src.utils.logging import get_logger

logger = get_logger("eda.round_02")

config = PipelineConfig()
TRAIN_PATH = config.data.train_path
TEST_FILE = os.path.join(config.data.test_path, "test_users.parquet")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "reports", "round_02_report.md")
SAMPLE_SIZE = 1_000_000


def main():
    logger.info("=" * 60)
    logger.info("ROUND 02: Schema Deep-Dive & Relationship Validation")
    logger.info("=" * 60)

    findings = []
    hypothesis_results = {}

    # ─── Load LazyFrames ──────────────────────────────────
    lf_listing = scan_table(os.path.join(TRAIN_PATH, "dim_listing"))
    lf_events = scan_table(os.path.join(TRAIN_PATH, "fact_user_events"))
    lf_snapshot = scan_table(os.path.join(TRAIN_PATH, "fact_listing_snapshot"))
    lf_interactions = scan_table(os.path.join(TRAIN_PATH, "fact_post_contact_interactions"))
    df_test = pl.read_parquet(TEST_FILE)

    # ═══════════════════════════════════════════════════════
    # H-001: project_id nullity by category
    # ═══════════════════════════════════════════════════════
    logger.info("Verifying H-001: project_id nullity by category...")

    h001_result = (
        lf_listing
        .select(['category', 'project_id'])
        .group_by('category')
        .agg([
            pl.len().alias('total'),
            pl.col('project_id').is_null().sum().alias('project_id_null')
        ])
        .with_columns(
            (pl.col('project_id_null') / pl.col('total') * 100).round(2).alias('null_pct')
        )
        .sort('category')
        .collect()
    )

    findings.append(f"### H-001 Verification: project_id Null Rate by Category\n```\n{h001_result}\n```\n")
    hypothesis_results['H-001'] = h001_result

    # ═══════════════════════════════════════════════════════
    # H-002: Cold-start analysis (test users overlap with train)
    # ═══════════════════════════════════════════════════════
    logger.info("Verifying H-002: Cold-start analysis...")

    # Get unique user_ids from training events (only logged-in users matter)
    train_users = (
        lf_events
        .filter(pl.col('is_login') == 'login')
        .select('user_id')
        .unique()
        .collect()
    )

    test_user_ids = df_test.select('user_id')

    # Inner join to find overlap
    overlap = test_user_ids.join(train_users, on='user_id', how='inner')

    overlap_count = len(overlap)
    total_test = len(test_user_ids)
    cold_start_count = total_test - overlap_count
    cold_start_pct = round((cold_start_count / total_test) * 100, 2)

    h002_text = (
        f"- Total test users: {total_test:,}\n"
        f"- Users with training event history: {overlap_count:,} ({round(overlap_count/total_test*100, 2)}%)\n"
        f"- Cold-start users (NO history): {cold_start_count:,} ({cold_start_pct}%)\n"
    )
    findings.append(f"### H-002 Verification: Test User Cold-Start Analysis\n{h002_text}\n")
    hypothesis_results['H-002'] = {'overlap': overlap_count, 'cold_start': cold_start_count, 'cold_start_pct': cold_start_pct}

    # Also check overlap with fact_post_contact_interactions
    interaction_users = (
        lf_interactions
        .select('user_id')
        .unique()
        .collect()
    )
    overlap_interactions = test_user_ids.join(interaction_users, on='user_id', how='inner')
    h002_text_extra = (
        f"- Test users with contact interaction history: {len(overlap_interactions):,} "
        f"({round(len(overlap_interactions)/total_test*100, 2)}%)\n"
    )
    findings.append(f"### H-002 Extra: Test User Overlap with Contact Interactions\n{h002_text_extra}\n")

    # ═══════════════════════════════════════════════════════
    # H-003: dwell_time_sec unit verification
    # ═══════════════════════════════════════════════════════
    logger.info("Verifying H-003: dwell_time_sec units...")

    # Compare dwell_time by event_type — contact events should have higher dwell
    dwell_by_event = (
        lf_events
        .filter(pl.col('dwell_time_sec').is_not_null())
        .head(SAMPLE_SIZE)
        .group_by('event_type')
        .agg([
            pl.col('dwell_time_sec').median().alias('median_dwell'),
            pl.col('dwell_time_sec').mean().alias('mean_dwell'),
            pl.col('dwell_time_sec').quantile(0.25).alias('p25_dwell'),
            pl.col('dwell_time_sec').quantile(0.75).alias('p75_dwell'),
            pl.len().alias('count')
        ])
        .sort('median_dwell', descending=True)
        .collect()
    )

    findings.append(f"### H-003 Verification: Dwell Time by Event Type\n```\n{dwell_by_event}\n```\n")

    # If in milliseconds, divide by 1000 and show
    dwell_converted = dwell_by_event.with_columns([
        (pl.col('median_dwell') / 1000).round(1).alias('median_sec_if_ms'),
        (pl.col('mean_dwell') / 1000).round(1).alias('mean_sec_if_ms'),
    ]).select(['event_type', 'median_dwell', 'median_sec_if_ms', 'mean_dwell', 'mean_sec_if_ms', 'count'])

    findings.append(f"### H-003 Conversion Test: If dwell_time_sec is actually milliseconds\n```\n{dwell_converted}\n```\n")
    hypothesis_results['H-003'] = dwell_converted

    # ═══════════════════════════════════════════════════════
    # H-004: other_interaction × is_contact cross-tab
    # ═══════════════════════════════════════════════════════
    logger.info("Verifying H-004: other_interaction × is_contact...")

    event_contact_crosstab = (
        lf_events
        .head(SAMPLE_SIZE)
        .group_by(['event_type', 'is_contact'])
        .agg(pl.len().alias('count'))
        .sort(['event_type', 'is_contact'])
        .collect()
    )

    findings.append(f"### H-004 Verification: event_type × is_contact Cross-tabulation\n```\n{event_contact_crosstab}\n```\n")
    hypothesis_results['H-004'] = event_contact_crosstab

    # ═══════════════════════════════════════════════════════
    # FK Integrity Checks
    # ═══════════════════════════════════════════════════════
    logger.info("Checking FK integrity...")

    listing_item_ids = lf_listing.select('item_id').unique().collect()

    # Check events item_ids in dim_listing
    event_item_ids = lf_events.head(SAMPLE_SIZE).select('item_id').unique().collect()
    orphan_events = event_item_ids.join(listing_item_ids, on='item_id', how='anti')

    fk_text = (
        f"- Unique item_ids in events sample: {len(event_item_ids):,}\n"
        f"- Orphan item_ids (in events but NOT in dim_listing): {len(orphan_events):,} "
        f"({round(len(orphan_events)/len(event_item_ids)*100, 2)}%)\n"
    )
    findings.append(f"### FK Integrity: fact_user_events → dim_listing\n{fk_text}\n")

    # Check snapshot item_ids in dim_listing
    snapshot_item_ids = lf_snapshot.select('item_id').unique().collect()
    orphan_snapshot = snapshot_item_ids.join(listing_item_ids, on='item_id', how='anti')

    fk_text2 = (
        f"- Unique item_ids in fact_listing_snapshot: {len(snapshot_item_ids):,}\n"
        f"- Orphan item_ids (NOT in dim_listing): {len(orphan_snapshot):,} "
        f"({round(len(orphan_snapshot)/len(snapshot_item_ids)*100, 2)}%)\n"
    )
    findings.append(f"### FK Integrity: fact_listing_snapshot → dim_listing\n{fk_text2}\n")

    # ═══════════════════════════════════════════════════════
    # Non-login user prevalence
    # ═══════════════════════════════════════════════════════
    logger.info("Checking login vs non-login distribution...")

    login_dist = (
        lf_events
        .head(SAMPLE_SIZE)
        .group_by('is_login')
        .agg(pl.len().alias('count'))
        .collect()
    )
    findings.append(f"### Login vs Non-login Distribution (1M sample)\n```\n{login_dist}\n```\n")

    # ═══════════════════════════════════════════════════════
    # GENERATE REPORT
    # ═══════════════════════════════════════════════════════
    logger.info("Generating report...")

    report_parts = [
        "# Round 02 Report: Schema Deep-Dive & Relationship Validation\n",
        "## Executive Summary\n",
        "Verified 4 hypotheses from Round 01. Checked FK integrity across tables. "
        "Analyzed cold-start problem and dwell_time units.\n",
        "",
        "## Methodology\n",
        "Used Polars LazyFrame aggregations with pushdown optimization. "
        "All data loaded via `src.utils.profiler.scan_table()`. "
        "fact_user_events sampled at 1M rows where noted.\n",
        "",
        "## Key Findings\n",
    ]

    for finding in findings:
        report_parts.append(finding)

    # Hypothesis verdicts
    report_parts.append("\n## Hypothesis Verdicts\n")
    report_parts.append("*Verdicts based on data evidence above. Final determination requires domain expert review.*\n")
    report_parts.append("- **H-001**: [VERDICT PENDING — see data above]\n")
    report_parts.append("- **H-002**: [VERDICT PENDING — see data above]\n")
    report_parts.append("- **H-003**: [VERDICT PENDING — see data above]\n")
    report_parts.append("- **H-004**: [VERDICT PENDING — see data above]\n")

    report_parts.append("\n## New Observations\n")
    report_parts.append("*(To be filled after analyzing the output)*\n")

    report_parts.append("\n## Code Reference\n")
    report_parts.append("- Code: `src/eda/round_02_schema_relations.py`\n")
    report_parts.append("- Modules used: `src/utils/profiler.py`, `src/utils/report_writer.py`\n")

    report_parts.append("\n## Next Steps\n")
    report_parts.append("Round 03: Basic Distributions & Statistical Overview\n")

    report_content = "\n".join(report_parts)
    write_report(REPORT_PATH, report_content)

    logger.info(f"Report written to {REPORT_PATH}")
    logger.info("Round 02 complete.")


if __name__ == "__main__":
    main()
