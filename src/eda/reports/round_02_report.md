# Round 02 Report: Schema Deep-Dive & Relationship Validation

## Executive Summary

Verified 4 hypotheses from Round 01. Checked FK integrity across tables. Analyzed cold-start problem and dwell_time units.


## Methodology

Used Polars LazyFrame aggregations with pushdown optimization. All data loaded via `src.utils.profiler.scan_table()`. fact_user_events sampled at 1M rows where noted.


## Key Findings

### H-001 Verification: project_id Null Rate by Category
```
shape: (5, 4)
┌──────────┬─────────┬─────────────────┬──────────┐
│ category ┆ total   ┆ project_id_null ┆ null_pct │
│ ---      ┆ ---     ┆ ---             ┆ ---      │
│ i64      ┆ u32     ┆ u32             ┆ f64      │
╞══════════╪═════════╪═════════════════╪══════════╡
│ 1010     ┆ 611823  ┆ 358538          ┆ 58.6     │
│ 1020     ┆ 1507864 ┆ 1458549         ┆ 96.73    │
│ 1030     ┆ 252402  ┆ 236091          ┆ 93.54    │
│ 1040     ┆ 373469  ┆ 341485          ┆ 91.44    │
│ 1050     ┆ 361556  ┆ 361556          ┆ 100.0    │
└──────────┴─────────┴─────────────────┴──────────┘
```

### H-002 Verification: Test User Cold-Start Analysis
- Total test users: 161,568
- Users with training event history: 58,153 (35.99%)
- Cold-start users (NO history): 103,415 (64.01%)


### H-002 Extra: Test User Overlap with Contact Interactions
- Test users with contact interaction history: 60,212 (37.27%)


### H-003 Verification: Dwell Time by Event Type
```
shape: (1, 6)
┌────────────┬──────────────┬──────────────┬───────────┬───────────┬─────────┐
│ event_type ┆ median_dwell ┆ mean_dwell   ┆ p25_dwell ┆ p75_dwell ┆ count   │
│ ---        ┆ ---          ┆ ---          ┆ ---       ┆ ---       ┆ ---     │
│ str        ┆ f64          ┆ f64          ┆ f64       ┆ f64       ┆ u32     │
╞════════════╪══════════════╪══════════════╪═══════════╪═══════════╪═════════╡
│ pageview   ┆ 17915.0      ┆ 52334.427702 ┆ 7159.0    ┆ 41297.0   ┆ 1000000 │
└────────────┴──────────────┴──────────────┴───────────┴───────────┴─────────┘
```

### H-003 Conversion Test: If dwell_time_sec is actually milliseconds
```
shape: (1, 6)
┌────────────┬──────────────┬──────────────────┬──────────────┬────────────────┬─────────┐
│ event_type ┆ median_dwell ┆ median_sec_if_ms ┆ mean_dwell   ┆ mean_sec_if_ms ┆ count   │
│ ---        ┆ ---          ┆ ---              ┆ ---          ┆ ---            ┆ ---     │
│ str        ┆ f64          ┆ f64              ┆ f64          ┆ f64            ┆ u32     │
╞════════════╪══════════════╪══════════════════╪══════════════╪════════════════╪═════════╡
│ pageview   ┆ 17915.0      ┆ 17.9             ┆ 52334.427702 ┆ 52.3           ┆ 1000000 │
└────────────┴──────────────┴──────────────────┴──────────────┴────────────────┴─────────┘
```

### H-004 Verification: event_type × is_contact Cross-tabulation
```
shape: (6, 3)
┌───────────────────┬────────────┬────────┐
│ event_type        ┆ is_contact ┆ count  │
│ ---               ┆ ---        ┆ ---    │
│ str               ┆ i64        ┆ u32    │
╞═══════════════════╪════════════╪════════╡
│ contact_chat      ┆ 1          ┆ 6628   │
│ contact_sms       ┆ 1          ┆ 798    │
│ contact_zalo      ┆ 1          ┆ 1452   │
│ other_interaction ┆ 1          ┆ 561188 │
│ pageview          ┆ 0          ┆ 404986 │
│ view_phone        ┆ 1          ┆ 24948  │
└───────────────────┴────────────┴────────┘
```

### FK Integrity: fact_user_events → dim_listing
- Unique item_ids in events sample: 330,140
- Orphan item_ids (in events but NOT in dim_listing): 9,584 (2.9%)


### FK Integrity: fact_listing_snapshot → dim_listing
- Unique item_ids in fact_listing_snapshot: 703,821
- Orphan item_ids (NOT in dim_listing): 11,158 (1.59%)


### Login vs Non-login Distribution (1M sample)
```
shape: (2, 2)
┌───────────┬────────┐
│ is_login  ┆ count  │
│ ---       ┆ ---    │
│ str       ┆ u32    │
╞═══════════╪════════╡
│ non-login ┆ 366436 │
│ login     ┆ 633564 │
└───────────┴────────┘
```


## Hypothesis Verdicts

*Verdicts based on data evidence above. Final determination requires domain expert review.*

- **H-001**: [VERDICT PENDING — see data above]

- **H-002**: [VERDICT PENDING — see data above]

- **H-003**: [VERDICT PENDING — see data above]

- **H-004**: [VERDICT PENDING — see data above]


## New Observations

*(To be filled after analyzing the output)*


## Code Reference

- Code: `src/eda/round_02_schema_relations.py`

- Modules used: `src/utils/profiler.py`, `src/utils/report_writer.py`


## Next Steps

Round 03: Basic Distributions & Statistical Overview
