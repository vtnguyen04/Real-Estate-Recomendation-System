# Round 01 Report: Data Profiling & Quality Assessment

## Executive Summary

Profiled all 5 datasets. Test set contains **161,568** users to predict for.

Total listings in dim_listing: **3,107,114**.

Total events in fact_user_events: **161,731,336** (sampled 1,000,000 for analysis).


## Methodology

Used `src.utils.profiler` module with Polars LazyFrames + PyArrow dataset scanning.
`fact_user_events` sampled at 1M rows to prevent OOM. All other tables fully loaded.


## Table Profiles

### dim_listing

- **Total Rows**: 3,107,114
- **Files**: 40
- **Sample Analyzed**: 3,107,114
- **Columns**: 24

#### Schema
| Column | Data Type |
|--------|-----------|
| `item_id` | String |
| `seller_id` | String |
| `category` | Int64 |
| `title` | String |
| `seller_type` | String |
| `ad_type` | String |
| `ad_status` | String |
| `area_sqm` | Float64 |
| `bedrooms` | Int64 |
| `bathrooms` | Int64 |
| `floors` | Int64 |
| `width_m` | Float64 |
| `direction` | String |
| `legal_status` | String |
| `house_type` | String |
| `furnishing` | String |
| `city_name` | String |
| `district_name` | String |
| `ward_name` | String |
| `project_id` | String |
| `price_bucket` | String |
| `images_count` | Int64 |
| `posted_date` | Date |
| `expected_expired_date` | Date |

#### Missing Values
| Column | Null Count | % Missing |
|--------|-----------|-----------|
| `project_id` | 2,756,219 | 88.71% |
| `direction` | 2,552,488 | 82.15% |
| `floors` | 2,191,222 | 70.52% |
| `furnishing` | 1,703,150 | 54.81% |
| `width_m` | 1,648,943 | 53.07% |
| `house_type` | 1,599,264 | 51.47% |
| `bathrooms` | 1,393,461 | 44.85% |
| `bedrooms` | 987,469 | 31.78% |
| `legal_status` | 428,389 | 13.79% |
| `images_count` | 10,547 | 0.34% |
| `area_sqm` | 59 | 0.00% |
| `district_name` | 6 | 0.00% |

#### Cardinality (Low-cardinality columns)
| Column | Unique Values |
|--------|--------------|
| `seller_type` | 2 |
| `ad_type` | 2 |
| `category` | 5 |
| `ad_status` | 5 |
| `house_type` | 5 |
| `furnishing` | 6 |
| `bathrooms` | 8 |
| `direction` | 9 |
| `bedrooms` | 12 |
| `legal_status` | 12 |
| `price_bucket` | 21 |
| `city_name` | 63 |
| `floors` | 168 |
| `images_count` | 175 |

---

### fact_listing_snapshot

- **Total Rows**: 19,762,167
- **Files**: 62
- **Sample Analyzed**: 19,762,167
- **Columns**: 5

#### Schema
| Column | Data Type |
|--------|-----------|
| `item_id` | String |
| `date` | Date |
| `views_24h` | Int64 |
| `contacts_24h` | Int64 |
| `listing_age_days` | Int64 |

#### Missing Values
| Column | Null Count | % Missing |
|--------|-----------|-----------|
| `contacts_24h` | 17,421,114 | 88.15% |
| `views_24h` | 9,288,229 | 47.00% |
| `listing_age_days` | 1,192 | 0.01% |

#### Cardinality (Low-cardinality columns)
| Column | Unique Values |
|--------|--------------|
| `date` | 152 |
| `contacts_24h` | 175 |

---

### fact_post_contact_interactions

- **Total Rows**: 25,486,445
- **Files**: 147
- **Sample Analyzed**: 25,486,445
- **Columns**: 10

#### Schema
| Column | Data Type |
|--------|-----------|
| `user_id` | String |
| `item_id` | String |
| `date` | Date |
| `adview_count` | Int64 |
| `lead_count` | Int64 |
| `chat_message_count` | Int64 |
| `chat_turn_count` | Int64 |
| `chat_lead` | Int64 |
| `purchased` | Boolean |
| `category` | Int64 |

#### Missing Values
| Column | Null Count | % Missing |
|--------|-----------|-----------|
| `chat_message_count` | 23,699,558 | 92.99% |
| `chat_turn_count` | 23,699,558 | 92.99% |
| `lead_count` | 22,829,994 | 89.58% |
| `chat_lead` | 22,829,994 | 89.58% |
| `adview_count` | 740,452 | 2.91% |

#### Cardinality (Low-cardinality columns)
| Column | Unique Values |
|--------|--------------|
| `purchased` | 2 |
| `chat_lead` | 3 |
| `category` | 5 |
| `chat_turn_count` | 58 |
| `adview_count` | 73 |
| `date` | 142 |
| `lead_count` | 155 |
| `chat_message_count` | 164 |

---

### fact_user_events

- **Total Rows**: 161,731,336
- **Files**: 500
- **Sample Analyzed**: 1,000,000
- **Columns**: 16

#### Schema
| Column | Data Type |
|--------|-----------|
| `is_login` | String |
| `user_id` | String |
| `session_id` | String |
| `event_id` | String |
| `item_id` | String |
| `city_name` | String |
| `category` | Int64 |
| `event_type` | String |
| `query` | String |
| `event_ts` | Datetime(time_unit='us', time_zone=None) |
| `surface` | String |
| `position` | Int64 |
| `device` | String |
| `dwell_time_sec` | Int64 |
| `is_contact` | Int64 |
| `date` | Date |

#### Missing Values
| Column | Null Count | % Missing |
|--------|-----------|-----------|
| `query` | 972,716 | 97.27% |
| `position` | 666,119 | 66.61% |
| `dwell_time_sec` | 626,088 | 62.61% |
| `city_name` | 21 | 0.00% |

#### Cardinality (Low-cardinality columns)
| Column | Unique Values |
|--------|--------------|
| `is_login` | 2 |
| `surface` | 2 |
| `is_contact` | 2 |
| `device` | 4 |
| `category` | 6 |
| `event_type` | 6 |
| `city_name` | 61 |
| `date` | 152 |

---

### test_users

- **Total Rows**: 161,568
- **Files**: 1
- **Sample Analyzed**: 161,568
- **Columns**: 1

#### Schema
| Column | Data Type |
|--------|-----------|
| `user_id` | String |

#### Missing Values
No missing values detected.

#### Cardinality (Low-cardinality columns)
No low-cardinality columns.

---

## Deep-Dive Distributions

#### Category Distribution (dim_listing)
```
shape: (5, 2)
┌──────────┬─────────┐
│ category ┆ count   │
│ ---      ┆ ---     │
│ i64      ┆ u32     │
╞══════════╪═════════╡
│ 1020     ┆ 1507864 │
│ 1010     ┆ 611823  │
│ 1040     ┆ 373469  │
│ 1050     ┆ 361556  │
│ 1030     ┆ 252402  │
└──────────┴─────────┘
```

#### Top 10 Price Buckets (dim_listing)
```
shape: (10, 2)
┌───────────────┬────────┐
│ price_bucket  ┆ count  │
│ ---           ┆ ---    │
│ str           ┆ u32    │
╞═══════════════╪════════╡
│ 3B–5B         ┆ 389913 │
│ 5B–7B         ┆ 306436 │
│ 3M–5M/tháng   ┆ 303905 │
│ 5M–7M/tháng   ┆ 251235 │
│ 7B–10B        ┆ 221694 │
│ 7M–10M/tháng  ┆ 220654 │
│ 10M–15M/tháng ┆ 202563 │
│ 2B–3B         ┆ 177297 │
│ >30M/tháng    ┆ 173355 │
│ 10B–15B       ┆ 118301 │
└───────────────┴────────┘
```

#### Seller Type Distribution (dim_listing)
```
shape: (2, 2)
┌─────────────┬─────────┐
│ seller_type ┆ count   │
│ ---         ┆ ---     │
│ str         ┆ u32     │
╞═════════════╪═════════╡
│ agent       ┆ 2593063 │
│ private     ┆ 514051  │
└─────────────┴─────────┘
```

#### Ad Type Distribution (dim_listing)
```
shape: (2, 2)
┌─────────┬─────────┐
│ ad_type ┆ count   │
│ ---     ┆ ---     │
│ str     ┆ u32     │
╞═════════╪═════════╡
│ sell    ┆ 1620385 │
│ let     ┆ 1486729 │
└─────────┴─────────┘
```

#### Event Type Distribution (fact_user_events, 1M sample)
```
shape: (6, 2)
┌───────────────────┬────────┐
│ event_type        ┆ count  │
│ ---               ┆ ---    │
│ str               ┆ u32    │
╞═══════════════════╪════════╡
│ other_interaction ┆ 561188 │
│ pageview          ┆ 404986 │
│ view_phone        ┆ 24948  │
│ contact_chat      ┆ 6628   │
│ contact_zalo      ┆ 1452   │
│ contact_sms       ┆ 798    │
└───────────────────┴────────┘
```

#### Dwell Time Stats (fact_user_events, 1M sample)
```
shape: (9, 2)
┌────────────┬──────────────┐
│ statistic  ┆ value        │
│ ---        ┆ ---          │
│ str        ┆ f64          │
╞════════════╪══════════════╡
│ count      ┆ 373912.0     │
│ null_count ┆ 626088.0     │
│ mean       ┆ 52096.483822 │
│ std        ┆ 157454.84201 │
│ min        ┆ 0.0          │
│ 25%        ┆ 7073.0       │
│ 50%        ┆ 17826.0      │
│ 75%        ┆ 41148.0      │
│ max        ┆ 1.5688164e7  │
└────────────┴──────────────┘
```

#### Device Distribution (fact_user_events, 1M sample)
```
shape: (4, 2)
┌─────────┬────────┐
│ device  ┆ count  │
│ ---     ┆ ---    │
│ str     ┆ u32    │
╞═════════╪════════╡
│ Desktop ┆ 324647 │
│ MSite   ┆ 265459 │
│ iOS     ┆ 253275 │
│ Android ┆ 156619 │
└─────────┴────────┘
```

#### Time Range (fact_user_events, 1M sample)
- Min: `2025-11-09 00:00:09.728002`
- Max: `2026-04-09 23:59:40.477005`

#### Purchased Distribution (fact_post_contact_interactions)
```
shape: (2, 2)
┌───────────┬──────────┐
│ purchased ┆ count    │
│ ---       ┆ ---      │
│ bool      ┆ u32      │
╞═══════════╪══════════╡
│ false     ┆ 24899961 │
│ true      ┆ 586484   │
└───────────┴──────────┘
```


## Observations (Raw — NOT conclusions)

These are raw observations from the data. They require verification in subsequent rounds.

- Observation 1: `dwell_time_sec` — median appears very high for 'seconds'. Needs unit verification.

- Observation 2: `other_interaction` is the most frequent event type. Đề thi has contradictory statements about whether it's positive.

- Observation 3: High nullity in structural listing attributes (floors 70%, direction 82%, project_id 89%). Likely systematic by category.

- Observation 4: `purchased` field — only ~2.3% True. Described by BTC as 'internal prediction, may be wrong'.

- Observation 5: `query` column is 97% null — only populated for search-initiated pageviews.

- Observation 6: `position` column is 67% null — only populated when item appears in a feed/search list.


## Hypotheses Generated (PENDING — to verify in Round 02+)

- **H-001**: Missing `project_id` correlates strongly with categories 1030 (nhà ở) and 1040 (đất nền). → Verify in Round 02.

- **H-002**: A significant portion of test_users have NO history in training events (Cold Start problem). → Verify in Round 02.

- **H-003**: `dwell_time_sec` is actually in milliseconds, not seconds. → Verify in Round 02.

- **H-004**: `other_interaction` is a distinct positive signal or noise? → Verify via cross-reference with `is_contact` flag.


## Code Reference

- Code: `src/eda/round_01_data_profiling.py`

- Modules used: `src/utils/profiler.py`, `src/utils/report_writer.py`


## Next Steps

Round 02: Schema Deep-Dive & Relationship Validation

- FK integrity check (item_id in facts vs dim_listing)

- User overlap between train events and test_users

- Verify H-001 through H-004
