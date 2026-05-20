# EDA Synthesis Report — Feature Blueprint & Strategy Recommendations

## Phase 1-2 EDA Summary (Rounds 01-07)

### Data Landscape
| Table | Rows | Description |
|-------|------|-------------|
| dim_listing | 3,107,114 | Listing attributes (24 cols) |
| fact_listing_snapshot | 19,762,167 | Daily views/contacts metrics (5 cols) |
| fact_post_contact_interactions | 25,486,445 | Post-contact engagement (10 cols) |
| fact_user_events | 161,731,336 | User clickstream (16 cols) |
| test_users | 161,568 | Users to predict for (1 col) |

### Critical Findings (Impact on Strategy)

| # | Finding | Evidence | Strategy Impact |
|---|---------|----------|-----------------|
| 🔴1 | **64% test users = cold-start** | 103,415/161,568 no training history | Cold-start fallback is #1 priority |
| 🔴2 | **other_interaction IS positive** | 100% have is_contact=1 | MUST include in positive_events |
| 🔴3 | **dwell_time_sec = milliseconds** | Median 17,915 → 17.9s corrected | All thresholds must ÷1000 |
| 🔴4 | **HCM + HN = 81% of events** | Geographic concentration | Cold-start by city essential |
| 🟡5 | **Listing freshness decays** | Views 2x higher at age 0-6 vs 14+ days | Freshness as ranking feature |
| 🟡6 | **Zombie listings = 8.4%** | 14,566 items, age>60d, 0 contacts | Exclude from candidates |
| 🟡7 | **Bot traffic minimal** | Only 2 users with score≥4 | Data is already clean |
| 🟡8 | **purchased=True = 2.3%** | Higher engagement metrics | Soft label, don't use as ground truth |
| 🟡9 | **Power law in users & items** | Median 21 events/user, P99=1271 | Feature normalization needed |
| 🟡10 | **Tết effect -40% traffic** | 654K vs 1.1M daily avg | Time-weight recent data higher |

---

## Feature Blueprint

### User Features (computed from fact_user_events)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| user_event_count | count events per user | Activity level |
| user_contact_rate | is_contact.sum / total events | Engagement propensity |
| user_category_preference | mode(category) | Primary interest |
| user_city_preference | mode(city_name) | Geographic preference |
| user_device_primary | mode(device) | Platform preference |
| user_avg_dwell_sec | avg(dwell_time_sec / 1000) | Browse depth |
| user_session_count | nunique(session_id) | Visit frequency |
| user_items_per_session | n_items / n_sessions | Browse breadth |
| user_recency_days | (cutoff_date - max_event_date).days | Recency signal |
| user_is_cold_start | user NOT in train events | Cold-start flag |

### Item Features (computed from dim_listing + fact_listing_snapshot)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| item_category | category (1010-1050) | Category matching |
| item_city | city_name | Geographic matching |
| item_seller_type | seller_type (agent/private) | Fairness feature |
| item_ad_type | ad_type (sell/let) | Intent matching |
| item_images_count | images_count | Quality proxy (R05: more images → more leads) |
| item_completeness | count non-null of 8 attribute fields | Quality score (R05) |
| item_avg_views_24h | mean(views_24h) from snapshot | Popularity signal |
| item_avg_contacts_24h | mean(contacts_24h) from snapshot | Demand signal |
| item_listing_age | max(listing_age_days) | Freshness decay (R05) |
| item_is_zombie | age>60 & views<5 & contacts=0 | Exclusion flag (R06) |
| item_price_bucket | price_bucket | Price matching |

### Interaction Features (computed from user-item pairs)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| user_item_category_match | user_pref_cat == item_cat | Preference alignment |
| user_item_city_match | user_pref_city == item_city | Geographic alignment |
| user_item_pageview_count | count pageviews for (user, item) | Interest strength |
| user_item_dwell_total | sum dwell for (user, item) | Engagement depth |
| user_item_recency | days since last interaction | Temporal relevance |

---

## Strategy Implementation Plan

### Stage 0: Data Cleaning
1. ✅ Bot removal (negligible — only 2 users)
2. ✅ Zombie listing exclusion (14,566 items, 8.4%)
3. ⚠️ Fix positive_events config (ADD `other_interaction`)
4. ⚠️ Fix dwell_time units (÷1000 everywhere)

### Stage 1: Candidate Generation
1. **Warm users (36%)**: ALS collaborative filtering on user-item positive interactions
2. **Cold users (64%)**: Popularity-based fallback by city + category
3. **Item-to-item**: Content similarity (category, city, price_bucket, seller_type)

### Stage 2: Feature Engineering
1. Compute all user/item/interaction features above
2. Join with dim_listing for item attributes
3. Pre-aggregate fact_user_events by user and by item (avoid OOM)

### Stage 3: Ranking
1. LightGBM LambdaRank on feature matrix
2. Target: is_contact (binary) or multi-class contact type
3. Validation: time-based split (last 3 days of training)

### Stage 4: Re-ranking
1. MMR for diversity (category, seller_type, city diversity in top-10)
2. Freshness boost (penalize listing_age > 30 days)
3. Fairness: ensure private sellers get minimum exposure

---

## Strategies Coverage Checklist

| Strategy Task | Status | Round |
|---------------|--------|-------|
| Task 1.1 Bot Detection | ✅ Done | R06 |
| Task 1.2 Zombie Listings | ✅ Done | R06 |
| Task 1.3 purchased Reverse Engineering | ✅ Done | R07 |
| Task 1.4 Price Anchoring | ✅ Done | R07 |
| Task 1.5 User Segmentation/Archetypes | ⚠️ Partial (contact funnel in R04) | R04 |
| Task 1.6 Sequential Patterns | ⚠️ Not separately done — session analysis in R04 covers basics | R04 |
| Temporal Analysis | ✅ Done | R03, R07 |
| Geographic Analysis | ✅ Done | R03, R07 |
| Listing Quality Analysis | ✅ Done | R05 |
| Schema & FK Integrity | ✅ Done | R02 |
| Missing Value Patterns | ✅ Done | R01 |

---

## All Reports & Figures

| Round | Report | Figures |
|-------|--------|---------|
| 01 | `reports/round_01_report.md` | 0 |
| 02 | `reports/round_02_report.md` | 0 |
| 03 | `reports/round_03_report.md` | 10 |
| 04 | `reports/round_04_report.md` | 5 |
| 05 | `reports/round_05_report.md` | 5 |
| 06 | `reports/round_06_report.md` | 3 |
| 07 | `reports/round_07_report.md` | 4 |
| **Total** | **7 reports** | **27 figures** |

## All Modules Created
- `src/utils/profiler.py` — Data scanning & profiling
- `src/utils/report_writer.py` — Markdown report generation
- `src/utils/plotting.py` — Reusable charting
