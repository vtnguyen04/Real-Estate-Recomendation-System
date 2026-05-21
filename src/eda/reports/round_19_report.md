# Round 19 Report: PCI (fact_post_contact_interactions) — Untapped Data Source

## Executive Summary
**fact_post_contact_interactions (PCI)** table contains **25.5M rows** of contact/lead data that our pipeline has NEVER used. Crucially, **10,654 currently "blind" test users** have rich PCI data (avg 16.3 items each) that can convert them from blind → warm. Additionally, **644,732 NEW (user,item) pairs** not in ALS training can be added to strengthen the model.

## Methodology
- Scanned all PCI parquet files (25.5M rows, 1.87M users, 574K items)
- Cross-referenced with test_users.parquet (161,568 users)
- Identified blind users (no contact in fact_user_events) who DO have PCI data
- Analyzed signal quality (lead_count, chat_messages, purchased)

## Key Findings

### Finding 1: PCI is a Massive Untapped Data Source
- 📊 **Data Evidence**: 25,486,445 rows, 1,872,512 users, 574,245 items, 4,067,917 total leads
- 📊 **Date range**: 2025-11-09 to 2026-04-09 (same as fact_user_events)
- 🏠 **Domain Explanation**: PCI aggregates daily contact/lead metrics per user-item pair. This is the same data as fact_user_events contacts but pre-aggregated and with additional fields (purchased, chat_turn_count)
- 💡 **Feature Idea**: Use PCI lead_count as supplementary ALS training signal
- 🎯 **Business Impact**: 37.3% of test users (60,212) have PCI data

### Finding 2: 10,654 Blind Users Have PCI Data (INS-059)
- 📊 **Data Evidence**: 
  - 107,066 blind test users (no fact_user_events contacts)
  - 10,654 of these have PCI data (9.9% of blind users)
  - 173,651 PCI rows for these users (avg 16.3 items/user)
  - 26,268 rows with lead_count > 0
  - 2,436 rows with purchased = True
- 🏠 **Domain Explanation**: These users interacted through the platform (viewed ads, submitted leads, chatted) but their events weren't captured in fact_user_events contact extraction. PCI is an independent data source with its own aggregation.
- 💡 **Feature Idea**: Build user preferences (city, category) from PCI data for these 10,654 users → use in IntentRecommender and SegPop
- 🎯 **Business Impact**: Converting 10,654 blind → warm reduces blind % from 66.3% to 59.7%

### Finding 3: 644K New Training Pairs for ALS (INS-060)
- 📊 **Data Evidence**:
  - PCI total lead pairs: 2,444,156
  - Already in ALS training: 1,799,424 (overlapping)
  - **NEW pairs from PCI: 644,732** (not in current ALS matrix)
  - New unique users: 237,086
- 🏠 **Domain Explanation**: PCI and fact_user_events capture contacts through different pipelines. PCI includes chat_lead and purchased signals that fact_user_events may not flag as contacts.
- 💡 **Feature Idea**: Merge PCI lead pairs into ALS training → denser matrix for existing users (more signal/user, INS-058 principle: density > size)
- 🎯 **Business Impact**: ALS training data grows from 13M → 13.6M pairs (+5%) while maintaining login-quality signal

### Finding 4: Category Distribution of Blind PCI Users
- 📊 **Data Evidence**:
  - 1020 (Căn hộ/CC): 84,155 rows (48.5%)
  - 1050 (Dự án): 32,859 rows (18.9%)
  - 1010 (Phòng trọ): 27,398 rows (15.8%)
  - 1040 (Đất nền): 17,460 rows (10.1%)
  - 1030 (Nhà ở): 11,779 rows (6.8%)
- 🏠 **Domain Explanation**: Căn hộ dominates — these are likely users browsing apartment projects through ads/campaigns who didn't login but submitted lead forms. High-intent commercial signals.

## Hypotheses Generated
- H-020: Adding PCI lead pairs to ALS training will improve Recall@10 by 2-5% → Status: PENDING
- H-021: Building preferences from PCI for 10,654 blind users will improve SegPop matching → Status: PENDING
- H-022: PCI `purchased=True` items are strongest signal — weight 2x in ALS → Status: PENDING

## Code Reference
- Script: `src/eda/round_19_pci_untapped.py`
- Report: `src/eda/reports/round_19_report.md`

## Next Steps
1. **Merge PCI lead pairs** into `.cache/als_contact_pairs.parquet` (login users only, per INS-057)
2. **Extract preferences** from PCI for 10,654 blind users → add to cold_user_prefs
3. **Retrain ALS** on enriched matrix (13.6M pairs)
4. **Offline eval** before any submission
