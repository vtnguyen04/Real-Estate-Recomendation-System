# Round 23 Report: Cold-Start Recall Ceiling Analysis

## Executive Summary
SegPop popularity-based cold-start has a **theoretical ceiling of ~1.6% Recall@10**
for blind users, even with PERFECT city+category knowledge. BĐS items are too sparse
for top-K popularity to cover GT contacts. 44% of blind user contacts are on items
posted ≤7 days — freshness matters more than historical popularity.

## Methodology
- Time-split: last 3 days of training = validation period
- Blind val users: 13,460 (contacted in val, never in train)
- Warm val users: 44,447
- Built (city, category) segment popularity from training period
- Tested recall@K at K=10,20,50,100,200,500
- Computed theoretical max: gave each user top-K from their TRUE (city, cat)

## Key Findings

### Finding 1: SegPop Ceiling is ~1.6%
- 📊 Even with PERFECT city+cat knowledge, Recall@10 = 0.0158
- 🏠 BĐS has 28,732 unique items contacted by 13,460 blind users in 3 days.
  Each (city, cat) segment has thousands of items but top-10 only covers tiny fraction.
- 💡 Popularity alone cannot solve cold-start in BĐS.

### Finding 2: Hit Rates at Various K
| K | Hit Rate |
|---|---------|
| 10 | 1.22% |
| 20 | 2.18% |
| 50 | 4.10% |
| 100 | 6.24% |
| 200 | 9.02% |
| 500 | 14.22% |

- 📊 Even at K=500, only 16.3% hit rate. Long-tail distribution.

### Finding 3: Blind User Contact Geography
| City | Contacts | % |
|------|----------|---|
| Tp Hồ Chí Minh | 39,167 | 73.8% |
| Đà Nẵng | 3,428 | 6.5% |
| Hà Nội | 3,404 | 6.4% |
| Bình Dương | 2,185 | 4.1% |
| Long An | 751 | 1.4% |
| Đồng Nai | 744 | 1.4% |
| Cần Thơ | 607 | 1.1% |
| Bà Rịa - Vũng Tàu | 394 | 0.7% |
| Lâm Đồng | 315 | 0.6% |
| Khánh Hòa | 282 | 0.5% |

- 📊 HCM dominates at ~74% of blind user contacts.

### Finding 4: Blind User Category Distribution
| Category | Contacts | % |
|----------|----------|---|
| 1050 | 21,014 | 39.6% |
| 1020 | 16,195 | 30.5% |
| 1010 | 8,431 | 15.9% |
| 1040 | 4,144 | 7.8% |
| 1030 | 3,305 | 6.2% |

- 📊 1050 (Dự án) is #1 for blind users (39.6%), unlike warm users where 1020 dominates.

### Finding 5: Item Freshness Is Critical
| Posted Within (days) | % of Contacts |
|---------------------|---------------|
| 1 | 11.2% |
| 3 | 27.5% |
| 7 | 43.9% |
| 14 | 59.1% |
| 30 | 75.1% |
| 60 | 85.9% |

- 📊 44% of blind contacts are on items ≤7 days old. Recency > historical popularity.
- 🏠 New listings get contacts quickly. Old popular items are stale.

### Finding 6: Score Decomposition
- Warm users: 52,329 (32.4%)
- Truly blind: 94,896 (58.7%)
- Current score: 0.034 = warm × 0.324 × ~0.10 recall
- To reach 0.10: need cold_recall = 0.100
- To reach 0.32: need cold_recall = 0.425

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
