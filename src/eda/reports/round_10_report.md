# Round 10 Report: Recall Optimization Strategies

## Executive Summary
This round explored various strategies to achieve a high Recall@10 (target: 0.3). The analysis reveals that simple global or category-based popularity (SegPop) yields a Recall@10 of ~0.0226, while filtering recent contacts within a user's category/city segment (Recent CC 7d) boosts it to 0.0522. The theoretical ceiling for a purely City-Category matched strategy is 0.0848, indicating that more granular user intent matching is required to break the 0.1 barrier.

## Methodology
- Extracted 17,666 Ground Truth (GT) items from the validation set (5000 users).
- Verified that 93.9% of these GT items exist in the active `dim_listing`.
- Simulated different candidate generation strategies (SegPop, Recent CC, Fresh Posted, Hybrid, History Replay).
- Calculated the theoretical upper bounds (ceilings) for Recall@10 if we perfectly guessed the correct items within a user's preferred City or City+Category.

## Raw Data Diagnostics

```
======================================================================
DIAGNOSTIC: What would it take to reach 0.3 Recall@10?
======================================================================

[1] Loading data...
  Val users: 5000, with prefs: 4850

[2] Analyzing GT items...
  Total GT items: 17666
  GT items in dim_listing: 16587 (93.9%)
  GT posted_date range: 2024-09-24 → 2026-04-09
  GT item age (days from split): median=7, mean=30, p90=80

[3] Analyzing recent contacts...
  Items contacted in last  7d: 107,635 | GT overlap: 12,615 (71.4%)
  Items contacted in last 14d: 149,543 | GT overlap: 13,301 (75.3%)
  Items contacted in last 30d: 218,009 | GT overlap: 13,707 (77.6%)
  Items contacted in last 60d: 283,567 | GT overlap: 13,761 (77.9%)
  Items contacted in last 90d: 389,575 | GT overlap: 13,768 (77.9%)

[4] Testing candidate strategies...

--- Strategy A: SegPop CC (all-time contacts, top-K per segment) ---
  SegPop CC top-10                                   | Recall@10: 0.0226 | Unique items: 1,187
  SegPop CC top-50                                   | Recall@10: 0.0226 | Unique items: 5,573
  SegPop CC top-100                                  | Recall@10: 0.0226 | Unique items: 10,371

--- Strategy B: Recent contacts in user's CC segment ---
  Recent CC (7d), top-10                             | Recall@10: 0.0522 | Unique items: 1,123
  Recent CC (14d), top-10                            | Recall@10: 0.0442 | Unique items: 1,151
  Recent CC (30d), top-10                            | Recall@10: 0.0382 | Unique items: 1,172
  Recent CC (60d), top-10                            | Recall@10: 0.0354 | Unique items: 1,177

--- Strategy C: Freshly POSTED items in user's CC segment ---
  Fresh posted (7d), top-10                          | Recall@10: 0.0184 | Unique items: 924
  Fresh posted (14d), top-10                         | Recall@10: 0.0202 | Unique items: 1,002
  Fresh posted (30d), top-10                         | Recall@10: 0.0208 | Unique items: 1,098

--- Strategy D: Hybrid (recent contacts + fresh posted) ---
  Hybrid: recent-contact(30d) + fresh-posted(30d)    | Recall@10: 0.0382 | Unique items: 1,176

--- Strategy E: User's own history replay + segment ---
  History replay(5) + SegPop CC(5)                   | Recall@10: 0.0176 | Unique items: 16,177

--- Strategy F: District-level SegPop ---
  CCD cascade (dist→cc)                              | Recall@10: 0.0418 | Unique items: 3,860

--- Strategy G: Theoretical ceilings ---
  CEILING: GT items matching user CC                 | Recall@10: 0.0848 | Unique items: 734
  CEILING: GT items matching user city               | Recall@10: 0.0418 | Unique items: 302
```

## Key Findings
### Finding 1: Recent Contacts Outperform Pure Popularity
- 📊 Data Evidence: SegPop top-10 yields 0.0226 Recall. Recent CC (7d) yields 0.0522 Recall.
- 🏠 Domain Explanation: BĐS listings that have been recently contacted are highly relevant and likely still available, making them better candidates than all-time popular listings.
- 💡 Feature Idea: Prioritize items that received contacts in the last 7 days within the user's preferred City+Category.
- 🎯 Business Impact: Improves cold-start recommendations by showing "trending right now" items instead of historically popular ones.

### Finding 2: The Theoretical Ceiling of City+Category is 0.0848
- 📊 Data Evidence: Even if we perfectly select the top 10 correct items from the user's preferred City+Category segment, the maximum possible Recall@10 is 0.0848.
- 🏠 Domain Explanation: City+Category is too broad (e.g., "Căn hộ HCM"). Users search in specific districts and price ranges.
- 💡 Feature Idea: We must use more granular segments: City+Category+District+Price (Intent Matching) to narrow down the candidate pool and improve precision.

## Hypotheses Generated
- H-014: Adding District and Price Bucket to the Segment Popularity will break the 0.0848 ceiling. → Status: PENDING

## Code Reference
- File: `src/eda/round_10_recall_strategies.py`

## Next Steps
- Analyze test users' pageview behavior to extract granular intents.
