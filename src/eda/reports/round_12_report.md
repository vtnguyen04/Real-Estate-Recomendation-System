# Round 12 Report: Pageview Window Optimization

## Executive Summary
This round focused on optimizing the look-back window for Pageview Replay. We found a clear trade-off between coverage (how many users have pageviews) and precision (how relevant those pageviews are). The 7-day window strikes the optimal balance, pushing the overall cascade Recall@10 to a peak of 0.2480 when combined with Co-Contact and Recent CC fallbacks.

## Methodology
- Evaluated 4 different pageview look-back windows: 3 days, 7 days, 14 days, and 30 days.
- Measured pure Pageview Replay Recall@10 and user coverage for each window.
- Measured the performance of the full cascade (Pageview → CoContact → RecentCC) for each window size to see the holistic impact.

## Raw Data Diagnostics

```
======================================================================
DIAGNOSTIC 3: Push pageview strategy to the limit
======================================================================

[1] Pageview window analysis...
  PV replay (3d, by count+recency) [1082/3000]            | Recall@10: 0.1387
  PV replay (7d, by count+recency) [1462/3000]            | Recall@10: 0.1813
  PV replay (14d, by count+recency) [1735/3000]           | Recall@10: 0.1970
  PV replay (30d, by count+recency) [1977/3000]           | Recall@10: 0.1977

[2] Include contact events in replay...
  All events weighted (7d) [1562/3000]                    | Recall@10: 0.1893
  All events weighted (14d) [1861/3000]                   | Recall@10: 0.2007
  All events weighted (30d) [2103/3000]                   | Recall@10: 0.2007

[3] PV replay + fallback for users without pageviews...
  Users with PV: 1562, CoContact: 1323, CC: 733
  FULL: PV(7d) → CoContact → RecentCC(7d)                 | Recall@10: 0.2480

[4] Longer PV + same fallback...
  FULL w/ PV(14d) → CoContact → RecentCC [1861]           | Recall@10: 0.2427
  FULL w/ PV(30d) → CoContact → RecentCC [2103]           | Recall@10: 0.2350
```

## Key Findings
### Finding 1: The "Golden Window" for Pageviews is 7 Days
- 📊 Data Evidence: Pure pageview replay reaches a peak of 0.1977 Recall at 30 days, but the cascade score peaks at 0.2480 with the 7-day window.
- 🏠 Domain Explanation: While a 30-day window provides the highest coverage (1977/3000 users), the pageviews from 3-4 weeks ago are stale. In real estate, after 30 days, users have either rented/bought or the item is no longer available. The 7-day window captures active, high-intent browsing.
- 💡 Feature Idea: Use a 14-day window for loading events, but aggressively decay the weight of older pageviews.

### Finding 2: The Cascade Synergy Breaks 0.24
- 📊 Data Evidence: When assembling the full cascade using the 7-day PV window: `PV(7d) → CoContact → RecentCC(7d)`, the Recall@10 skyrockets to 0.2480. Using a 30-day PV window actually drops the cascade score to 0.2350.
- 🏠 Domain Explanation: When we rely too heavily on stale 30-day pageviews, we push out the highly relevant (but non-viewed) fallback items from RecentCC.
- 💡 Feature Idea: Cap the number of pageview candidates (e.g., max 3 or 4) so that CoContact and RecentCC have room to contribute to the top 10.

## Hypotheses Generated
- H-016: Adding a Co-view graph (users who viewed X also viewed Y) will improve cold-start coverage further than Co-Contact, since views are 100x more frequent than contacts. → Status: PENDING

## Code Reference
- File: `src/eda/round_12_pageview_optimization.py`

## Next Steps
- Analyze cascade hyperparameter configurations (how many items to allocate to each source).
