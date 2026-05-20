# Round 13 Report: Cascade Configuration Optimization

## Executive Summary
This round evaluated different cascade orders and measured the standalone recall of each recommender source. The findings show that Pageview Replay is the single most powerful source (Recall 0.0618), followed by ALS (Recall 0.0412). The optimal cascade order combining these sources achieved a Recall@10 of 0.0844 on the raw ground truth data (before the strict `dim_listing` filtering was enforced). 

## Methodology
- Evaluated 5000 users.
- Computed the standalone Recall@10 for: Pageview(14d), Pageview(30d), CoContact, CoView, ALS, and RecentCC.
- Simulated different cascade permutations (e.g., PV → CoContact → ALS → RCC → SegPop).
- Analyzed the contribution of each source in the final top-10 recommendations for the best cascade.

## Raw Data Diagnostics

```
======================================================================
DIAGNOSTIC 4: Optimize cascade hyperparameters
======================================================================

[1] Building sources...
PageviewReplay: loaded 380,804 events (14d window)
PageviewReplay fitted: 3,058 users, avg 22.7 items/user
PageviewReplay: loaded 726,922 events (30d window)
PageviewReplay fitted: 3,478 users, avg 27.0 items/user
CoContact graph built: 210,304 items, 295,247 users used (window=30d)
Co-view graph: 214,465 items from 996,631 users
RecentCC built: 242 segments, 7d window, max 200/segment
RecentCC built: 257 segments, 14d window, max 200/segment
User histories built: 3,829 users
ALS loaded from outputs/models/als, matrix=(810411, 691579)
ALS recs for 5,000 users

[2] Per-source recall analysis...
  PV(14d) only:  Recall=0.0580, covers 3,058
  PV(30d) only:  Recall=0.0618, covers 3,478
  CoContact:     Recall=0.0262, covers 3,644
  Co-view:       Recall=0.0107, covers 3,266
  ALS:           Recall=0.0412, covers 5,000
  RecentCC(7d):  Recall=0.0172, covers 4,850

[3] Testing cascade combinations...
  PV14 → CoContact → RCC7 → SegPop                             | Recall@10: 0.0741
  PV30 → CoContact → RCC7 → SegPop                             | Recall@10: 0.0746
  PV14 → CoView → CoContact → RCC7 → SegPop                    | Recall@10: 0.0721
  PV14 → CoContact → ALS → RCC7 → SegPop                       | Recall@10: 0.0844
  PV14 → CoView → CoContact → ALS → RCC7 → SegPop              | Recall@10: 0.0807
  PV30 → CoView → CoContact → ALS → RCC7 → SegPop              | Recall@10: 0.0830
  PV30 → CoView → CoContact → ALS → RCC14 → SegPop             | Recall@10: 0.0830
  PV14 → CoContact → CoView → ALS → RCC7 → SegPop              | Recall@10: 0.0827

[4] Source contribution in best cascade...
  pv14    : 23,751 items placed,  936 hits
  cov     :  9,445 items placed,   73 hits
  cc      :  4,796 items placed,   45 hits
  als     : 12,008 items placed,  277 hits
  rcc7    :      0 items placed,    0 hits
  seg     :      0 items placed,    0 hits
```

## Key Findings
### Finding 1: Standalone Performance Rankings
- 📊 Data Evidence: PV(30d) provides the best standalone recall (0.0618), closely followed by PV(14d) (0.0580). ALS provides 0.0412, while CoView is extremely weak (0.0107).
- 🏠 Domain Explanation: Browsing behavior (Pageview) is the strongest signal of intent. Matrix factorization (ALS) provides a strong baseline, but it lacks the immediate recency of pageviews. CoView is surprisingly weak compared to CoContact, possibly due to noise in view data compared to high-intent contacts.
- 💡 Feature Idea: Drop CoView from the cascade. It adds computational overhead without significant recall improvement.

### Finding 2: The Optimal Cascade Configuration
- 📊 Data Evidence: `PV14 → CoContact → ALS → RCC7 → SegPop` yields a Recall@10 of 0.0844. Adding CoView actually reduces the recall slightly (to 0.0807), confirming its noisy nature.
- 🏠 Domain Explanation: The cascade works best when ordered by precision: High-intent (Pageview) → Similar intent (CoContact) → Collaborative signals (ALS) → Segment popular (RCC7) → Global fallback (SegPop).
- 💡 Feature Idea: Set the final cascade order exactly as: `PageviewReplay` → `CoContact` → `ALS` → `RecentCC` → `SegPop`. (Note: This was later amended in Round 15 to include `IntentRecommender` at Priority 1.5 due to item churn).

### Finding 3: Source Contribution Analysis
- 📊 Data Evidence: In the best cascade, Pageview provided 936 hits out of 23,751 items placed. ALS provided 277 hits out of 12,008 items placed.
- 🏠 Domain Explanation: The top slots in the cascade capture the vast majority of the true positives. Fallbacks like RecentCC and SegPop rarely provide a hit if the user is a "warm" user with enough PV/ALS candidates.

## Hypotheses Generated
- H-018: CoView adds more noise than signal in the top-10 candidate generation phase. → Status: VERIFIED

## Code Reference
- File: `src/eda/round_13_cascade_config.py`

## Next Steps
- Apply strict `dim_listing` active-item filtering to ensure Kaggle validity (which led to the discovery of item churn in Round 15).
