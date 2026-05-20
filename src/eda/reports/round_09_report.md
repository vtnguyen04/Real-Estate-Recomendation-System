# Round 09 Report: Health Metrics Baseline & Ground-Truth Distribution

## Executive Summary
Quantified the **marketplace health** of our current submission across 4 axes:
Diversity, Fairness, Freshness, and Coverage. Discovered critical miscalibrations
that explain why pure Recall optimisation hurts the system's long-term health.

## Methodology
- Joined `submission.csv` with `dim_listing` for item metadata
- Computed natural distribution from training positive contacts (`lead_count > 0`)
- Compared submission distribution vs. ground-truth contact distribution
- Code: `src/eda/round_09_health_baseline.py`

---

## Key Findings

### Finding 1: Fairness — Agent/Private Seller Ratio is Badly Miscalibrated 🔴

| Source | Agent | Private |
|--------|-------|---------|
| Submission | **27.3%** | **72.7%** |
| GT contacts | 52.0% | 48.0% |
| Gap | −24.7 pp | +24.7 pp |

- **Problem**: We massively over-represent private sellers and under-represent agents.
- **Domain explanation**: Agents have far more listings (77% of dim_listing) and collectively
  generate 52% of all contacts. Our model is biased toward the high-lead-per-listing private sellers
  at the expense of coverage for the broader agent market.
- **Feature idea**: `seller_type_fairness_score` — penalise over-representation of private sellers
  in the top-K list using KL divergence from GT distribution.
- **Business impact**: Agents pay for premium placement; systematically under-serving them
  damages marketplace trust and revenue.

### Finding 2: Category Imbalance — Projects (1050) Over-Served 🟡

| Category | Submission | GT contacts | Gap |
|----------|-----------|------------|-----|
| 1010 (phòng trọ) | 11.3% | 15.6% | −4.3 pp |
| 1020 (căn hộ)   | 41.2% | 44.6% | −3.4 pp |
| 1030 (nhà ở)    | 8.7%  | 6.5%  | +2.3 pp |
| 1040 (đất nền)  | 9.8%  | 10.2% | −0.4 pp |
| 1050 (dự án)    | **29.0%** | 23.1% | **+5.9 pp** |

- **Problem**: Category 1050 (new projects) is over-recommended by 5.9 percentage points.
  Category 1010 (phòng trọ, the highest-volume rental segment) is under-served.
- **Feature idea**: `category_exposure_bonus` — boost categories below GT ratio in reranker.

### Finding 3: Freshness "Paradox" — Debunked by Survivorship Bias 🟡

| | Median Age | Mean Age | Freshness Score |
|--|-----------|---------|----------------|
| Submission | **10 days** | 36 days | **0.4443** |
| GT contacts | 97 days | 106 days | 0.0861 |

- **Initial misleading finding**: Users seem to contact listings that are 97 days old on average. It initially looked like our model over-prioritises brand-new listings.
- **True Explanation (Survivorship Bias)**: The high age of GT contacts is an illusion. Low-quality listings are removed early. Only the highest quality listings survive to 90+ days. In reality, **69.7% of contacts happen in the first 7 days**.
- **Recommendation**: Do NOT raise half-life to 30d. Keep half-life at 7d for the ALS signal to capture the "Golden Moment".
- **Feature idea**: `item_momentum_score` — items that maintain contacts over time are valuable, but freshness MUST remain the dominant signal for new listings.

### Finding 4: Coverage Extremely Low — Popularity Bias 🔴

| Metric | Value |
|--------|-------|
| Items recommended | 115,340 / 3,107,114 (**3.71%**) |
| Top-1% items share | **81.9%** of all recommendation slots |

- **Problem**: Massive popularity bias. 96.3% of the item catalogue never gets recommended.
  The platform's newer sellers and long-tail listings get zero exposure.
- **Business impact**: Hurts seller retention (new sellers never get traction), reduces
  marketplace diversity, creates a "rich-get-richer" feedback loop.
- **Recommendation**: Add `coverage_diversity_bonus` in reranker — boost items that have
  not been seen in the current recommendation list yet (long-tail exposure).

### Finding 5: Category Diversity Decent but Not Optimal 🟢

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Avg category entropy | **0.6947** | Good — most lists have 2+ categories |
| Avg distinct cities/user | **2.995** | Good — geographic spread |

- Users get recommendations across categories and cities, which is positive.
- Room for improvement: push entropy toward 0.8+ via stricter ILD enforcement.

---

## Ground-Truth Distribution (Saved to `.cache/gt_dist.json`)

```json
{
  "agent_ratio": 0.520,
  "category_dist": {
    "1010": 0.156,
    "1020": 0.446,
    "1030": 0.065,
    "1040": 0.102,
    "1050": 0.231
  }
}
```

This is now the calibration source for `HealthMetrics` (used by `MultiObjectiveReranker`).

---

## Hypotheses Generated

- **H-009**: Increasing fairness weight `γ` from 0.15 to 0.25 will reduce agent/private gap
  without significantly hurting Recall@10. → Status: PENDING (test in reranker ablation)
- **H-010**: Raising ALS half-life from 7d to 30d will better align with GT age distribution. → Status: REJECTED (Debunked by Survivorship Bias in PDF lifecycle analysis. Keep 7d).
- **H-011**: Adding a long-tail exposure bonus in trending will improve Coverage from 3.71%
  to >8% with <2% Recall drop. → Status: PENDING

---

## Next Steps

- **R10**: Sequential patterns — category transition analysis (apartment → house)
- **R13**: ALS half-life ablation study (3d vs 7d vs 14d vs 30d offline Recall@10)
- **R14**: Coverage & long-tail exposure improvement strategies

## Code Reference
- Code: `src/eda/round_09_health_baseline.py`
- Data output: `.cache/gt_dist.json`
