# Round 14 Report: Intent Matching Basic (Ward + Price)

## Executive Summary
This round attempted a highly aggressive, ultra-granular intent matching strategy by extracting the user's preferred Ward, Category, and Price Bucket. The goal was to see if matching at the Ward-level (Phường/Xã) would yield high precision. The results were completely negative: only 0.1% of Ground Truth items matched this strict 3-way intent.

## Methodology
- Sampled 3000 users and extracted their Ward, Price Bucket, and Category from historical pageviews.
- Counted how many of the 48,842 Ground Truth interactions exactly matched this 3-way tuple.

## Raw Data Diagnostics

```
======================================================================
DIAGNOSTIC 5: Validate Intent (Ward + Price + Category) matching
======================================================================

[1] Loading user history and ground truth...
  Users with profiles: 1123 / 3000

[2] Checking exact 3-way intent match...
  GT items matching exact (Ward, Price, Cat) intent: 57 / 48842 (0.1%)
```

## Key Findings
### Finding 1: Ward-Level Matching is Too Strict
- 📊 Data Evidence: Users with extractable profiles: 1123 / 3000. GT items matching exact (Ward, Price, Cat) intent: 57 / 48,842 (0.1%).
- 🏠 Domain Explanation: While users have a preferred District (Quận/Huyện), they are extremely flexible across Wards (Phường/Xã) within that District or adjacent Districts. Real estate supply at the Ward level is too sparse. Furthermore, price ranges are fluid; a user looking at 3B might stretch to 4B if the property is good.
- 💡 Feature Idea: Abandon Ward-level intent matching. Elevate intent matching to the District or City level, and use a broader Price Bucket (or relax price matching entirely if supply is low).
- 🎯 Business Impact: Prevent the recommendation system from returning zero results due to over-filtering, a common pitfall in strict Content-Based systems.

## Hypotheses Generated
- H-017: Intent matching at the (District, Category, Price) level will have orders of magnitude better coverage than the Ward level. → Status: VERIFIED (via Diagnostic 6 / Round 15)

## Code Reference
- File: `src/eda/round_14_intent_matching_basic.py`

## Next Steps
- Implement District-level Intent Matching (already done in Round 15).
