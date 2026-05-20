# Round 18 Report: Independent Candidate Generator Evaluation

## Executive Summary
Đánh giá công bằng mức trần Recall@200 của TẤT CẢ các mô hình Candidate Generation đang có trong `src/models/candidates/` để dẹp bỏ bias.

## Data Evidence
```
2. LightALS              : 0.1749
3. IntentRecommender     : 0.1140
1. PageviewReplay        : 0.0913
4. UserKNN               : 0.0862
5. SellerExpansion       : 0.0302
6. SegmentPopularity     : 0.0079
```

## Domain Explanation & Next Steps
Kết quả này sẽ là kim chỉ nam tuyệt đối để sắp xếp Priority trong CascadeCandidateGenerator.
