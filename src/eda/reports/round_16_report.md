# Round 16 Report: Adview Count Correlation

## Executive Summary
Phân tích sự tương quan giữa số lượt xem tin (`views_24h`) và lượt liên hệ (`contacts_24h`) để tìm ra các tín hiệu phi tuyến tính (non-linear) dùng cho mô hình Reranker. Phát hiện ra rằng conversion rate có dạng hình chữ U: cao nhất ở các tin ít view và tin cực nhiều view.

## Methodology
- **Data**: `fact_listing_snapshot`
- **Approach**: Gom nhóm các item theo `views_24h`, tính tổng `contacts_24h` và tính tỷ lệ chuyển đổi (`Conversion Rate`) cho từng nhóm view. Đánh giá hệ số tương quan Pearson.

## Key Findings
### Finding 1: Non-linear Correlation between Views and Contacts
- 📊 **Data Evidence**: Pearson Correlation = 0.7571. Conversion rate: 0 views = 10.3%, 30 views = 8.7%, 150+ views = 10.1%.
- 🏠 **Domain Explanation**: Tin ít view nhưng ra số nhanh là "hàng ngộp", "giá sập sàn" nên bị chốt ngay. Tin view trung bình là nhà phổ thông, user ngâm cứu lâu. Tin view khủng (150+) thường là dự án hot, tạo hiệu ứng FOMO (sợ bỏ lỡ) thúc đẩy tỷ lệ liên hệ.
- 💡 **Feature Idea**: Thêm `views_24h` và `contact_conversion_rate = contacts_24h / (views_24h + 1)` vào thuật toán LightGBM.
- 🎯 **Business Impact**: Giúp mô hình Reranker nhận diện chính xác các "hàng ngộp" để đẩy lên Top 1, thỏa mãn nhu cầu săn sale của user.

## Hypotheses Generated
- H-016: Mô hình LightGBM khi thêm feature `contact_conversion_rate` sẽ tăng NDCG@10 lên ít nhất 5% so với mô hình không có feature này. → Status: [PENDING]

## Code Reference
- File: `src/eda/round_16_adview_correlation.py`
- Figures: `reports/figures/round_16_adview_correlation.png` (N/A)

## Next Steps
Tiếp tục phân tích hành vi người dùng: Sự dịch chuyển giữa các danh mục (Category Transitions) ở Round 17.
