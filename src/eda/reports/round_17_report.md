# Round 17 Report: Sequential Category Transitions

## Executive Summary
Phân tích tỷ lệ người dùng giữ nguyên nhu cầu (cùng Category) trong các lần liên hệ liên tiếp. Phát hiện độ "chung tình" của người dùng BĐS rất cao, lên tới 75.11%, đặc biệt cao ở phân khúc Dự Án (1050) với 87.2%.

## Methodology
- **Data**: `fact_post_contact_interactions`
- **Approach**: Sort các events của từng user theo thời gian, tính sự dịch chuyển (transition probability) từ `prev_category` sang `category` hiện tại.

## Key Findings
### Finding 1: The "Sticky" Category Phenomenon
- 📊 **Data Evidence**: 1010 -> 1010 = 71.3%; 1020 -> 1020 = 76.5%; 1050 -> 1050 = 87.2%. Tỷ lệ không đổi Category trung bình = 75.11%.
- 🏠 **Domain Explanation**: Người mua/thuê BĐS có mục tiêu rất rõ ràng (VD: đang tìm phòng trọ thì không rảnh đi xem dự án cao cấp). Phân khúc Dự Án (1050) có độ dính cực cao vì đây là giới đầu tư có dòng tiền mạnh, khác biệt hoàn toàn với khách hàng mua ở thực (1030).
- 💡 **Feature Idea**: Bổ sung `is_same_category_as_last_view` (Binary flag) vào LightGBM để làm tín hiệu chặn các gợi ý lạc quẻ.
- 🎯 **Business Impact**: Giảm mạnh tỷ lệ False Positive, tiết kiệm số lần hiển thị (impression) cho các tin không phù hợp, từ đó tăng độ chính xác của top 10 gợi ý cuối cùng.

## Hypotheses Generated
- H-017: Nếu áp dụng hình phạt (penalty) cực lớn cho các candidate sai Category so với lượt liên hệ gần nhất, Recall@10 sẽ tăng do nhường chỗ cho các tin đúng nhu cầu. → Status: [PENDING]

## Code Reference
- File: `src/eda/round_17_sequential_transitions.py`
- Figures: `reports/figures/round_17_sequential_transitions.png` (N/A)

## Next Steps
Tiếp tục tổng hợp tất cả các insights để viết file cấu trúc Feature Engineering chuẩn bị cho việc training mô hình LightGBM Reranker.
