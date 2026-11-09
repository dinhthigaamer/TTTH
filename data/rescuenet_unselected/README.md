# Nguồn RescueNet chưa được chọn

`50x50.csv` và `100x100.csv` ghi chỉ mục ảnh/mask gốc chưa được đưa vào bộ instance tương ứng.
Không có instance hoặc cặp ngân sách dự phòng được sinh cho các ảnh này; dữ liệu gốc vẫn ở `data/RescueNet/`.

- `proven_incompatible`: có chứng minh không đáp ứng tiêu chí với cấu hình và seed bắt buộc.
- `not_certified`: tìm kiếm chưa chứng nhận được đủ 9 cấu hình; không kết luận ảnh bất khả thi.

Dùng các đường dẫn `image_path`, `mask_path` để xem lại nguồn; `reason_by_seed` ghi lý do theo seed.
Nhóm này không thuộc val/test thực nghiệm, không gộp vào bộ instance khi đánh giá thuật toán.
