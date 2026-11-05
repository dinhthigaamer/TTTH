# Mô hình hoá bài toán Deadline-constrained Multi-UAV Risk Coverage

Tài liệu mô hình hoá cho đề tài: tối đa hoá mức rủi ro được bao phủ trước deadline, dưới ràng buộc pin của nhiều UAV. Phiên bản này theo hướng đã thống nhất: độ cao bay của mỗi UAV được thiết lập riêng và không đưa vào mô hình, và mỗi lần quan sát nhìn được đúng một ô.

## 1. Bối cảnh và quy trình

Sau thảm họa (lũ, động đất, bão), cần dùng nhiều UAV để khảo sát vùng bị ảnh hưởng. Do thời gian và pin có hạn, UAV không thể phủ toàn bộ khu vực, nên phải ưu tiên những vùng có rủi ro cao. Quy trình tổng thể:

1. Ảnh chụp khu vực từ vệ tinh.
2. Phân rã khu vực thành lưới các ô vuông.
3. Đánh dấu mức rủi ro cho từng ô (từ phân đoạn ảnh RescueNet/C2A hoặc dữ liệu giả lập).
4. Thiết kế lộ trình cho từng UAV để tối đa tổng rủi ro được phủ trước deadline.

Tài liệu này mô hình hoá bước 2 đến 4 thành một bài toán tối ưu tổ hợp. Khi mỗi quan sát chỉ nhìn một ô, bài toán là một dạng Team Orienteering trên lưới, với ngân sách thời gian và pin riêng cho từng UAV. Đây cũng là dạng bài toán của Song et al. (ICRA 2022, bài MAEl-CPP), nên có thể dùng bài đó làm mốc cho baseline.

## 2. Bảng ký hiệu

| **Ký hiệu**           | **Ý nghĩa**                                              |
| --------------------- | -------------------------------------------------------- |
| `m`, `n`              | Số hàng, số cột của lưới                                 |
| u                     | Cạnh mỗi ô vuông (mét)                                   |
| C = {(i, j)}          | Tập ô; mỗi ô cũng là một điểm quan sát                   |
| w_c ≥ 0               | Mức rủi ro của ô `c`                                     |
| K                     | Số UAV; `k = 1..K`                                       |
| s_k ∈ C               | Ô chứa depot (vị trí xuất phát và quay về) của UAV `k`   |
| D                     | Deadline chung                                           |
| B_k                   | Dung lượng pin của UAV `k`                               |
| η ∈ (0, 1]            | Hệ số an toàn pin                                        |
| v_h                   | Tốc độ bay ngang                                         |
| e_h                   | Năng lượng trên mét bay ngang                            |
| τ                     | Thời gian chụp và tính toán tại mỗi ô                    |
| ε                     | Năng lượng chụp tại mỗi ô                                |
| R_k = (c_1, ..., c_L) | Lộ trình của UAV `k`: dãy các ô được quan sát, `L = L_k` |

Ghi chú: các hằng số `v_h`, `e_h`, `τ`, `ε`, `B_k` có thể khác nhau giữa các UAV. Để gọn, các chỉ số `k` được bỏ khi không gây nhầm lẫn. Lưới (`m`, `n`, `u`) và mảng `w_c` dùng chung cho mọi UAV.

## 3. Mô hình môi trường

### 3.1 Lưới và mức rủi ro

Khu vực được chia thành lưới `m × n` ô vuông cạnh `u`. Mỗi ô `c` có trọng số rủi ro `w_c ≥ 0`. Với dữ liệu phân đoạn (mask lớp), mức rủi ro của ô là trung bình rủi ro theo lớp của các điểm ảnh trong ô:

```
w_c = (1 / |A_c|) · Σ_{x ∈ A_c} ρ(label(x))
```

trong đó `A_c` là tập điểm ảnh thuộc ô `c`, `ρ` là bảng ánh xạ từ lớp (ví dụ nhà sập hoàn toàn, đường bị chặn, ngập nước) sang giá trị rủi ro trong `[0, 1]`. Bảng `ρ` là giả định thiết kế, cần được nêu rõ, dùng chung cho cả nhóm và kiểm tra theo tài liệu của dataset.

Vì lưới được chia một lần và dùng chung, nếu các ô có cùng số điểm ảnh thì dùng trung bình hay tổng chỉ khác nhau một hệ số hằng, và tỉ lệ rủi ro được phủ (mục 10) không đổi.

### 3.2 Điểm quan sát và vùng phủ

Giả định chính: một lần quan sát tại ô `(i, j)` chỉ nhìn được đúng ô `(i, j)`. Do đó tập điểm quan sát ứng viên chính là `P = C`, và vùng phủ là `cov(i, j) = {(i, j)}`.

Ý nghĩa của giả định này:

- Mỗi ô là đơn vị quan sát; mỗi UAV được cấu hình (độ cao, camera) sao cho vùng nhìn thấy vừa một ô. Độ cao vì vậy **không làm thay đổi vùng phủ** và không được đưa vào mô hình.
- Không có chồng lấn giữa các vùng phủ, nên mỗi ô chỉ mang giá trị `w_c` một lần khi được ghé.
- Giả định này khớp với bài báo Song et al., giúp cài lại baseline dễ dàng.

## 4. Mô hình UAV

Mỗi UAV `k` xuất phát từ depot `s_k`, thực hiện các quan sát rồi quay về depot. Độ cao bay của mỗi UAV được thiết lập riêng từ trước và không đưa vào mô hình: độ cao không ảnh hưởng đến vùng phủ, thời gian hay năng lượng, nên mô hình chỉ xét chuyển động ngang trên mặt phẳng lưới.

Với hai ô `a = (i, j)` và `b = (i', j')`, khoảng cách ngang giữa tâm hai ô (bay thẳng) là:

```
d(a, b) = u · √((i − i')² + (j − j')²)
```

Thời gian và năng lượng bay ngang:

- Thời gian bay:

  ```text
  t(a, b) = d(a, b) / v_h
  ```

- Năng lượng bay:

  ```text
  e(a, b) = e_h · d(a, b)
  ```

Với lộ trình `R_k = (c_1, ..., c_L)`, đặt `c_0 = s_k`:

- Thời điểm đến ô thứ `l`: `T_l = Σ_{q=1..l} t(c_{q−1}, c_q) + (l − 1) · τ`
- Tổng năng lượng: `E(R_k) = Σ_{q=1..L} e(c_{q−1}, c_q) + L · ε + e(c_L, s_k)`

Năng lượng bao gồm cả chặng bay về depot để UAV không hết pin giữa đường. Với lộ trình rỗng (`L = 0`) UAV không bay và không tốn gì.

## 5. Phát biểu bài toán tối ưu

**Biến quyết định:** các lộ trình `R_1, ..., R_K`, mỗi lộ trình là một dãy ô.

**Hàm mục tiêu:** tổng rủi ro của các ô được ghé bởi ít nhất một UAV. Mỗi ô chỉ được tính một lần dù bị ghé nhiều lần:

```
U = ∪_{k=1..K} {c : c thuộc R_k}
f(R_1, ..., R_K) = Σ_{c ∈ U} w_c
```

**Bài toán:**

```
max f(R_1, ..., R_K)
```

theo các ràng buộc:

- (C1) Deadline: `T_L(R_k) ≤ D` với mọi `k` (UAV đến ô quan sát cuối trước thời điểm `D`).
- (C2) Pin: `E(R_k) ≤ η · B_k` với mọi `k`.
- (C3) Mỗi lộ trình là dãy ô hợp lệ trong `C`, không lặp ô trong một lộ trình.

Ghi chú: nếu yêu cầu tính cả thời gian chụp tại điểm cuối thì thay `T_L` bằng `T_L + τ` trong (C1).

Hai UAV không bắt buộc phải ghé các ô khác nhau, vì mục tiêu dùng hợp tập nên ô bị ghé hai lần không mang thêm giá trị. Vì khoảng cách là Euclid (thoả bất đẳng thức tam giác), bỏ một lần ghé trùng không làm tăng thời gian hay năng lượng của lộ trình, nên luôn tồn tại nghiệm tối ưu mà các lộ trình rời nhau về ô. Do đó có thể thêm ràng buộc rời nhau mà không đổi giá trị tối ưu, và cách thêm này tiện cho thuật toán và MILP.

## 6. Tính chất của bài toán

### 6.1 Cấu trúc

Vì mỗi ô chỉ được tính một lần và vùng phủ không chồng lấn, hàm mục tiêu là **cộng tính (modular)**: giá trị của một tập ô chính là tổng `w_c`. Bài toán là **Team Orienteering** trên lưới với `K` UAV, mỗi UAV có hai ràng buộc ngân sách (thời gian và năng lượng) có dạng tuyến tính theo độ dài đường bay và số ô được chụp. Chi phí chèn một ô vào lộ trình là một cặp (thời gian thêm, năng lượng thêm), và cả hai đều phải thoả.

### 6.2 Độ khó

Với `K = 1` và ngân sách đủ lớn, bài toán chứa Orienteering là trường hợp riêng, vốn là bài toán NP-khó. Do đó cần heuristic hoặc giải chính xác trên instance nhỏ.

### 6.3 Đảm bảo xấp xỉ

Do hai ràng buộc ngân sách đồng thời và nhiều UAV, các kết quả xấp xỉ đã biết của Orienteering (một ràng buộc) không áp dụng nguyên vẹn. Cần chứng minh riêng, hoặc chỉ dùng như hướng dẫn thiết kế heuristic và báo cáo bằng thực nghiệm. Trên instance nhỏ có thể giải chính xác bằng MILP để đo khoảng cách của heuristic so với nghiệm tối ưu.

## 7. Giả thiết và hạn chế của mô hình

- Mỗi quan sát chỉ nhìn một ô; không xét footprint nhiều ô, góc nghiêng camera, che khuất hay thời tiết.
- Mỗi UAV giữ một độ cao cố định định trước trong suốt nhiệm vụ; độ cao không ảnh hưởng vùng phủ, thời gian hay năng lượng nên bị bỏ qua (kể cả chi phí lên và xuống).
- Mỗi ô được phủ hoặc không (nhị phân), chưa có xác suất phát hiện.
- Rủi ro tĩnh trong suốt nhiệm vụ.
- Mô hình năng lượng tuyến tính theo quãng đường, bỏ qua gió và tải trọng.
- Deadline `D` dùng chung cho mọi UAV.
- Không xét va chạm hay phân tách giữa các UAV.
- Các UAV dùng chung một lưới; khác biệt giữa các UAV nằm ở pin, tốc độ, năng lượng và depot.

## 8. Các mở rộng có thể xét

- Footprint nhiều ô: cho phép `cov(p)` là một hình vuông bán kính `r` ô, khi đó hàm mục tiêu trở thành submodular (max coverage có trọng số) và độ cao thật sự tham gia vào đánh đổi.
- Đưa độ cao vào mô hình (chi phí lên xuống, độ cao biến thiên hoặc chọn độ cao như một siêu tham số).
- Phủ theo xác suất phát hiện: `gain = Σ w_c · (độ tin cậy quan sát)`.
- Rủi ro thay đổi theo thời gian hoặc có hạn thời gian riêng cho từng ô.
- Deadline riêng cho từng UAV.
- Ràng buộc đáp xuống hoặc sạc giữa chừng.

## 9. Ánh xạ sang thuật toán và dữ liệu

| **Thành phần**    | **Cách dùng**                                                                                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Baseline          | Các thuật toán trong bài Song et al. (ICRA 2022): Zig-Zag, Na-Greedy, Heu-Greedy, SVReC. Cần chỉnh để kiểm tra đồng thời deadline và pin, có chặng về depot, và các UAV không chọn trùng ô |
| Thuật toán cơ bản | Greedy thích nghi: chọn (UAV, ô, vị trí chèn) có `w_c` / chi phí chèn chuẩn hoá lớn nhất mà vẫn thoả ràng buộc                                                                             |
| Hướng cải tiến    | Local search (2-opt, thay ô), cluster-first route-second, thuật toán di truyền (GA)                                                                                                        |
| Cận tối ưu        | MILP trên instance nhỏ để đo khoảng cách của heuristic so với nghiệm tối ưu                                                                                                                |
| RescueNet / C2A   | Sinh bản đồ rủi ro `w_c` từ nhãn của dataset, chia lưới 50×50 và 100×100                                                                                                                   |
| Dữ liệu giả lập   | Sinh instance với kích thước và ngân sách kiểm soát được, dùng để cài và kiểm thử thuật toán trước                                                                                         |

Tham số của một instance gồm: lưới (`m`, `n`, `u`) và mảng `w_c`; `K`, `D`, `η`; với mỗi UAV `k` gồm `s_k`, `B_k`, `v_h`, `e_h`, `τ`, `ε`. Nên chọn `D` và `B_k` sao cho UAV chỉ phủ được một phần tổng rủi ro, vì bài toán chỉ có ý nghĩa khi ngân sách không đủ phủ hết.

## 10. Chỉ số đánh giá

- Tỉ lệ rủi ro được phủ: `f / Σ_{c ∈ C} w_c`.
- Cover rate `σ` (tỉ lệ số ô được phủ) và tổng trọng số được phủ `ΣR` như trong bài Song et al., tính theo cùng một mảng `w_c` cho mọi thuật toán, để so sánh trực tiếp được.
- Số ô được quan sát, mức sử dụng pin và thời gian của từng UAV.
- Thời gian chạy thuật toán.
- So sánh với baseline, với cận trên (nếu có) hoặc nghiệm tối ưu trên instance nhỏ.
