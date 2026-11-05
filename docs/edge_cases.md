# Các edge case cho bộ test

Tài liệu này liệt kê các nhóm instance đặc biệt mà đội sinh test nên tạo, bên cạnh các instance thông thường (xem `instance_format.md` về định dạng và cách chọn `D`, `B`). Danh sách là gợi ý dựa trên mô hình hiện tại, chưa được chạy thử trên thuật toán nào; có thể bỏ bớt nhóm không cần cho phạm vi đề tài.

## Nguyên tắc chung

* Phần lớn case nên sinh ở kích thước nhỏ (3x3 đến 10x10) để giải được bằng MILP và đối chiếu. Một số nhóm nên có thêm bản 50x50 để thử quy mô.
* Mỗi nhóm nên có vài seed.
* Gắn tiền tố vào tên file theo nhóm để nhóm thuật toán chạy riêng từng nhóm (cột "Tiền tố" bên dưới).
* Với các case có kết quả mong đợi rõ ràng (lộ trình rỗng, tỉ lệ phủ 100%, mục tiêu 0...), ghi kết quả mong đợi vào trường `source` hoặc một file đi kèm, để dùng làm kiểm thử tự động.

## Bảng tổng hợp nhóm


| Nhóm                             | Tiền tố      | Mục đích                                                          |
| --------------------------------- | -------------- | -------------------------------------------------------------------- |
| 1. Ngân sách ở mức biên      | `edge_budget_` | Kiểm tra ràng buộc deadline, pin và dấu bằng                   |
| 2. Chi phí chụp và bay         | `edge_cost_`   | Kiểm tra tỉ lệ giữa chi phí cố định mỗi ô và chi phí bay |
| 3. Hình dạng bản đồ rủi ro  | `edge_map_`    | Kiểm tra thuật toán trên các phân bố rủi ro khó             |
| 4. Hình học lưới              | `edge_grid_`   | Kiểm tra kích thước, hình dạng và khoảng cách chéo         |
| 5. Vị trí depot                 | `edge_depot_`  | Kiểm tra ảnh hưởng của điểm xuất phát                       |
| 6. Số lượng và cấu hình UAV | `edge_uav_`    | Kiểm tra chia việc giữa các UAV, UAV không đồng nhất         |
| 7. Đáp án biết trước        | `known_`       | Đo trực tiếp chất lượng thuật toán không cần solver        |
| 8. Không hợp lệ                | `invalid_`     | Kiểm tra bộ kiểm tra báo lỗi đúng                             |

## 1. Ngân sách ở mức biên (`edge_budget_`)


| Case               | Cách sinh                                                     | Kiểm tra gì                                   |
| ------------------ | -------------------------------------------------------------- | ----------------------------------------------- |
| Không bay được | `B`hoặc`D`nhỏ đến mức không ô nào ghé được         | Thuật toán trả lộ trình rỗng, không lỗi |
| Vừa đủ một ô  | `eta * B`bằng đúng chi phí đi, chụp và về của một ô | Xử lý dấu bằng, dung sai số thực          |
| Phủ hết được  | Ngân sách lớn hơn chi phí quét hết lưới               | Tỉ lệ phủ 100%, bài toán gần giống TSP   |
| Chỉ`D`chặt       | `B`rất lớn                                                   | Ràng buộc deadline quyết định              |
| Chỉ`B`chặt       | `D`rất lớn                                                   | Ràng buộc pin quyết định                   |
| Cả hai chặt      | Chỉnh sao cho cả hai gần chạm                              | Thuật toán kiểm tra đủ cả hai ràng buộc |

## 2. Chi phí chụp và bay (`edge_cost_`)

* `tau = 0` và `eps = 0`: chỉ còn chi phí bay.
* `tau` hoặc `eps` rất lớn so với chi phí bay: mỗi UAV chỉ ghé được vài ô, nên phải ưu tiên ô giá trị cao nhất.
* `eta = 1` và `eta` nhỏ (ví dụ 0.5): kiểm tra hệ số an toàn pin.

## 3. Hình dạng bản đồ rủi ro (`edge_map_`)


| Case                                                     | Ý nghĩa                                                                                |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Toàn 0                                                  | Mục tiêu bằng 0; tỉ lệ phủ chia cho 0 phải được xử lý                        |
| Đồng đều                                             | Mọi ô cùng`w`; thử cách phá hoà của thuật toán                                 |
| Một ô nóng duy nhất                                  | Đặt ở gần depot, và ở góc xa nhất                                                |
| Một ô nóng ngoài tầm                                | Quá xa để đi và về trong ngân sách                                               |
| Nhiều cụm nóng cách xa nhau, giữa là vùng 0       | Thuật toán chỉ nhìn 8 ô lân cận dễ kẹt (như "Weight Sparse" trong Song et al.) |
| Một ô rất cao ở xa, kèm cụm ô trung bình ở gần | Greedy theo tỉ số`w`/ chi phí có thể chọn sai so với tối ưu                     |
| Dải gradient, bàn cờ, nhiễu ngẫu nhiên             | Đa dạng hình dạng                                                                    |
| `w`chênh lệch rất lớn hoặc phân bố đuôi nặng   | Kiểm tra độ ổn định về số                                                        |

## 4. Hình học lưới (`edge_grid_`)

* Lưới 1x1, 1xn, mx1 (một dòng hoặc một cột), 2x2.
* Lưới không vuông (ví dụ 5x20).
* Các ô nằm trên đường chéo, để kiểm tra khoảng cách chéo `u * sqrt(2)` tính đúng.
* Lưới lớn 100x100 để thử thời gian chạy.

## 5. Vị trí depot (`edge_depot_`)

* Cả `K` UAV cùng một depot.
* Depot ở góc, ở giữa, ở bốn góc khác nhau.
* Depot đặt trên ô có `w` cao nhất, và trên ô có `w = 0`.
* Depot ngay sát cụm nóng, và depot ở góc đối diện.

## 6. Số lượng và cấu hình UAV (`edge_uav_`)

* `K = 1`; `K` lớn hơn số cụm nóng; `K` lớn hơn số ô có `w > 0`.
* Hai UAV cùng depot với một cụm nóng duy nhất: nên chia việc, không ghé trùng.
* UAV không đồng nhất: pin lệch nhiều, một UAV pin quá nhỏ không bay được, một UAV pin rất lớn, tốc độ `v_h` khác nhau.
* Mỗi UAV đặt gần một cụm riêng (nghiệm tốt là mỗi UAV một cụm).

## 7. Instance có đáp án biết trước (`known_`)

Dùng để đo trực tiếp chất lượng thuật toán mà không cần solver:

* Các ô có `w` dương nằm dọc một đường thẳng từ depot, và ngân sách vừa đủ đi hết đường rồi về.
* Ngân sách đủ phủ hết lưới; khi đó mục tiêu bằng tổng mọi `w`.
* Bản nhỏ giải bằng MILP để làm chuẩn.

## 8. Instance và nghiệm không hợp lệ (`invalid_`)

Nhóm này tách riêng vì mục đích là bộ kiểm tra phải báo lỗi, không phải để thuật toán giải.

**Instance không hợp lệ:**

* Depot ngoài lưới.
* `w` âm hoặc NaN.
* `K` khác độ dài `uavs`.
* Ma trận `w` sai kích thước.
* `eta = 0` hoặc lớn hơn 1.
* `v_h = 0`.

**Nghiệm không hợp lệ:**

* Lặp ô trong một lộ trình.
* Ô ngoài lưới.
* Vượt `D`.
* Vượt `eta * B`.
* Số lộ trình khác `K`.
* Lộ trình thiếu hoặc thừa một UAV.
