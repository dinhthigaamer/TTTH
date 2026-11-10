# PROMPT GỬI CLAUDE FREE TẠO SLIDE (PHONG CÁCH HỌC THUẬT HUST: TỐI GIẢN, TẬP TRUNG CHỮ & SỐ LIỆU)

> **Hướng dẫn**: Copy toàn bộ nội dung trong khung bên dưới và gửi cho Claude (kèm theo file template HUST của bạn). Prompt đã được tinh chỉnh để Claude tạo slide theo đúng phong cách báo cáo học thuật: tập trung vào công thức, logic thuật toán, mã giả, bảng số liệu và sơ đồ luồng dạng text; không vẽ hình khối trang trí, không màu mè rườm rà.

---

```text
Bạn hãy đóng vai trò là sinh viên Đại học Bách khoa Hà Nội (HUST) đang chuẩn bị nội dung slide báo cáo môn Tính toán Tiến hóa (Evolutionary Computation). 

YÊU CẦU VỀ PHONG CÁCH TRÌNH BÀY SLIDE (RẤT QUAN TRỌNG):
1. Phong cách học thuật kỹ thuật: Tập trung chủ yếu vào CHỮ, CÔNG THỨC TOÁN, MÃ GIẢ và SỐ LIỆU THỰC NGHIỆM.
2. Tránh tuyệt đối: Không dùng các hình khối trang trí đồ họa rườm rà, không chia quá nhiều thẻ màu mè dạng infographic marketing, không mô tả icon/ảnh minh họa thừa thãi.
3. Cần trực quan hóa thì chỉ dùng: BẢNG SỐ LIỆU (Table) và SƠ ĐỒ LUỒNG LOGIC ĐƠN GIẢN dạng text/mũi tên (ví dụ: A -> B -> C).
4. Định dạng đầu ra: Hãy xuất ra nội dung chi tiết cho từng Slide (Slide 1, Slide 2,... khoảng 13 - 15 slides) với tiêu đề rõ ràng, các gạch đầu dòng cô đọng và bảng biểu số liệu chính xác từ dữ liệu cung cấp dưới đây.

Dưới đây là toàn bộ thông tin khoa học cần đưa vào slide, chia đúng theo 5 NỘI DUNG:

==================================================
PHẦN 1. PHÁT BIỂU BÀI TOÁN
==================================================
* Bối cảnh: Giám sát khu vực thảm họa thiên tai (lũ lụt, sạt lở, bão). Cần điều phối đội bay gồm K UAV xuất phát từ các trạm gốc (depot) để khảo sát, chụp ảnh và thu thập dữ liệu.
* Vấn đề cốt lõi: 
  - Vùng thảm họa rất lớn, thời gian khẩn cấp (Deadline D) và dung lượng pin UAV (B_k) rất hạn chế.
  - UAV không thể bay phủ hết 100% diện tích.
  - Mục tiêu bắt buộc: Tối đa hóa mức độ rủi ro thu thập được, tập trung vào các "điểm nóng" nguy hiểm thay vì bay tuần tra lan man vào vùng an toàn.
* Hai bài toán nghiên cứu được tích hợp trong đề tài:
  - Bài toán 1 (Disaster Environment Coverage - Song et al., 2022): Lập lộ trình cho K UAV trên lưới rủi ro liên tục w_c ∈ [0, 1] dưới ràng buộc đồng thời về deadline và pin. Áp dụng cho: Zig-Zag, Na-Greedy, Heu-Greedy, SVReC và GA.
  - Bài toán 2 (Priority-based CPP - Cho et al., 2026): Lập lộ trình đa UAV trên khu vực có phân tầng ưu tiên (High / Medium / Low) quanh tâm thảm họa; áp dụng thuật toán đàn kiến ACO để cân bằng tải và tối ưu đường bay.

==================================================
PHẦN 2. MÔ HÌNH HÓA TOÁN HỌC
==================================================
1. Môi trường và UAV:
   - Khu vực chia thành lưới m × n ô vuông cạnh u (mét). Mỗi ô c = (i, j) có mức rủi ro w_c ≥ 0.
   - Khoảng cách Euclid bay thẳng giữa 2 ô: d(a, b) = u · sqrt((i - i')^2 + (j - j')^2).
   - Đội K UAV: UAV k có trạm xuất phát/về s_k, tốc độ bay v_h (m/s), tiêu thụ năng lượng e_h (J/m), thời gian quan sát mỗi ô τ (s), năng lượng chụp ε, pin danh định B_k, hệ số an toàn pin η ∈ (0, 1].
2. Biến quyết định:
   - Lộ trình UAV k: R_k = (c_1, c_2, ..., c_L) với c_0 = s_k.
   - Thời gian hoàn thành quan sát ô cuối: T(R_k) = [Σ_{q=1..L} d(c_{q-1}, c_q)] / v_h + (L - 1)·τ.
   - Tổng năng lượng (bảo đảm có đủ pin bay từ ô cuối về depot): E(R_k) = e_h · [Quãng bay + d(c_L, s_k)] + L·ε.
3. Hàm mục tiêu (Modular Team Orienteering trên lưới):
   max f(R_1, ..., R_K) = Σ_{c ∈ ∪ R_k} w_c   (mỗi ô chỉ tính điểm 1 lần dù nhiều UAV ghé qua)
4. Ba ràng buộc bắt buộc:
   - (C1) Deadline: T(R_k) ≤ D, ∀k = 1..K.
   - (C2) Pin an toàn: E(R_k) ≤ η · B_k, ∀k = 1..K.
   - (C3) Lộ trình hợp lệ: Không lặp ô trong một lộ trình và các lộ trình rời nhau (R_i ∩ R_j = ∅).

==================================================
PHẦN 3. HAI BÀI BÁO GẦN VỚI ĐỀ TÀI
==================================================
1. Bài báo 1: Song et al. (IEEE ICRA 2022) - "Multi-UAV Disaster Environment Coverage Planning with Limited-Endurance"
   - Mô hình hóa MAEl-CPP trên bản đồ nhiệt thảm họa.
   - Chỉ ra 4 hiện tượng làm thuật toán tham lam cục bộ thất bại:
     + Weight Trap: Bị kẹt trong cụm rủi ro thấp do các ô xung quanh đã phủ hết.
     + Weight Interference: Các UAV tranh chấp, dẫm chân lên cùng một cụm.
     + Weight Sparse: Cụm rủi ro cao bị cô lập bởi dải ô rủi ro bằng 0.
     + Weight Redundancy: Nhiều ô lân cận có cùng trọng số lớn nhất.
   - Đề xuất các thuật toán: Na-Greedy, Heu-Greedy và SVReC.
2. Bài báo 2: Cho et al. (Springer Soft Computing, 2026) - "Coverage path planning for multiple UAVs in prioritized regions using ant colony optimization"
   - Bài toán tìm kiếm cứu nạn đa UAV trên các vùng ưu tiên.
   - Đóng góp: Thuật toán ACO (Algorithm 3-2, Eq. 18-23) với luật chọn ô kết hợp mùi Pheromone τ và Heuristic η = w/T; cơ chế nhảy thoát Deadlock khi không còn lân cận; cập nhật Pheromone 2 mức (cục bộ sau mỗi bước và toàn cục sau mỗi thế hệ).

==================================================
PHẦN 4. MÔ TẢ BỘ TEST BENCHMARK
==================================================
- Nguồn dữ liệu: Dataset viễn thám thực tế RescueNet (ảnh thảm họa bão Michael), ánh xạ nhãn thiệt hại thực tế sang trọng số rủi ro w_c ∈ [0, 1].
- Các nhóm dữ liệu thực nghiệm:
  + Nhóm 1: 288 instances cỡ nhỏ và vừa (< 50×50, gồm 15×20, 30×40), K = 3 UAVs, 3 mức ngân sách b10 (10%), b25 (25%), b50 (50%).
  + Nhóm 2: Bộ RescueNet 50×50 (2.500 ô) và 100×100 (10.000 ô), K = 6 UAVs tại 6 depot khác nhau, 3 mức ngân sách blow, bmedium, bhigh.
- Quy ước thiết kế test: Ngân sách D và B được khống chế ở mức biên để UAV chỉ phủ được 10% - 50% tổng rủi ro, buộc thuật toán phải cạnh tranh và tối ưu việc chọn ô.

==================================================
PHẦN 5. KẾT QUẢ CÁC THUẬT TOÁN ĐÃ CÀI ĐẶT
==================================================

5.1. Mô tả nguyên lý & Mã giả 6 thuật toán:

(1) Zig-Zag (Baseline quét tuần tự):
- Nguyên lý: Quét boustrophedon theo hàng, bỏ qua giá trị rủi ro w. Chia dải hàng cho K UAV theo vị trí depot, dừng ở ô đầu tiên vi phạm C1 hoặc C2.
- Mã giả:
  Chia m hàng thành K dải liền kề cho K UAV
  Với mỗi UAV k:
      Tạo đường bay quét zic-zac trên dải
      Duyệt từng ô c: Nếu can_observe(c) thì đi tiếp, ngược lại dừng UAV k

(2) Na-Greedy (Tham lam ngây thơ):
- Nguyên lý: Tại mỗi bước, xét 8 ô lân cận chưa phủ, chọn ô có w lớn nhất. Dừng khi lân cận không còn ô khả thi (dễ mắc bẫy Weight Trap).
- Mã giả:
  Lặp luân phiên các UAV còn hoạt động:
      Nbrs = {c ∈ 8_lân_cận(cur) : c chưa phủ ∧ can_observe(c)}
      Nếu Nbrs rỗng: UAV dừng
      c* = argmax_{c ∈ Nbrs} w[c] (hòa điểm chọn ô gần hơn)
      Quan sát c*; Đóng ô c*

(3) Heu-Greedy (Tham lam có phạt):
- Nguyên lý: Cho phép UAV bay qua (transit) ô đã phủ khi bị kẹt với trọng số bị trừ điểm phạt r = w - max(w) ≤ 0. Giúp thoát bẫy nhưng dễ dao động qua lại làm tốn pin vô ích.

(4) SVReC (SVeC + VwRR - Thuật toán tốt nhất của Song et al.):
- Nguyên lý: Nếu trung bình lân cận < r_th (= mean(w)), cho phép UAV "nhảy xa" đến ô chưa phủ có w > r_th gần nhất (SVeC). Khi hòa điểm, chọn ô có trung bình rủi ro xung quanh cao nhất (VwRR).
- Mã giả:
  Lặp luân phiên các UAV:
      Nbrs = {c ∈ 8_lân_cận(cur) : c chưa phủ ∧ can_observe(c)}
      Nếu Nbrs rỗng hoặc mean(w[Nbrs]) < mean(w):
          t = ô chưa phủ có w > mean(w) gần nhất mà can_observe(t)
          Nếu có t: Quan sát t; Tiếp tục vòng lặp
          Nếu Nbrs rỗng: UAV dừng
      c* = Ô có trung bình w lân cận cao nhất trong Nbrs
      Quan sát c*; Đóng ô c*

(5) ACO (Ant Colony Optimization - Cho et al., 2026):
- Nguyên lý: Đàn kiến A con qua G thế hệ. Điểm hấp dẫn kết hợp mùi Pheromone τ và Heuristic η = w/T. Quy tắc ACS chọn ô theo q_0 (khai thác) và Roulette Wheel (khám phá). Có bước nhảy thoát Deadlock khi lân cận rỗng. Cập nhật pheromone cục bộ sau mỗi bước và toàn cục sau mỗi thế hệ.
- Sơ đồ luồng logic:
  [Khởi tạo τ_0 = 1.0] -> [Thế hệ g = 1..G] -> [Kiến a = 1..A] -> [Chọn ô: max τ^α · η^β (q ≤ q_0) hoặc Roulette] -> [Cập nhật cục bộ τ(c) = (1-ρ)τ + ρτ_0] -> [Cập nhật toàn cục τ = (1-ε)τ + εΔτ trên Best Route]

(6) GA (Genetic Algorithm - Memetic):
- Nguyên lý: Nhiễm sắc thể là ma trận trọng số ưu tiên gen prio kích thước m×n. Hàm giải mã _decode điều phối K UAV theo tỷ số prio[c] / dist(cur, c) đảm bảo 100% cá thể hợp lệ với C1-C3. Tiến hóa qua Elitism, Tournament Selection, Uniform Crossover và Đột biến nhiễu ngẫu nhiên.

--------------------------------------------------
5.2. Bảng số liệu đối sánh thực nghiệm:

Bảng 1: So sánh độ phức tạp lý thuyết (N = m·n, S là số bước bay tối đa):
| Thuật toán | Chi phí mỗi bước | Tổng độ phức tạp thời gian | Cơ chế xử lý bẫy tham lam |
|---|---|---|---|
| Zig-Zag | O(1) | O(N + K log K) | Không có (bỏ qua rủi ro w) |
| Na-Greedy | O(1) | O(N + K·S) | Không có (dễ kẹt Weight Trap) |
| Heu-Greedy | O(1) | O(N + K·S) | Phạt điểm ô đã phủ để bay qua |
| SVReC | O(1) -> O(N log N) | O(min(N, KS)·N log N) xấu nhất | Nhảy SVeC tới ô rủi ro cao gần nhất + phá hòa VwRR |
| ACO | O(1) -> O(N log N) | O(G·A·(N + KS)) thông thường | Deadlock Jump + Học mùi Pheromone đa thế hệ |
| GA | O(1) -> O(N log N) | O(G_ga·P·(N + KS)) thông thường | Tiến hóa ma trận ưu tiên prio + Giải mã hợp lệ |

Bảng 2: Thống kê tổng rủi ro phủ (f) trên 288 instances tập con (< 50×50, K = 3 UAVs):
| Thuật toán | Giá trị trung bình (Mean) | Trung vị (Median) | Giá trị lớn nhất (Max) | So với Zig-Zag |
|---|---|---|---|---|
| Zig-Zag | 34.06 | 17.18 | 295.66 | Mốc cơ sở |
| Na-Greedy | 34.20 | 20.23 | 196.27 | +0.4% |
| Heu-Greedy | 47.59 | 27.00 | 380.77 | +39.7% |
| SVReC | 69.34 | 42.79 | 425.04 | +103.6% |

Bảng 3: Đối đầu 6 thuật toán trên RescueNet 50×50 và 100×100 (K = 6 UAVs - Điểm f và Tỷ lệ risk-coverage %):
| Cấu hình test | Tổng W | Zig-Zag | Na-Greedy | Heu-Greedy | SVReC | ACO | GA |
|---|---|---|---|---|---|---|---|
| 50×50 (blow) | 269.5 | 12.68 (4.71%) | 25.72 (9.54%) | 25.92 (9.62%) | 32.45 (12.04%) | 30.54 (11.33%) | 40.04 (14.86%) |
| 50×50 (bmedium) | 269.5 | 16.08 (5.97%) | 27.67 (10.26%) | 29.00 (10.76%) | 51.77 (19.21%) | 47.45 (17.60%) | 58.00 (21.52%) |
| 50×50 (bhigh) | 269.5 | 19.97 (7.41%) | 32.22 (11.95%) | 42.30 (15.69%) | 77.12 (28.61%) | 67.28 (24.96%) | 84.51 (31.35%) |
| 100×100 (blow) | 1078.2 | 67.69 (6.28%) | 62.06 (5.76%) | 91.49 (8.49%) | 286.92 (26.61%) | 202.00 (18.74%) | 286.92 (26.61%) |
| 100×100 (bmedium)| 1078.2 | 106.37 (9.87%) | 81.46 (7.56%) | 178.01 (16.51%) | 427.59 (39.66%) | 373.96 (34.68%) | 430.18 (39.90%) |
| 100×100 (bhigh) | 1078.2 | 133.03 (12.34%)| 91.36 (8.47%) | 262.20 (24.32%) | 435.33 (40.38%) | 457.76 (42.46%) | 491.00 (45.54%) |

Bảng 4: So sánh thời gian chạy trung bình (Runtime):
| Nhóm thuật toán | Thời gian thực thi trên lưới 50×50 | Thời gian thực thi trên lưới 100×100 |
|---|---|---|
| Nhóm Greedy (Zig-Zag, Na-Greedy, Heu-Greedy, SVReC) | 0.001s - 0.074s | 0.003s - 0.128s |
| Thuật toán đàn kiến (ACO) | 1.69s - 2.29s | 5.97s - 8.97s |
| Thuật toán di truyền (GA) | 2.18s - 3.17s | 8.12s - 12.32s |

==================================================
KẾT LUẬN CẦN RÚT RA TRÊN SLIDE TỔNG KẾT:
1. Đánh đổi giữa Thời gian và Chất lượng: Nhóm Greedy chạy siêu nhanh (< 0.1s) nhưng chất lượng thấp vì thiển cận; ACO và GA tốn thêm vài giây tính toán (vẫn đáp ứng tốt thời gian thực cứu nạn) nhưng tăng vọt tỷ lệ bao phủ rủi ro.
2. Hiệu quả thuật toán:
   - Zig-Zag kém nhất (chỉ đạt ~5% - 12%) do bay lan man vào ô không có rủi ro.
   - SVReC là giải thuật Heuristic tốt nhất của bài báo Song et al. nhờ cơ chế nhảy xa SVeC.
   - GA và ACO vượt trội hoàn toàn trên các lưới lớn (đạt tới 42.5% - 45.5% tổng rủi ro, gấp 3 - 4 lần Zig-Zag) nhờ khả năng học tập tích lũy và tìm kiếm toàn cục.
```
