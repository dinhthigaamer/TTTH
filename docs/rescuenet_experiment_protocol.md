# Phương pháp sinh instance RescueNet

Tài liệu mô tả quy trình chọn ảnh và sinh hai bộ 50×50, 100×100 cho bài toán
Deadline-constrained Multi-UAV Risk Coverage. Chỉ những ảnh có đủ chín instance
được chứng nhận mức phủ tối ưu trong 10–50% mới được đưa vào bộ dữ liệu chính.
Input và nghiệm tuân theo [chuẩn chung](instance_format.md);
chi phí và ràng buộc tuân theo [mô hình bài toán](mo_hinh_deadline_constrained_multi_uav_risk_coverage.md).

## 1. Cấu hình cố định

| Thành phần | Giá trị |
|---|---|
| Nguồn | 4.494 ảnh/mask RescueNet |
| Phân hoạch nguồn | 899 val, 3.595 test, giữ theo manifest đã đóng băng |
| Lưới | 50×50 hoặc 100×100; cạnh ô mô phỏng u=40 m |
| UAV | K=6, sáu depot khác nhau |
| Seed | 42, 43, 44 |
| Biến thể mỗi ảnh được chọn | 3 mức ngân sách × 3 seed = 9 instance |
| Thông số chung | v_h=15 m/s, e_h=1, tau=2 s, eps=0, eta=1 |
| Hệ số pin | [0.8,0.9,0.95,1.05,1.1,1.2] |

Tham số nằm trong [config 50×50](../configs/rescuenet_instances.json) và
[config 100×100](../configs/rescuenet_instances_100x100.json).
[Manifest nguồn](../configs/rescuenet_split.csv) lưu ID, nhóm cảnh, split, đường dẫn và checksum.
Ảnh được chọn giữ nguyên split nguồn; các biến thể của cùng ảnh luôn nằm cùng tập.
Val dùng chọn tham số thuật toán, test dùng đánh giá sau khi cố định tham số.

## 2. Bản đồ rủi ro

Mask ID gốc được crop giữa thành 3.000×3.000 pixel, không resize hoặc nội suy nhãn.
Phần rìa ngoài crop không tham gia instance.

| Lưới | Pixel mỗi ô | Số ô | Quy mô mô phỏng theo u=40 m |
|---|---|---:|---|
| 50×50 | 60×60 | 2.500 | 2×2 km |
| 100×100 | 30×30 | 10.000 | 4×4 km |

Với tập pixel A của một ô, trọng số ô là:

$$
w_c=\frac{1}{|A_c|}\sum_{x\in A_c}\rho(\operatorname{label}(x)).
$$

w được làm tròn đến 12 chữ số. Chỉ số hàng/cột bắt đầu từ 0.
u=40 m là giả định mô phỏng, không suy ra từ độ phân giải địa lý của ảnh.
Hai bộ dùng cùng crop, nhưng không mô phỏng cùng một diện tích vật lý cố định.

| Lớp | rho |
|---|---:|
| Background | 0.00 |
| Water | 0.10 |
| Building No Damage | 0.00 |
| Building Minor Damage | 0.25 |
| Building Major Damage | 0.75 |
| Building Total Destruction | 1.00 |
| Vehicle | 0.00 |
| Road-Clear | 0.00 |
| Road-Blocked | 0.20 |
| Tree | 0.20 |
| Pool | 0.00 |

rho biểu diễn ưu tiên khảo sát, không phải xác suất thiệt hại.
Tree không phân biệt mức hư hại; Water không phân biệt ngập và nước tự nhiên.

## 3. Depot và ngân sách

Mỗi cặp `(image_id, seed)` khởi tạo NumPy RNG từ SHA-256 để lấy sáu ô depot
không hoàn lại. Depot không phụ thuộc rủi ro và giữ nguyên giữa ba mức ngân sách.
Mỗi UAV xuất phát và quay về depot riêng; không áp đặt khoảng cách tối thiểu giữa depot.

Một UAV tham chiếu quét zig-zag đủ L=mn ô từ depot `uavs[0]`.
Trong tám hướng quét hàng/cột và lật hướng, chọn năng lượng nhỏ nhất, rồi thời gian nhỏ hơn khi bằng năng lượng.
Với S_full là quãng đường đến ô cuối, r_full là chặng quay về:

$$
E_{\mathrm{full}}=e_h(S_{\mathrm{full}}+r_{\mathrm{full}})+L\varepsilon,
\qquad T_{\mathrm{full}}=S_{\mathrm{full}}/v_h+(L-1)\tau.
$$

Pin cơ sở và deadline được chọn bằng:

$$
B_{\mathrm{base}}=\operatorname{round}_6\!\left(\frac{\alpha_E E_{\mathrm{full}}}{K\eta}\right),
\qquad D=\operatorname{round}_6\!\left(\frac{\alpha_T T_{\mathrm{full}}}{K}\right).
$$

Bước đầu dùng alpha_E=alpha_T; khi cần, chọn hai hệ số độc lập.
Đây là quy tắc thiết kế ngân sách, không phải công thức sinh B/D bắt buộc của mô hình.
Các hệ số tài nguyên không đồng nghĩa với tỷ lệ rủi ro phủ.

Với a_k là hệ số pin đã chốt:

$$
\widetilde B_k=\operatorname{round}_{6,\mathrm{HALF\_UP}}(a_kB_{\mathrm{base}}),
\quad B_k=\widetilde B_k\ (k=1,\ldots,5),
\quad B_6=6B_{\mathrm{base}}-\sum_{k=1}^{5}\widetilde B_k.
$$

Tổng pin đội bằng đúng 6*B_base ở độ chính xác một phần triệu.
Chỉ số k=1…6 tương ứng `uavs[0]`…`uavs[5]`. B dùng đơn vị năng lượng quy ước.

## 4. Điều kiện khả thi và chứng nhận mức phủ

Với route R_k gồm p>0 ô, đặt c_0=s_k, S_k là quãng đường đến ô cuối và r_k là chặng về:

$$
d(a,b)=u\sqrt{(a_i-b_i)^2+(a_j-b_j)^2},
\quad S_k=\sum_{q=1}^{p}d(c_{q-1},c_q),\quad r_k=d(c_p,s_k).
$$

$$
T_{\mathrm{last}}(R_k)=S_k/v_h+(p-1)\tau\le D,
\qquad E(R_k)=e_h(S_k+r_k)+p\varepsilon\le\eta B_k.
$$

Deadline không tính chụp ở ô cuối hoặc thời gian quay về; pin có tính chặng quay về.
Route không lặp ô; ô chung giữa nhiều UAV chỉ cộng thưởng một lần.

Với W là tổng rủi ro và F* là giá trị tối ưu, mỗi instance được giữ phải có chứng nhận:

$$
0.1W\le F_{\mathrm{LB}}\le F^*\le F_{\mathrm{UB}}\le0.5W,\qquad W>0.
$$

Cận dưới là tổng rủi ro của một nghiệm khả thi, tạo bằng tiền tố zig-zag hoặc greedy
trên cửa sổ 9×9 cùng 32 ô rủi ro cao nhất; khi cần tìm trên toàn bộ ô còn lại.
Bộ kiểm tra tính lại chi phí và mục tiêu từ route, không tin giá trị lưu sẵn.

Cận trên dùng các ô có rủi ro dương có thể ghé riêng lẻ rồi quay về:

$$
C_k=\{c:w_c>0,\ d(s_k,c)/v_h\le D,\ 2e_hd(s_k,c)+\varepsilon\le\eta B_k\}.
$$

Đặt d_k là khoảng cách gần nhất từ depot k đến một ô rủi ro dương,
delta là khoảng cách nhỏ nhất giữa hai ô rủi ro dương khác nhau (lấy u nếu chỉ có một ô).
Số ô rủi ro dương được UAV k ghé không vượt:

$$
q_k=\max\!\left(0,\min\left(mn,
\left\lfloor\frac{D-d_k/v_h}{\delta/v_h+\tau}\right\rfloor+1,
\left\lfloor\frac{\eta B_k-2e_hd_k+e_h\delta}{e_h\delta+\varepsilon}\right\rfloor\right)\right).
$$

Gọi H_q(S) là tổng q trọng số lớn nhất trong S, hoặc toàn bộ S nếu có ít hơn q ô:

$$
F_{\mathrm{UB}}=\min\left(W,
H_{\sum_kq_k}\!\left(\bigcup_kC_k\right),\ \sum_kH_{q_k}(C_k)\right).
$$

Mỗi ô được quan sát phải khả đạt và mỗi UAV ghé không quá q_k ô rủi ro.
Bỏ điều kiện nối đường bay và cho phép đếm trùng chỉ làm tăng miền nghiệm;
vì vậy cận trên không thấp hơn tối ưu thực sự.
Phương pháp cũ `reachable_top_weights_v1` dùng cận yếu hơn d_k=0, delta=u và vẫn hợp lệ.
Trọng số được đổi sang số nguyên theo đơn vị 10^-12; kiểm tra khoảng 10–50% không dựa trên trung bình tập.
Các kiểm tra hình học và số ô được làm tròn ra ngoài để giữ tính bảo thủ.

## 5. Chọn ảnh và sinh instance

1. Kiểm tra mask nguồn, tạo w và sinh depot theo seed cố định.
2. Tìm ngân sách trong phạm vi hệ số [0.00001,0.5], dùng 28 bước chia đôi và cận trên ≤50%.
3. Thử ba mức tương đối [0.6,0.8,1]; khi cần dùng các bộ mức gần nhau hơn ghi trong config.
4. Nếu cần, cố định alpha_E trong [0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.5] rồi tìm alpha_T độc lập.
5. Chỉ ghi JSON khi đủ ba cặp B/D khác nhau sau làm tròn cho cả ba seed, và cả chín chứng nhận đều đạt.

Quy tắc áp dụng giống nhau cho val/test; không dùng kết quả thuật toán được so sánh để chọn ngân sách.
Do rủi ro theo ô là rời rạc, mức phủ không bắt buộc tăng đều giữa ba mức tài nguyên.
Khoảng 10–50% được chứng nhận cho giá trị tối ưu; một thuật toán kém vẫn có thể đạt dưới 10%.

Ảnh chưa được chọn chỉ được ghi vào `data/rescuenet_unselected/50x50.csv` hoặc `100x100.csv`:

- `proven_incompatible`: có chứng minh không đáp ứng với cấu hình và seed bắt buộc.
- `not_certified`: tìm kiếm chưa có đủ chứng nhận; không kết luận ảnh bất khả thi.

Các CSV chỉ lưu ID, đường dẫn ảnh/mask gốc và lý do. Không tạo JSON, route hoặc ngân sách dự phòng
cho ảnh bị loại. Dữ liệu gốc không bị di chuyển. Không có edge instance trong hai bộ đầu ra.

## 6. Tái lập và kiểm tra

Từ gốc repo, sau khi đặt mask gốc theo đường dẫn trong manifest nguồn:

```powershell
python -m pip install -r requirements-instances.txt
python -m unittest discover -s tests -v
python scripts/generate_rescuenet_instances.py --grid-size 50 --workers 6
python scripts/generate_rescuenet_instances.py --grid-size 100 --workers 6
python scripts/validate_instances.py data/instances/rescuenet_50x50 --workers 6
python scripts/validate_instances.py data/instances/rescuenet_100x100 --workers 6
```

Mỗi thư mục instance chỉ có val, test, manifest.csv, certificates.jsonl, summary.json và README.md.
Certificates lưu nghiệm theo chuẩn solution, cận kiểm chứng và SHA-256 của input.
Validator đọc lại tất cả JSON, tính lại cận và nghiệm, kiểm tra số lượng, split và ba cặp ngân sách khác nhau.
Số lượng ảnh được chọn và instance được ghi trong summary; ảnh còn lại không thuộc tập thực nghiệm.

Đánh giá greedy tham chiếu bằng `scripts/evaluate_instances.py --data <thu_muc_instance>`.
Thống kê lấy trung bình ba seed trong từng ảnh rồi tổng hợp theo ảnh, riêng từng mức ngân sách.
