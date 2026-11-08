# Phương pháp sinh instance RescueNet

Tài liệu mô tả cách xây dựng bộ instance `rescuenet_50x50_K6_distinct_depots_v4`
cho bài toán Deadline-constrained Multi-UAV Risk Coverage.
Chi phí và ràng buộc tuân theo [mô hình bài toán](mo_hinh_deadline_constrained_multi_uav_risk_coverage.md);
JSON tuân theo [chuẩn input và nghiệm](instance_format.md).

## 1. Cấu hình thực nghiệm

| Thành phần | Giá trị |
|---|---|
| Ảnh nguồn | 4.494; phân hoạch cố định: 899 val, 3.595 test |
| Lưới | 50×50 ô; cạnh ô mô phỏng u=40 m |
| UAV | K=6, sáu depot riêng |
| Seed | 42, 43, 44 |
| Mức ngân sách | low, medium, high |
| Số lượng | 9 instance/ảnh; tổng 40.446 = 8.091 val + 32.355 test |
| Thông số chung | v_h=15 m/s, e_h=1, tau=2 s, eps=0, eta=1 |

[Cấu hình JSON](../configs/rescuenet_instances.json) là nguồn tham số chính.
[Manifest nguồn](../configs/rescuenet_split.csv) cố định ID ảnh, split, nhóm cảnh và checksum.
Các biến thể của cùng ảnh nằm cùng split. Val dùng chọn tham số; test dùng đánh giá sau khi cố định tham số.

## 2. Xây dựng bản đồ rủi ro

Mask ID gốc được crop giữa thành 3.000×3.000 pixel rồi chia thành 50×50 ô, mỗi ô 60×60 pixel.
Phần rìa ngoài crop không tham gia instance; mask không bị kéo giãn.
Với tập pixel $A_{ij}$ thuộc ô hàng i, cột j:

$$
w_{ij}=\frac{1}{3600}\sum_{x\in A_{ij}}\rho(\operatorname{label}(x)).
$$

w được làm tròn đến 12 chữ số thập phân. Chỉ số ô bắt đầu từ 0.
u=40 m là thang mô phỏng được lựa chọn, không suy ra từ độ phân giải địa lý của ảnh.

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

rho biểu diễn ưu tiên khảo sát. Nhãn Tree không phân biệt mức hư hại; Water không phân biệt nước tự nhiên và ngập.
343 bản đồ có tổng w=0 được giữ làm trường hợp biên; tỷ lệ phủ của chúng là N/A.

## 3. Bố trí depot

Mỗi cặp `(image_id, seed)` xác định sáu ô depot có thứ tự, không trùng nhau.
SHA-256 của ID ảnh và seed được dùng khởi tạo bộ sinh số NumPy: sinh ô đầu tiên,
sau đó lấy năm ô không hoàn lại từ 2.499 ô còn lại.

Cách chọn độc lập với rủi ro, không áp đặt khoảng cách tối thiểu giữa các depot.
Cùng ảnh/seed giữ nguyên sáu depot và thứ tự UAV ở cả ba mức ngân sách.
Mỗi UAV xuất phát và quay về depot riêng; vị trí cất/hạ cánh thực tế không thuộc mô hình.

## 4. Chi phí và điều kiện khả thi

Với UAV k có depot $s_k$, route $R_k=(c_1,\ldots,c_L)$, L>0, đặt $c_0=s_k$:

$$
d(a,b)=u\sqrt{(a_i-b_i)^2+(a_j-b_j)^2},
$$

$$
S_k=\sum_{q=1}^{L}d(c_{q-1},c_q),\qquad r_k=d(c_L,s_k).
$$

$S_k$ là quãng đường đến ô cuối; $r_k$ là chặng quay về depot riêng. Hai ràng buộc:

$$
T_{\mathrm{last}}(R_k)=\frac{S_k}{v_h}+(L-1)\tau\le D,
$$

$$
E(R_k)=e_h(S_k+r_k)+L\varepsilon\le\eta B_k.
$$

D là deadline chung (giây); B_k là dung lượng pin riêng; eta là hệ số an toàn.
Trong JSON, tau và eps tương ứng $\tau$ và $\varepsilon$.
B dùng đơn vị năng lượng quy ước. Deadline đến lúc tới ô cuối; pin bao gồm chặng quay về.
Route rỗng được phép. Ô depot chỉ nhận thưởng khi được ghi là ô quan sát trong route.

## 5. Thiết kế ngân sách

### 5.1. Đường quét tham chiếu

Một UAV giả định quét zig-zag đủ 2.500 ô từ depot `uavs[0]`.
Trong tám hướng quét hàng/cột và lật hướng, chọn hướng có năng lượng nhỏ nhất;
nếu bằng năng lượng thì chọn thời gian đến ô cuối nhỏ hơn.
Với quãng đường đến ô cuối $S_{\mathrm{full}}$ và chặng về $r_{\mathrm{full}}$:

$$
E_{\mathrm{full}}=e_h(S_{\mathrm{full}}+r_{\mathrm{full}})+2500\varepsilon,
$$

$$
T_{\mathrm{full}}=\frac{S_{\mathrm{full}}}{v_h}+2499\tau.
$$

### 5.2. Deadline và pin từng UAV

| Mức | alpha |
|---|---:|
| low | 0.125 |
| medium | 0.20 |
| high | 0.24 |

Với K=6, pin cơ sở và deadline được xác định bằng:

$$
B_{\mathrm{base}}=\operatorname{round}_6\!\left(\frac{\alpha E_{\mathrm{full}}}{K\eta}\right),
\qquad D=\operatorname{round}_6\!\left(\frac{\alpha T_{\mathrm{full}}}{K}\right).
$$

$\operatorname{round}_6$ là `round(..., 6)` của Python. Phân bổ pin theo thứ tự UAV:

$$
(a_1,\ldots,a_6)=(0.8,0.9,0.95,1.05,1.1,1.2),
\qquad \widetilde B_k=\operatorname{round}_{6,\mathrm{HALF\_UP}}(a_kB_{\mathrm{base}}).
$$

Năm UAV đầu dùng $B_k=\widetilde B_k$; UAV cuối nhận phần dư làm tròn:

$$
B_6=6B_{\mathrm{base}}-\sum_{k=1}^{5}\widetilde B_k,
\qquad \sum_{k=1}^{6}B_k=6B_{\mathrm{base}}.
$$

Chỉ số k=1…6 tương ứng `uavs[0]`…`uavs[5]`.
B cơ sở là pin trung bình; mỗi UAV được kiểm tra bằng B_k và depot riêng.
Cả pin và deadline cùng thay đổi giữa ba mức.

Đây là quy tắc thiết kế ngân sách của bộ dữ liệu, dựa trên gợi ý zig-zag trong chuẩn instance.
Mô hình quy định công thức chi phí và điều kiện khả thi, không bắt buộc cách chọn B/D trên.
alpha biểu diễn tài nguyên tham chiếu, không phải tỷ lệ rủi ro phủ.

### 5.3. Ví dụ

[Instance ảnh 10778, low, seed 42](examples/rescuenet_10778_50x50_K6_blow_s42.json)
có E_full=102490.26479563164 và T_full=11745.309329876338 giây.
Với alpha=0.125, B_base=2135.213850 và D=244.693944 giây.
Sáu pin là [1708.171080, 1921.692465, 2028.453158, 2241.974543, 2348.735235, 2562.256619],
tổng bằng 12811.283100.

## 6. Kiểm chứng và đánh giá

Generator kiểm tra mask nguồn, phép tổng hợp w và tính hợp lệ của mọi instance.
Validator đối chiếu JSON với cấu hình, seed, depot, pin, deadline và checksum lưới.
Khi chấm nghiệm, không lặp ô trong một route; ô được nhiều UAV ghé chỉ cộng thưởng một lần.

Greedy tham chiếu xử lý UAV theo thứ tự JSON, chọn ô chưa quan sát có tỷ số
$w_c/(d/u+1)$ lớn nhất trong các ô còn thỏa deadline và pin, tính cả chặng về depot riêng.
Mọi route được kiểm tra lại độc lập.
Tỷ lệ rủi ro phủ bằng tổng w đã quan sát chia tổng w toàn bản đồ.
Thống kê lấy trung bình ba seed trong từng ảnh rồi tổng hợp theo ảnh, riêng từng mức; loại tỷ lệ N/A.

Ba alpha bắt nguồn từ pilot 32 nhóm ảnh val với depot chung và pin đồng nhất,
hướng đến mức phủ 15/30/45%. Ngân sách được giữ nguyên khi chuyển sang depot riêng và pin khác nhau.
Kết quả greedy đã đo trên test của phiên bản hiện tại:

| Mức | Phủ trung bình (%) | Trung vị theo ảnh (%) |
|---|---:|---:|
| low | 33.55 | 27.36 |
| medium | 53.20 | 49.06 |
| high | 60.40 | 59.02 |

Đây là mức đạt được của heuristic, không phải nghiệm tối ưu hay chặn trên.
Bộ hiện tại không bảo đảm mỗi instance phủ 10–50%; medium/high vượt 50% về trung bình test.
Các thay đổi ngân sách tiếp theo phải được chọn trên val trước khi đánh giá test.

## 7. Tái lập

Từ thư mục gốc repo, với mask gốc đúng đường dẫn trong manifest nguồn:

```powershell
python -m pip install -r requirements-instances.txt
python -m unittest discover -s tests -v
python scripts/generate_rescuenet_instances.py --workers 6
python scripts/validate_instances.py --workers 6
python scripts/evaluate_instances.py --workers 6
```

Đầu ra nằm trong `data/instances/rescuenet_50x50/`, gồm val, test, manifest, summary và README.
Kết quả đánh giá nằm trong `outputs/rescuenet_coverage/`.
`manifest.csv` ghi B_base tại cột B, pin riêng tại B_values và sáu depot tại depot_values.
`summary.json` lưu checksum và kết quả kiểm chứng.
Generator từ chối trộn các cấu hình khác nhau trong cùng bộ dữ liệu.
