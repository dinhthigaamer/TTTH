# Định dạng instance và nghiệm

Tài liệu này mô tả định dạng file test (instance) và file nghiệm (solution) cho bài toán Deadline-constrained Multi-UAV Risk Coverage. Mọi nhóm (bộ test, thuật toán, bộ kiểm tra) dùng chung định dạng này.

Mô hình tương ứng: mỗi UAV bay ở độ cao cố định thiết lập riêng và **không đưa vào mô hình**; mỗi lần quan sát nhìn được đúng một ô; chỉ xét bay ngang trên lưới.

## Quy ước chung

* **Đơn vị:**`u` mét; `v_h` m/s; `D`, `tau` giây; `B`, `eps` cùng một đơn vị năng lượng; `e_h` là năng lượng trên mét.
* **Chỉ số ô:** đánh số từ 0. Ô `[i, j]` ở hàng `i` (0 đến `m-1`) và cột `j` (0 đến `n-1`). `w[i][j]` là rủi ro của ô đó.
* **Khoảng cách giữa hai ô**`[i, j]` và `[i', j']` là `d = u * sqrt((i - i')^2 + (j - j')^2)` (bay thẳng giữa tâm hai ô). Không lưu riêng giá trị đi ngang hay đi chéo; mọi nhóm tính từ công thức này.
* **Độ cao UAV** không có trong instance.
* **Deadline `D`** dùng chung cho mọi UAV.

## Định dạng instance (`*.json`)

```json
{
  "name": "toy_3x3_K2",
  "source": {"dataset": "synthetic", "seed": 1},
  "grid": {"m": 3, "n": 3, "u": 40, "w": [[0.0, 0.2, 0.9], [0.1, 0.8, 0.7], [0.0, 0.1, 0.3]]},
  "K": 2,
  "D": 600,
  "eta": 0.9,
  "uavs": [
    {"depot": [0, 0], "B": 300, "v_h": 15, "e_h": 1, "tau": 2, "eps": 0},
    {"depot": [0, 0], "B": 280, "v_h": 15, "e_h": 1, "tau": 2, "eps": 0}
  ],
  "rho": {"intact": 0.0, "flooded": 0.5, "collapsed": 1.0}
}
```

### Ý nghĩa các trường


| Trường          | Ý nghĩa                                                                                                                                           | Đơn vị / ràng buộc                 |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `name`            | Tên instance, nên mã hoá kích thước,`K`, mức ngân sách, seed (xem mục đặt tên).                                                       | Chuỗi, duy nhất                       |
| `source`          | Nguồn dữ liệu để tái lập: tên dataset, ảnh gốc, seed sinh (nếu giả lập). Thuật toán không cần.                                     | Đối tượng tự do                    |
| `grid.m`,`grid.n` | Số hàng, số cột của lưới.                                                                                                                    | Số nguyên dương                     |
| `grid.u`          | Cạnh một ô. Quyết định khoảng cách thật giữa các ô. Một giá trị chung cho toàn lưới.                                              | Mét, dương                           |
| `grid.w`          | Ma trận`m x n`rủi ro. UAV được thưởng`w[i][j]`khi ghé ô`[i, j]`. Sinh từ nhãn dataset: trung bình`rho(nhãn)`theo điểm ảnh trong ô. | Số thực,`>= 0`(thường trong [0, 1]) |
| `K`               | Số UAV. Phải bằng độ dài của`uavs`.                                                                                                          | Số nguyên dương                     |
| `D`               | Deadline chung: thời điểm muộn nhất UAV đến ô quan sát cuối cùng.                                                                        | Giây, dương                          |
| `eta`             | Hệ số an toàn pin. Mỗi UAV chỉ được dùng tối đa`eta * B`năng lượng.                                                                   | Thuộc (0, 1]                           |
| `uavs[k].depot`   | Ô xuất phát và quay về`[i, j]`. Nhiều UAV có thể trùng depot.                                                                              | Nằm trong lưới                       |
| `uavs[k].B`       | Dung lượng pin của UAV`k`.                                                                                                                       | Đơn vị năng lượng, dương        |
| `uavs[k].v_h`     | Tốc độ bay ngang.                                                                                                                                | m/s, dương                            |
| `uavs[k].e_h`     | Năng lượng tiêu thụ mỗi mét bay ngang.                                                                                                       | Năng lượng/m, dương                |
| `uavs[k].tau`     | Thời gian chụp và tính toán tại mỗi ô (chi phí cố định mỗi lần ghé).                                                                 | Giây,`>= 0`                            |
| `uavs[k].eps`     | Năng lượng chụp tại mỗi ô (chi phí cố định mỗi lần ghé).                                                                              | Đơn vị năng lượng,`>= 0`          |
| `rho`             | Bảng nhãn sang rủi ro đã dùng để sinh`w`. Chỉ để tra cứu và tái lập.                                                                 | Giá trị trong [0, 1]                  |

### Điều kiện hợp lệ của instance

Schema (`docs/instance.schema.json`) kiểm tra kiểu và khoảng giá trị. Các điều kiện chéo giữa các trường cần bộ kiểm tra tự viết:

* `grid.w` có đúng `m` hàng và mỗi hàng có đúng `n` phần tử.
* `K` bằng độ dài của `uavs`.
* Mỗi `depot = [i, j]` thoả `0 <= i < m` và `0 <= j < n`.
* Mỗi UAV bay được ít nhất một ô: tồn tại một ô sao cho chuyến đi, chụp và về depot thoả cả `D` và `eta * B`. Nếu không, loại instance.

## Định dạng nghiệm (`*.solution.json`)

```json
{
  "instance": "toy_3x3_K2",
  "algorithm": "greedy_insertion",
  "runtime_sec": 0.01,
  "routes": [
    [[0, 1], [1, 1], [0, 2]],
    [[1, 0]]
  ]
}
```


| Trường      | Ý nghĩa                                                                                                                                                                                                                                    | Bắt buộc |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `instance`    | `name`của instance mà nghiệm này dành cho.                                                                                                                                                                                              | Có        |
| `routes`      | Danh sách`K`lộ trình, theo đúng thứ tự UAV trong`uavs`. Mỗi lộ trình là dãy ô`[i, j]`đi qua, theo thứ tự.**Không**gồm depot (depot được ngầm hiểu ở đầu và cuối). Lộ trình rỗng`[]`nghĩa là UAV không bay. | Có        |
| `algorithm`   | Tên thuật toán sinh ra nghiệm.                                                                                                                                                                                                           | Không     |
| `runtime_sec` | Thời gian chạy thuật toán (giây).                                                                                                                                                                                                       | Không     |

Giá trị mục tiêu **không** lưu trong file nghiệm: bộ kiểm tra luôn tự tính lại.

## Cách bộ kiểm tra xét nghiệm

Với lộ trình `(c_1, ..., c_L)` của UAV `k` và `c_0 = c_{L+1}` là depot của UAV đó, gọi `S = d(c_0, c_1) + ... + d(c_{L-1}, c_L)` là quãng đường bay đến ô cuối, và `r = d(c_L, c_0)` là quãng đường về depot. Với `L = 0` thì không có ràng buộc nào.

1. **Deadline:**`S / v_h + (L - 1) * tau <= D`.
2. **Pin:**`e_h * (S + r) + L * eps <= eta * B`.
3. **Không lặp ô** trong cùng một lộ trình.
4. **Ô hợp lệ:** mọi `[i, j]` nằm trong lưới. Ô chứa depot được phép ghé như mọi ô khác.
5. **Số lộ trình:**`routes` có đúng `K` phần tử.

So sánh số thực dùng dung sai nhỏ (ví dụ `1e-9`).

Hai UAV được phép ghé cùng một ô, nhưng ô đó chỉ được tính một lần trong mục tiêu. Vì khoảng cách là Euclid nên luôn có nghiệm tối ưu với các lộ trình rời nhau về ô; thuật toán có thể thêm ràng buộc này cho gọn.

### Các chỉ số báo cáo

* **Mục tiêu**`f` = tổng `w` của tập ô được ghé bởi ít nhất một UAV.
* **Tỉ lệ rủi ro phủ** = `f` chia cho tổng mọi `w`.
* **Cover rate**`sigma` = số ô được ghé chia cho `m * n` (để so với Song et al.).
* Với từng UAV: số ô ghé, thời gian đến ô cuối, năng lượng đã dùng so với `eta * B`.
* Thời gian chạy thuật toán.

Khi so với bài Song et al., mọi thuật toán (kể cả baseline) đều được chấm bằng cùng mảng `w` này.

## Quy ước đặt tên file

```
<nguon>_<m>x<n>_K<K>_b<mucngansach>_s<seed>.json
```

Ví dụ `rescuenet_50x50_K3_b25_s2.json`: dữ liệu RescueNet, lưới 50x50, 3 UAV, ngân sách bằng khoảng 25% chi phí quét hết lưới, seed 2. File nghiệm cùng tên, đuôi `.solution.json`, kèm tên thuật toán nếu cần (ví dụ `..._s2.greedy.solution.json`).

## Gợi ý khi sinh instance

* Chọn `D` và `B` sao cho UAV chỉ phủ được khoảng 10% đến 50% tổng rủi ro; bài toán chỉ có ý nghĩa khi ngân sách không đủ phủ hết.
* Một cách làm: tính thời gian và năng lượng cho một UAV quét zig-zag toàn lưới, đặt `D` và `B` bằng một tỉ lệ (ví dụ 10%, 25%, 50%) của giá trị đó, rồi chạy thử một thuật toán nhanh để kiểm tra tỉ lệ rủi ro phủ.
* Mỗi cấu hình nên sinh nhiều seed (ví dụ 5 đến 10).
* Bộ tái hiện Song et al. (ICRA 2022, MAEl-CPP): `u = 40`, `v_h = 15`, `tau = 2`, `eps = 0`, `e_h = 1`, `eta = 1`, `B = 40 * d_m` (với `d_m` là độ dài đường đi tối đa tính theo ô), `D` đủ lớn để không bị ràng buộc theo thời gian.
* `u` là giả định thiết kế, không suy ra từ dataset. Ghi rõ giá trị đã chọn.
* Với C2A, kiểm tra loại nhãn thực tế trước khi tính `w` (có thể không phải mask phân đoạn như RescueNet).
