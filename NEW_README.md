# Multi-UAV Risk Coverage: mở rộng đa mục tiêu

Bài toán: nhiều UAV bay từ depot, quan sát các ô trên lưới `m × n` (mỗi ô có rủi ro `w_c`), chịu ràng buộc deadline (C1), pin (C2) và lộ trình không lặp ô (C3).

Bản gốc chỉ cực đại **tổng rủi ro phủ**. Phần mở rộng thêm mục tiêu thứ hai đo mức **"sớm"** khi thu thập các ô rủi ro cao:

```
fcov = Σ_{i∈U} w_i                          (độ phủ rủi ro, giữ nguyên)
fben = Σ_{i∈U∩H} w_i · Tmin_i / T_i         (thu thập ô rủi ro cao càng sớm càng tốt)
```

* `U`: tập ô được quan sát. `H`: tập ô rủi ro cao, cố định theo instance (top `p`% theo `w`, mặc định `p = 10%`).
* `T_i = F/v_h + l·τ`: thời điểm hoàn tất quan sát ô `i` (`F` là quãng bay tích luỹ, `l` là thứ tự ô, tính từ 1, không tính chặng về depot).
* `Tmin_i = min_k [ d(depot_k, i)/v_h_k + τ_k ]`: thời điểm sớm nhất có thể. Tỉ số `Tmin_i / T_i ∈ (0, 1]`.
* Khi báo cáo, hai mục tiêu được chuẩn hoá về [0, 1]: `fcov / Σw` và `fben / Σ_H w`.

Hai mục tiêu xung đột vì quét ô rủi ro thấp làm tăng `fcov` nhưng làm trễ các ô H phía sau. Mức xung đột phụ thuộc instance, nên `demo_moo.py` có bước kiểm tra riêng (xem bên dưới).

## Cấu trúc file


| File                                                  | Vai trò                                                                                                                                                           |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `common.py`                                           | Mô hình bài toán,`evaluate`, khung`run_shared_closed`,**phần mở rộng đa mục tiêu**(H, Tmin,`T[c]`,`fben`, đồng hồ`clock`, tiện ích Pareto, HV, IGD) |
| `zigzag.py`,`na_greedy.py`,`heu_greedy.py`,`svrec.py` | 4 greedy gốc,**không sửa**, chỉ chấm lại bằng`evaluate(inst, routes, H)`                                                                                    |
| `h_first.py`                                          | Greedy "đi H trước" (baseline thiên về`fben`, đồng thời là seed)                                                                                          |
| `ga.py`                                               | GA gốc`ga()`, thêm`ga_scalar`(GA đơn mục tiêu có trọng số) và`nsga2`                                                                                     |
| `aco.py`                                              | ACO gốc`aco()`, thêm`mo_aco`(MO-ACO)                                                                                                                             |
| `demo.py`                                             | Demo gốc (4 greedy, chỉ số cũ)                                                                                                                                 |
| `demo_moo.py`                                         | Demo đa mục tiêu: chấm lại baseline, kiểm tra xung đột, chạy NSGA-II / MO-ACO, báo cáo HV và IGD                                                       |

## Cài đặt

Cần Python 3.9+ và `numpy`. `matplotlib` là tuỳ chọn (chỉ để vẽ mặt trận Pareto).

```bash
pip install numpy matplotlib
```

## Cách chạy

Đặt tất cả file trong cùng một thư mục, rồi chạy:

```bash
# Kiểm tra công thức fben bằng ví dụ tính tay của đề xuất (kỳ vọng: selftest OK: 0.9 0.375)
python common.py

# Demo gốc (chỉ số cũ, không đổi so với trước)
python demo.py [seed]

# Demo đa mục tiêu, cấu hình mặc định (lưới 50x50, K=3, p=10%)
python demo_moo.py

# Độ nhạy theo p ∈ {5%, 10%, 20%}
python demo_moo.py --sens

# Chạy nhẹ để thử nhanh
python demo_moo.py --P 20 --G 15 --iters 5

# Chạy đầy đủ hơn theo gợi ý của đề xuất (P = 30-40, G = 30-50)
python demo_moo.py --P 40 --G 50 --seed 1
```

### Tham số của `demo_moo.py`


| Tham số  | Mặc định | Ý nghĩa                                                       |
| --------- | ----------- | --------------------------------------------------------------- |
| `--seed`  | 0           | Seed sinh instance và seed ngẫu nhiên của các thuật toán |
| `--m`     | 50          | Kích thước lưới (`m × m`)                                 |
| `--K`     | 3           | Số UAV                                                         |
| `--p`     | 0.10        | Tỉ lệ ô rủi ro cao (tập H)                                 |
| `--sens`  | tắt        | Chạy lần lượt p = 5%, 10%, 20%                              |
| `--P`     | 30          | Kích thước quần thể (GA, NSGA-II)                          |
| `--G`     | 30          | Số thế hệ (GA, NSGA-II)                                      |
| `--A`     | 20          | Số kiến mỗi thế hệ (MO-ACO)                                |
| `--iters` | 10          | Số thế hệ của MO-ACO                                        |
| `--out`   | `runs`      | Thư mục lưu kết quả                                        |

### Demo làm gì

1. **Chấm lại baseline**: chạy Zig-Zag, Na-Greedy, Heu-Greedy, SVReC và H-First trên cùng instance, in `fcov`, `fben` (chuẩn hoá) và số ô H được phủ. Lộ trình của mỗi lần chạy được lưu vào `runs/*.json` để chấm lại khi đổi H hoặc cách tính `Ti`.
2. **Kiểm tra xung đột (mục 3 của đề xuất)**: dùng `ga_scalar` tối ưu riêng `fcov` và riêng `fben`, rồi đánh giá chéo. Nếu mỗi nghiệm đạt trên khoảng 95% nghiệm kia ở chỉ số còn lại thì mặt trận hẹp, ngược lại thì xung đột có thật.
3. **Chạy các thuật toán đa mục tiêu**: GA có trọng số, NSGA-II và MO-ACO. Quần thể và archive được seed bằng SVReC, Na-Greedy, Heu-Greedy và H-First.
4. **Báo cáo**: Hypervolume (điểm tham chiếu (0,0)), IGD (reference front gộp từ mọi thuật toán), số điểm trên mặt trận và thời gian chạy. Hình mặt trận lưu ở `runs/front_p<p>_s<seed>.png`.

## Dùng như thư viện

```python
from common import make_random_instance, evaluate, high_risk_mask
from h_first import h_first
from ga import nsga2
from aco import mo_aco

inst = make_random_instance(m=50, n=50, K=3, seed=0)
H = high_risk_mask(inst, p=0.10)

# Chấm một lộ trình bất kỳ bằng cả hai chỉ số
sol = h_first(inst, H)
r = evaluate(inst, sol.routes, H)
print(r["feasible"], r["objs"])      # (fcov chuẩn hoá, fben chuẩn hoá)

# NSGA-II / MO-ACO trả về archive Pareto và lịch sử Hypervolume
arch, hv_hist = nsga2(inst, H, pop_size=30, n_generations=30, seed=0)
arch2, _ = mo_aco(inst, H, G=10, A=20, seed=0)
for objs, routes in arch.items:      # mỗi phần tử là ((fcov, fben), routes)
    print(objs)
```

`evaluate(inst, routes)` không truyền `H` thì hoạt động đúng như bản gốc. Truyền `H` thì kết quả có thêm `T` (ma trận `T[c]`), `f_cov`, `f_ben`, `f_cov_norm`, `f_ben_norm`, `H_covered` và `objs`.

## Điểm cần lưu ý

* **(C1)** vẫn giữ dạng `(L−1)τ` như đề xuất. Nếu đổi thành `F/v_h + Lτ ≤ D`, chỉ cần sửa `UAVState.can_observe` và `route_metrics` trong `common.py`, rồi chạy lại toàn bộ baseline.
* **`T[c]` khi chấm điểm** tính theo mô hình bay thẳng giữa các ô quan sát. Với Heu-Greedy, nơi có bước bay qua ô đã phủ, giá trị này có thể nhỏ hơn đồng hồ lúc lập kế hoạch.
* **Nhiều depot**: `Tmin_i` lấy nhỏ nhất theo mọi UAV (mỗi UAV dùng `v_h`, `τ` riêng).
* **Một lần chạy chưa đủ để kết luận** về mức xung đột. Nên chạy nhiều seed (`--seed`) và dùng `--sens`, rồi gộp các điểm để dựng reference front cho IGD.
