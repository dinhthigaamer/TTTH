# Mô tả chi tiết thuật toán ACO (`aco.py`) và GA (`ga.py`)

*(Tài liệu viết theo đúng cấu trúc và ký hiệu trong `Mô tả thuật toán.docx` của bạn Thi để cả nhóm đọc hiểu và đối chiếu trực tiếp với mã nguồn).*

Ký hiệu dùng chung (từ `common.py`):
- Lưới $m \times n$ ô ($N = m \cdot n$), cạnh ô $u$ (mét), trọng số rủi ro $w_c \ge 0$.
- $K$ UAV; mỗi UAV có trạm xuất phát/về `depot`, pin $B$, hệ số an toàn pin $\eta \in (0, 1]$, tốc độ bay $v_h$, năng lượng bay mỗi mét $e_h$, thời gian chụp mỗi ô $\tau$, năng lượng chụp $\epsilon$.
- $S \le \frac{\eta B}{e_h u}$ là số bước bay tối đa của một UAV.
- Ba ràng buộc: **(C1) Deadline** $T_L = \frac{F}{v_h} + (L-1)\tau \le D$; **(C2) Pin** $E = e_h(F + d_{\text{về depot}}) + L\epsilon \le \eta B$; **(C3) Không lặp ô** (dùng chung mảng `closed` trong `run_shared_closed`).

---

## Phần III. Mô tả thuật toán đàn kiến ACO (`aco.py`)

### 1. Ý tưởng và ánh xạ từ bài báo Cho et al. (2026)
- **Nguồn bài báo**: Cho et al. (2026), *"Coverage path planning for multiple UAVs in prioritized regions using ant colony optimization"*, Mục 4.2.2, **Algorithm 3-2**, các phương trình **(18)–(23)**.
- **Ý tưởng**: Các thuật toán tham lam đơn lẻ (`Na-Greedy`, `Heu-Greedy`, `SVReC`) chỉ xây dựng **1 lộ trình duy nhất** nên một quyết định sai ở bước đầu sẽ làm kẹt UAV về sau. ACO cho **$A$ con kiến qua $G$ thế hệ** cùng xây dựng lộ trình trên khung `run_shared_closed`. Mỗi con kiến chọn ô dựa trên sự kết hợp giữa:
  1. **Thông tin Heuristic $\eta(i, j)$**: Đánh giá lợi ích tức thời (rủi ro cao, thời gian bay ngắn).
  2. **Vết mùi Pheromone $\tau$**: Kinh nghiệm tích lũy của cả đàn kiến từ các lộ trình có tổng rủi ro phủ cao nhất ở các thế hệ trước.

### 2. Bốn thành phần chính trong `aco.py`

#### 2.1 Tập ứng viên $\Pi_a$ và xử lý kẹt (Deadlock Handling – Assumption 3, Alg. 3-2 dòng 13–17)
- **Bước lân cận**: Từ vị trí hiện tại `s.cur`, kiến lọc các ô 8-lân cận chưa phủ và thỏa mãn đồng thời (C1), (C2):
  $$\Pi_a = \{c \in N_8(\text{s.cur}) : \neg \text{closed}[c] \land \text{s.can\_observe}(c)\}$$
- **Xử lý Deadlock**: Khi $\Pi_a = \emptyset$ (mọi ô xung quanh đều đã phủ hoặc không đủ pin/deadline), thay vì dừng ngay như `Na-Greedy`, kiến được phép nhảy sang các ô chưa phủ có $w_c > \text{mean}(w)$ gần vị trí hiện tại nhất mà thỏa `s.can_observe(c)` (lấy tối đa 8 ứng viên gần nhất đưa vào $\Pi_a$). Nếu vẫn rỗng thì UAV đó mới dừng (`return None`).

#### 2.2 Thông tin Heuristic $\eta(i, j)$ và Quy tắc chọn ô (Eq. 18 – Eq. 20)
- **Heuristic (Eq. 20 điều chỉnh có trọng số rủi ro)**: Kết hợp mức rủi ro $w_c$ và thời gian di chuyển + chụp $T_{i,c} = \frac{d(\text{s.cur}, c)}{v_h} + \tau$:
  $$\eta(\text{s.cur}, c) = \frac{w_c + 10^{-3}}{d(\text{s.cur}, c)/v_h + \tau}$$
- **Điểm hấp dẫn của mỗi ứng viên $c \in \Pi_a$**:
  $$\text{desirability}(c) = [\tau(c)]^\alpha \cdot [\eta(\text{s.cur}, c)]^\beta$$
- **Quy tắc chuyển trạng thái (Eq. 18 & Eq. 19)**: Sinh số ngẫu nhiên $q \in [0, 1]$:
  - Nếu $q \le q_0$ (*Khai thác - Exploitation*): Chọn ô có $\text{desirability}(c)$ lớn nhất.
  - Nếu $q > q_0$ (*Khám phá - Roulette Wheel Selection*): Chọn ngẫu nhiên ô $c \in \Pi_a$ với xác suất:
    $$p(c) = \frac{\text{desirability}(c)}{\sum_{k \in \Pi_a} \text{desirability}(k)}$$

#### 2.3 Cập nhật Pheromone cục bộ (Local Update – Eq. 21)
- Ngay sau khi một con kiến chọn quan sát ô `chosen`, mùi pheromone tại ô đó bị làm giảm nhẹ:
  $$\tau[\text{chosen}] \leftarrow (1 - \rho)\cdot \tau[\text{chosen}] + \rho \cdot \tau_0$$
- **Tác dụng**: Giảm nhẹ độ hấp dẫn của ô vừa đi qua đối với các con kiến kế tiếp trong cùng thế hệ, giúp đàn kiến tỏa ra thử nhiều nhánh đường bay khác nhau.

#### 2.4 Cập nhật Pheromone toàn cục (Global Update – Eq. 22 & Eq. 23)
- Sau khi cả $A$ con kiến trong thế hệ hoàn thành lộ trình, thuật toán lưu lại lộ trình tốt nhất từ đầu đến hiện tại là `best_routes` (có tổng rủi ro `best_f` cao nhất).
- Bay hơi toàn cục và bồi đắp thêm mùi pheromone lên đúng các ô thuộc `best_routes`:
  $$\tau \leftarrow (1 - \epsilon)\cdot \tau, \qquad \tau[c] \leftarrow \tau[c] + \epsilon \cdot \Delta\tau \quad (\forall c \in \text{best\_routes})$$
  với $\Delta\tau = \tau_0 \cdot \left(1 + \frac{\text{best\_f}}{\sum w}\right)$.

### 3. Pseudocode (`aco.py`)
```text
khởi tạo τ[i, j] ← τ_0 = 1.0 với mọi ô; best_f ← -1, best_routes ← rỗng
lặp g = 1 .. G (mặc định G = 10 thế hệ):
    lặp a = 1 .. A (mặc định A = 20 con kiến):
        chạy run_shared_closed cho K UAV với bước đi make_aco_step:
            Π_a ← {c ∈ N8(s.cur) : c chưa phủ ∧ s.can_observe(c)}
            nếu Π_a = ∅:
                Π_a ← tối đa 8 ô chưa phủ có w > mean(w), khả thi, gần s.cur nhất
            nếu Π_a = ∅: UAV s dừng
            với mỗi c ∈ Π_a: score(c) ← τ[c]^α · ((w[c] + 1e-3) / (dist(s.cur, c)/v_h + uav.tau))^β
            nếu rand() ≤ q_0: chọn c* = argmax score(c)
            ngược lại: chọn c* theo Roulette Wheel với p(c) = score(c) / Σ score
            τ[c*] ← (1 − ρ)·τ[c*] + ρ·τ_0       # cập nhật cục bộ (Eq. 21)
            quan sát c*
        f_a ← evaluate(routes).f
        nếu f_a > best_f: best_f ← f_a; best_routes ← routes
    τ ← (1 − ε)·τ                               # bay hơi toàn cục (Eq. 22)
    với mỗi c ∈ best_routes: τ[c] ← τ[c] + ε·Δτ  # bồi đắp nghiệm tốt nhất (Eq. 23)
trả về Solution("ACO", best_routes)
```

### 4. Độ phức tạp của `aco.py`

| Thành phần | Chi phí mỗi lần gọi | Ghi chú |
| :--- | :---: | :--- |
| Lọc 8 ô lân cận $\Pi_a$ + tính `desirability` + chọn ô | $O(1)$ | Tối đa 8 ứng viên $\times O(1)$ phép tính |
| Cập nhật Pheromone cục bộ (Eq. 21) | $O(1)$ | Gán trực tiếp phần tử mảng |
| Xử lý Deadlock (khi $\Pi_a = \emptyset$) | $O(N \log N)$ | `argwhere` $O(N)$ + `argsort` khoảng cách $O(N \log N)$ |
| Một con kiến (`run_shared_closed` + `evaluate`) | $O(N + K \cdot S)$ đến $O(\min(N, K \cdot S) \cdot N \log N)$ | Thường bước 8-lân cận chiếm đa số, chỉ nhảy khi kẹt |
| Cập nhật Pheromone toàn cục mỗi thế hệ | $O(N + K \cdot S)$ | Nhân ma trận $\tau$ kích thước $N$ và cộng trên `best_routes` |

- **Tổng thời gian**:
  - Thông thường (ít khi kẹt Deadlock): **$O(G \cdot A \cdot (N + K \cdot S))$**.
  - Trường hợp xấu nhất (mọi bước đều kẹt phải nhảy xa): **$O(G \cdot A \cdot \min(N, K \cdot S) \cdot N \log N)$**.
- **Bộ nhớ**: **$O(N)$** cho ma trận pheromone $\tau$ và mảng `closed`.

---

## Phần IV. Mô tả thuật toán di truyền GA (`ga.py`)

### 1. Ý tưởng: Biểu diễn gián tiếp qua bản đồ gen ưu tiên + Bộ giải mã khả thi (`_decode`)
- **Tại sao không mã hóa trực tiếp danh sách ô vào nhiễm sắc thể?**
  Nếu nối trực tiếp danh sách các ô của $K$ UAV thành chuỗi rồi cắt ghép (crossover) hoặc đổi chỗ (mutation), cá thể con sinh ra sẽ lập tức bị **trùng lặp ô hoặc vượt quá pin $\eta B$ và deadline $D$**.
- **Cách giải quyết trong `ga.py`**:
  - Mỗi cá thể (Chromosome) trong quần thể là một **ma trận trọng số ưu tiên giả lập `prio` kích thước $m \times n$**.
  - Khi đánh giá một cá thể, hàm `_decode(inst, prio, r_th)` dùng chính khung `run_shared_closed` và `UAVState.can_observe` trong `common.py` nhưng đi theo sự dẫn đường của `prio` thay vì `w`.
  - Nhờ vậy, **100% cá thể con sau lai ghép và đột biến luôn thỏa mãn tuyệt đối cả (C1), (C2) và (C3)**! Khi ma trận `prio` tiến hóa qua các thế hệ, nó học được cách "nâng" trọng số của các ô nằm trên hành lang dẫn tới cụm rủi ro lớn và "hạ" trọng số của các ô gây mắc bẫy *Weight Trap*.

### 2. Các thành phần chính trong `ga.py`

1. **Khởi tạo quần thể (`pop_size = 12`)**:
   - Lấy nghiệm tốt nhất giữa `svrec(inst)` và `na_greedy(inst)` làm mốc khởi đầu `best_routes`.
   - Cá thể đầu tiên `pop[0]` lấy bằng ma trận `w`, trong đó các ô thuộc `best_routes` được nhân hệ số `1.35` (bơm tri thức từ SVReC vào quần thể để đảm bảo GA luôn có chất lượng $\ge$ SVReC).
   - 11 cá thể còn lại được tạo bằng cách nhân `w` với nhiễu ngẫu nhiên `uniform(0.7, 1.3)` để tạo đa dạng hướng tìm kiếm.
2. **Hàm giải mã `_decode(inst, prio, r_th)`**:
   - Tại mỗi bước của UAV `s`, lấy các ô 8-lân cận chưa phủ và khả thi `cands`.
   - Nếu `cands` rỗng hoặc trung bình rủi ro thật `mean(w[cands]) < r_th`, bổ sung tối đa 6 ô chưa phủ có `w > r_th` gần `s.cur` nhất mà khả thi (giúp UAV nhảy qua vùng trũng).
   - Chọn ô tiếp theo tối đa hóa tỷ số giữa gen ưu tiên và khoảng cách:
     $$\text{best} = \arg\max_{c \in \text{cands}} \frac{\text{prio}[c]}{\text{dist}(\text{s.cur}, c) / u}$$
   - Sau khi `run_shared_closed` kết thúc, độ thích nghi (Fitness) của cá thể được tính trên **mảng rủi ro thật `inst.w`**: `f_val = evaluate(inst, routes)["f"]`.
3. **Chọn lọc, Lai ghép và Đột biến qua `n_generations = 12` thế hệ**:
   - **Elitism**: Giữ nguyên 2 cá thể có fitness cao nhất sang thế hệ mới.
   - **Chọn lọc giải đấu (Binary Tournament Selection)**: Bốc ngẫu nhiên 2 cá thể, chọn cá thể có fitness cao hơn làm cha `p1` (tương tự cho mẹ `p2`).
   - **Lai ghép đồng nhất (Uniform Crossover, xác suất `0.85`)**: Với mỗi ô $(i, j)$, tung đồng xu 50/50 để lấy giá trị `prio[i, j]` từ `p1` hoặc `p2`.
   - **Đột biến nhiễu tỷ lệ (Multiplicative Mutation, xác suất `0.40`)**: Chọn ngẫu nhiên 15% số ô trên ma trận `child` và nhân với hệ số ngẫu nhiên trong `[0.6, 1.5]`.

### 3. Pseudocode (`ga.py`)
```text
r_th ← mean(w)
best_routes ← max(svrec(inst), na_greedy(inst)) theo hàm mục tiêu f
khởi tạo pop[0] ← w (nhân 1.35 tại các ô thuộc best_routes)
khởi tạo pop[1..P-1] ← w * uniform(0.7, 1.3)
đánh giá fit[i] ← evaluate(_decode(inst, pop[i], r_th)).f với mọi i = 0..P-1
lặp gen = 1 .. G_ga (mặc định 12 thế hệ):
    new_pop ← [2 cá thể có fit cao nhất trong pop]   # Elitism
    khi |new_pop| < P:
        p1 ← TournamentSelect(pop, fit); p2 ← TournamentSelect(pop, fit)
        nếu rand() < crossover_prob: child ← UniformCrossover(p1, p2)
        ngược lại: child ← bản sao của p1
        nếu rand() < mutation_prob:
            nhân 15% số ô ngẫu nhiên của child với uniform(0.6, 1.5)
        routes ← _decode(inst, child, r_th)
        f_c ← evaluate(inst, routes).f
        nếu f_c > best_f: best_f ← f_c; best_routes ← routes
        thêm child vào new_pop
    pop ← new_pop
trả về Solution("GA", best_routes)
```

### 4. Độ phức tạp của `ga.py`
- **Mỗi lần giải mã `_decode`**: Tương đương một lần chạy biến thể SVReC, tốn $O(N + K \cdot S)$ khi bước cục bộ và tối đa $O(\min(N, K \cdot S) \cdot N \log N)$ nếu mọi bước đều nhảy.
- **Lai ghép + Đột biến**: Thao tác ma trận trên $N$ phần tử tốn $O(N)$.
- **Tổng thời gian sau $G_{\text{ga}}$ thế hệ với quần thể $P$**:
  - Thông thường: **$O(G_{\text{ga}} \cdot P \cdot (N + K \cdot S))$**.
  - Trường hợp xấu nhất: **$O(G_{\text{ga}} \cdot P \cdot \min(N, K \cdot S) \cdot N \log N)$**.
- **Bộ nhớ**: **$O(P \cdot N)$** để lưu $P$ ma trận `prio` kích thước $m \times n$.

