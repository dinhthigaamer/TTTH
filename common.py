"""
common.py - Mô hình bài toán Deadline-constrained Multi-UAV Risk Coverage
(theo tài liệu mô hình hoá: mỗi quan sát nhìn đúng 1 ô, độ cao không đưa vào mô hình).

Chứa: Instance/UAV, tính thời gian-năng lượng của lộ trình, kiểm tra ràng buộc
(C1) deadline, (C2) pin, (C3) lộ trình hợp lệ không lặp ô, các chỉ số đánh giá (mục 10),
và bộ khung chạy "round-robin, dùng chung tập CLOSE" cho các thuật toán greedy.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

Cell = Tuple[int, int]
Objs = Tuple[float, float]
TOL = 1e-9


# --------------------------------------------------------------------------- #
# Dữ liệu bài toán
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UAV:
    depot: Cell      # s_k
    B: float         # dung lượng pin B_k
    v_h: float       # tốc độ bay ngang
    e_h: float       # năng lượng / mét bay ngang
    tau: float       # thời gian chụp + tính toán mỗi ô
    eps: float       # năng lượng chụp mỗi ô


@dataclass
class Instance:
    w: np.ndarray            # mảng rủi ro w_c, shape (m, n), w >= 0
    u: float                 # cạnh mỗi ô (mét)
    D: float                 # deadline chung
    eta: float               # hệ số an toàn pin, (0, 1]
    uavs: List[UAV]

    def __post_init__(self):
        self.w = np.asarray(self.w, dtype=float)
        assert self.w.ndim == 2 and (
            self.w >= 0).all(), "w phải là ma trận không âm"
        assert 0 < self.eta <= 1
        for k, a in enumerate(self.uavs):
            assert 0 <= a.depot[0] < self.m and 0 <= a.depot[
                1] < self.n, f"depot UAV {k} ngoài lưới"

    @property
    def m(self) -> int:
        return self.w.shape[0]

    @property
    def n(self) -> int:
        return self.w.shape[1]

    @property
    def K(self) -> int:
        return len(self.uavs)

    def dist(self, a: Cell, b: Cell) -> float:
        """d(a, b) = u * sqrt((i-i')^2 + (j-j')^2)"""
        return self.u * math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class Solution:
    name: str
    routes: List[List[Cell]]                       # R_k: dãy ô được quan sát
    # đường bay thực lúc lập kế hoạch (gồm ô chỉ bay qua)
    walks: Optional[List[List[Cell]]] = None


def neighbors8(cell: Cell, m: int, n: int) -> Iterator[Cell]:
    """8 ô lân cận nằm trong lưới."""
    i, j = cell
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            a, b = i + di, j + dj
            if 0 <= a < m and 0 <= b < n:
                yield (a, b)


# --------------------------------------------------------------------------- #
# Thời gian / năng lượng của lộ trình (mục 4) và đánh giá (mục 5, 10)
# --------------------------------------------------------------------------- #
def route_metrics(inst: Instance, k: int, route: List[Cell]) -> Tuple[float, float]:
    """Trả về (T_L, E(R_k)). Lộ trình rỗng: (0, 0).
    T_L = sum t(c_{q-1}, c_q) + (L-1) tau ;  E = e_h * (tổng quãng bay + quãng về depot) + L * eps."""
    if not route:
        return 0.0, 0.0
    a = inst.uavs[k]
    cur, flight = a.depot, 0.0
    for c in route:
        flight += inst.dist(cur, c)
        cur = c
    L = len(route)
    T = flight / a.v_h + (L - 1) * a.tau
    E = a.e_h * (flight + inst.dist(cur, a.depot)) + L * a.eps
    return T, E


# --------------------------------------------------------------------------- #
# Mục tiêu thứ hai fben (mở rộng đa mục tiêu): tập H, Tmin, thời điểm quan sát Ti
#   fben = sum_{i in U∩H} w_i * Tmin_i / T_i ;  T_i = F/v_h + l*tau (l tính từ 1)
#   Tmin_i = min_k d(depot_k, i)/v_h_k + tau_k ;  H = top p% ô theo w
# --------------------------------------------------------------------------- #
def high_risk_mask(inst: Instance, p: float = 0.10) -> np.ndarray:
    """Mặt nạ bool (m, n): top p% ô theo w (hoà điểm: theo chỉ số, ổn định)."""
    N = inst.m * inst.n
    h = max(1, int(math.ceil(p * N)))
    order = np.argsort(-inst.w.ravel(), kind="stable")[:h]
    mask = np.zeros(N, dtype=bool)
    mask[order] = True
    return mask.reshape(inst.m, inst.n)


def t_min_map(inst: Instance) -> np.ndarray:
    """Tmin[c] = min_k d(depot_k, c)/v_h_k + tau_k."""
    ii, jj = np.mgrid[0:inst.m, 0:inst.n]
    best = np.full((inst.m, inst.n), np.inf)
    for a in inst.uavs:
        t = inst.u * np.hypot(ii - a.depot[0], jj - a.depot[1]) / a.v_h + a.tau
        best = np.minimum(best, t)
    return best


def cell_times(inst: Instance, k: int, route: List[Cell]) -> List[float]:
    """T_i của từng ô trong lộ trình (bay thẳng giữa các ô quan sát): T = F/v_h + l*tau, l = 1, 2, ..."""
    a = inst.uavs[k]
    cur, F, out = a.depot, 0.0, []
    for l, c in enumerate(route, start=1):
        F += inst.dist(cur, c)
        cur = c
        out.append(F / a.v_h + l * a.tau)
    return out


def evaluate(inst: Instance, routes: List[List[Cell]], H: Optional[np.ndarray] = None) -> Dict:
    """Kiểm tra (C1)-(C3) và tính các chỉ số: f, tỉ lệ rủi ro phủ, cover rate σ, ΣR, pin, thời gian.
    Nếu truyền H (mặt nạ ô rủi ro cao) thì trả thêm T[c], f_cov, f_ben, f_cov_norm, f_ben_norm,
    H_covered và objs = (f_cov_norm, f_ben_norm) (cùng cực đại hoá). Không truyền H: hành vi như cũ."""
    violations: List[str] = []
    covered = np.zeros((inst.m, inst.n), dtype=bool)
    per_uav = []
    total_visits = 0
    for k, r in enumerate(routes):
        if len(set(r)) != len(r):
            violations.append(f"UAV {k}: lặp ô trong lộ trình (C3)")
        for c in r:
            if not (0 <= c[0] < inst.m and 0 <= c[1] < inst.n):
                violations.append(f"UAV {k}: ô {c} ngoài lưới (C3)")
        T, E = route_metrics(inst, k, r)
        cap = inst.eta * inst.uavs[k].B
        if T > inst.D + TOL:
            violations.append(f"UAV {k}: T_L={T:.2f} > D={inst.D:.2f} (C1)")
        if E > cap + TOL:
            violations.append(f"UAV {k}: E={E:.2f} > eta*B={cap:.2f} (C2)")
        per_uav.append(dict(cells=len(r), time=T, energy=E,
                       battery_use=E / cap if cap > 0 else 0.0))
        for c in r:
            covered[c] = True
        total_visits += len(r)
    f = float(inst.w[covered].sum())
    total = float(inst.w.sum())
    res = dict(
        feasible=not violations, violations=violations,
        f=f, risk_ratio=f / total if total > 0 else 0.0,
        cover_rate=float(covered.sum()) / covered.size,     # σ
        sum_R=f,                                            # ΣR (cùng mảng w)
        cells=int(covered.sum()), duplicate_visits=total_visits - int(covered.sum()),
        per_uav=per_uav,
    )
    if H is not None:
        # T[c]: thời điểm hoàn tất quan sát ô c
        T = np.full((inst.m, inst.n), np.nan)
        for k, r in enumerate(routes):
            for c, t in zip(r, cell_times(inst, k, r)):
                if np.isnan(T[c]) or t < T[c]:
                    T[c] = t
        tmin = t_min_map(inst)
        sel = H & ~np.isnan(T)
        f_ben = float((inst.w[sel] * tmin[sel] / T[sel]).sum())
        ben_max = float(inst.w[H].sum())
        f_ben_norm = f_ben / ben_max if ben_max > 0 else 0.0
        res.update(T=T, f_cov=f, f_cov_norm=res["risk_ratio"], f_ben=f_ben, f_ben_norm=f_ben_norm,
                   H_covered=int(sel.sum()), objs=(res["risk_ratio"], f_ben_norm))
    return res


# --------------------------------------------------------------------------- #
# Trạng thái 1 UAV trong lúc lập kế hoạch cục bộ (greedy)
# --------------------------------------------------------------------------- #
class UAVState:
    """Theo dõi quãng đường đã bay `flight` (có thể gồm cả các bước chỉ bay qua ô đã phủ),
    danh sách ô quan sát `route` và đường bay `walk`.

    Kiểm tra khả thi dùng quãng bay thực `flight` nên luôn bảo thủ: tính theo mô hình (bay thẳng
    giữa các ô quan sát) chi phí chỉ nhỏ hơn hoặc bằng, nhờ bất đẳng thức tam giác.
    """

    def __init__(self, inst: Instance, k: int):
        self.inst, self.k, self.uav = inst, k, inst.uavs[k]
        self.cur: Cell = self.uav.depot
        self.flight = 0.0
        self.route: List[Cell] = []
        self.walk: List[Cell] = []
        self.active = True

    def _energy(self, flight: float, cell: Cell, n_obs: int) -> float:
        a = self.uav
        return a.e_h * (flight + self.inst.dist(cell, a.depot)) + n_obs * a.eps

    def can_observe(self, cell: Cell) -> bool:
        """Quan sát `cell` (bay thẳng từ vị trí hiện tại) mà vẫn thoả (C1) và (C2) kể cả chặng về depot."""
        a, inst = self.uav, self.inst
        flight = self.flight + inst.dist(self.cur, cell)
        L = len(self.route) + 1
        if flight / a.v_h + (L - 1) * a.tau > inst.D + TOL:
            return False
        return self._energy(flight, cell, L) <= inst.eta * a.B + TOL

    def can_transit(self, cell: Cell) -> bool:
        """Bay tới `cell` mà không quan sát (chỉ cần còn đủ pin để về depot)."""
        flight = self.flight + self.inst.dist(self.cur, cell)
        return self._energy(flight, cell, len(self.route)) <= self.inst.eta * self.uav.B + TOL

    def observe(self, cell: Cell) -> None:
        self.flight += self.inst.dist(self.cur, cell)
        self.cur = cell
        self.route.append(cell)
        self.walk.append(cell)

    def transit(self, cell: Cell) -> None:
        self.flight += self.inst.dist(self.cur, cell)
        self.cur = cell
        self.walk.append(cell)


def clock(s: UAVState) -> float:
    """Đồng hồ t_k của UAV: thời điểm hoàn tất ô vừa quan sát (đã gồm quãng bay thực, kể cả bay qua)."""
    return s.flight / s.uav.v_h + len(s.route) * s.uav.tau


# --------------------------------------------------------------------------- #
# Khung chạy nhiều UAV dùng chung tập CLOSE (các UAV thay phiên đi từng bước)
# --------------------------------------------------------------------------- #
# ("observe", cell) | ("transit", cell) | None
Action = Optional[Tuple[str, Cell]]
StepFn = Callable[[Instance, UAVState, np.ndarray], Action]


def run_shared_closed(inst: Instance, step_fn: StepFn, name: str) -> Solution:
    """Mọi UAV dùng chung CLOSE (ô đã được quan sát), nên các lộ trình rời nhau (khớp mục 5).
    Mỗi vòng, mỗi UAV còn hoạt động đi đúng 1 bước do `step_fn` quyết định; trả None thì UAV dừng."""
    closed = np.zeros((inst.m, inst.n), dtype=bool)
    states = [UAVState(inst, k) for k in range(inst.K)]
    # chặn an toàn; thực tế pin đã giới hạn số bước
    max_rounds = 50 * inst.m * inst.n + 100
    for _ in range(max_rounds):
        if not any(s.active for s in states):
            break
        for s in states:
            if not s.active:
                continue
            act = step_fn(inst, s, closed)
            if act is None:
                s.active = False
                continue
            kind, cell = act
            if kind == "observe":
                s.observe(cell)
                closed[cell] = True
            else:
                s.transit(cell)
    return Solution(name, [s.route for s in states], [s.walk for s in states])


# --------------------------------------------------------------------------- #
# Lưu/đọc lộ trình, tiện ích Pareto (cực đại hoá), tiện ích chọn đích nhảy
# --------------------------------------------------------------------------- #
def save_routes(path: str, name: str, routes: List[List[Cell]]) -> None:
    """Lưu lộ trình để chấm lại khi đổi H hoặc quy ước Ti."""
    with open(path, "w") as f:
        json.dump({"name": name, "routes": [
                  [list(c) for c in r] for r in routes]}, f)


def load_routes(path: str) -> List[List[Cell]]:
    with open(path) as f:
        return [[(int(c[0]), int(c[1])) for c in r] for r in json.load(f)["routes"]]


# --------------------------------------------------------------------------- #
# Pareto (cực đại hoá cả hai mục tiêu)
# --------------------------------------------------------------------------- #
def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def fast_nds(objs: np.ndarray) -> List[List[int]]:
    """Xếp hạng không bị chi phối (NSGA-II). Trả về danh sách các mặt F0, F1, ..."""
    n = len(objs)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                cnt[p] += 1
        if cnt[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(objs: np.ndarray) -> np.ndarray:
    """Crowding distance của một tập điểm (coi như một mặt)."""
    n, d = objs.shape
    cd = np.zeros(n)
    if n <= 2:
        cd[:] = np.inf
        return cd
    for j in range(d):
        o = np.argsort(objs[:, j], kind="stable")
        cd[o[0]] = cd[o[-1]] = np.inf
        span = objs[o[-1], j] - objs[o[0], j]
        if span <= 0:
            continue
        cd[o[1:-1]] += (objs[o[2:], j] - objs[o[:-2], j]) / span
    return cd


class ParetoArchive:
    """Archive nghiệm không bị chi phối, cắt bằng crowding distance khi vượt `cap`."""

    def __init__(self, cap: int = 40):
        self.cap = cap
        self.items: List[Tuple[Objs, List[List[Cell]]]] = []

    def add(self, objs: Objs, routes: List[List[Cell]]) -> bool:
        objs = (float(objs[0]), float(objs[1]))
        for o, _ in self.items:
            if o == objs or dominates(o, objs):
                return False
        self.items = [(o, r) for o, r in self.items if not dominates(objs, o)]
        self.items.append((objs, [list(r) for r in routes]))
        while len(self.items) > self.cap:
            cd = crowding(np.array([o for o, _ in self.items]))
            self.items.pop(int(np.argmin(cd)))
        return True

    def points(self) -> np.ndarray:
        return np.array([o for o, _ in self.items]) if self.items else np.zeros((0, 2))


def pareto_filter(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    keep = [i for i in range(len(points))
            if not any(dominates(points[j], points[i]) for j in range(len(points)) if j != i)]
    return np.unique(points[keep], axis=0)


def hypervolume_2d(points: np.ndarray, ref: Objs = (0.0, 0.0)) -> float:
    """Hypervolume 2 mục tiêu (cực đại), điểm tham chiếu `ref` (mặc định gốc toạ độ)."""
    if len(points) == 0:
        return 0.0
    pts = pareto_filter(np.asarray(points, dtype=float))
    pts = pts[np.argsort(-pts[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y > prev_y and x > ref[0]:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return float(hv)


def igd(front: np.ndarray, ref_front: np.ndarray) -> float:
    """IGD: trung bình khoảng cách từ mỗi điểm reference tới điểm gần nhất của `front` (nhỏ hơn = tốt)."""
    if len(front) == 0 or len(ref_front) == 0:
        return float("inf")
    d = np.linalg.norm(ref_front[:, None, :] - front[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


# --------------------------------------------------------------------------- #
# Tiện ích chọn đích cho bước nhảy / xử lý kẹt (dùng chung greedy mới, ACO, GA)
# --------------------------------------------------------------------------- #
def jump_target(inst: Instance, s: UAVState, closed: np.ndarray, score: np.ndarray) -> Optional[Cell]:
    """Ô chưa phủ khả thi (C1, C2) có score/(1+khoảng cách ô) lớn nhất."""
    idx = np.argwhere(~closed)
    if len(idx) == 0:
        return None
    d = np.hypot(idx[:, 0] - s.cur[0], idx[:, 1] - s.cur[1])
    sc = (score[idx[:, 0], idx[:, 1]] + 1e-12) / (1.0 + d)
    for o in np.argsort(-sc, kind="stable"):
        c = (int(idx[o, 0]), int(idx[o, 1]))
        if s.can_observe(c):
            return c
    return None


def nearest_uncovered_H(cur: Cell, closed: np.ndarray, H: np.ndarray) -> Optional[Cell]:
    idx = np.argwhere(H & ~closed)
    if len(idx) == 0:
        return None
    d = np.hypot(idx[:, 0] - cur[0], idx[:, 1] - cur[1])
    o = np.lexsort((idx[:, 1], idx[:, 0], d))[0]
    return (int(idx[o, 0]), int(idx[o, 1]))


# --------------------------------------------------------------------------- #
# Sinh instance giả lập
# --------------------------------------------------------------------------- #
def make_random_instance(m: int = 30, n: int = 30, K: int = 3, seed: int = 0, u: float = 40.0,
                         flight_budget_m: float = 1500.0, v_h: float = 15.0, e_h: float = 10.0,
                         tau: float = 2.0, eps: float = 5.0, eta: float = 0.9) -> Instance:
    """Bản đồ rủi ro gồm vài đốm Gaussian + nhiễu; ngân sách chọn để mỗi UAV chỉ phủ một phần."""
    rng = np.random.default_rng(seed)
    ii, jj = np.mgrid[0:m, 0:n]
    w = np.zeros((m, n))
    for _ in range(max(3, (m * n) // 150)):
        ci, cj = rng.uniform(0, m), rng.uniform(0, n)
        s = rng.uniform(0.04, 0.12) * max(m, n)
        w += rng.uniform(0.5, 1.0) * np.exp(-((ii - ci) **
                                              2 + (jj - cj) ** 2) / (2 * s * s))
    w += 0.05 * rng.random((m, n))
    w = np.clip(w / w.max(), 0, 1)
    max_cells = flight_budget_m / u
    # pin đủ cho ~flight_budget_m mét
    B = eta ** -1 * (e_h * flight_budget_m + eps * max_cells)
    D = flight_budget_m / v_h + tau * max_cells
    uavs = [UAV(depot=(int(rng.integers(m)), int(rng.integers(n))), B=B, v_h=v_h, e_h=e_h, tau=tau, eps=eps)
            for _ in range(K)]
    return Instance(w=w, u=u, D=D, eta=eta, uavs=uavs)


# Kiểm tra bằng ví dụ tính tay của đề xuất (v_h=5, u=10, tau=3, w=0.9 cách depot 1 bước)
def _selftest() -> None:
    w = np.full((3, 3), 0.1)
    w[1, 0] = 0.9
    uav = UAV(depot=(1, 1), B=1e9, v_h=5.0, e_h=1.0, tau=3.0, eps=0.0)
    inst = Instance(w=w, u=10.0, D=1e9, eta=1.0, uavs=[uav])
    H = np.zeros((3, 3), dtype=bool)
    H[1, 0] = True
    direct = evaluate(inst, [[(1, 0)]], H)
    # T = Tmin = 5 s
    assert abs(direct["f_ben"] - 0.9) < 1e-9, direct["f_ben"]
    # F = 30 m, l = 2 -> T = 12 s
    detour = evaluate(inst, [[(1, 2), (1, 0)]], H)
    assert abs(detour["f_ben"] - 0.9 * 5 / 12) < 1e-9, detour["f_ben"]
    print("selftest OK:", direct["f_ben"], detour["f_ben"])


if __name__ == "__main__":
    _selftest()
