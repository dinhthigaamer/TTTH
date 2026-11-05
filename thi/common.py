"""
common.py - Mô hình bài toán Deadline-constrained Multi-UAV Risk Coverage
(theo tài liệu mô hình hoá: mỗi quan sát nhìn đúng 1 ô, độ cao không đưa vào mô hình).

Chứa: Instance/UAV, tính thời gian-năng lượng của lộ trình, kiểm tra ràng buộc
(C1) deadline, (C2) pin, (C3) lộ trình hợp lệ không lặp ô, các chỉ số đánh giá (mục 10),
và bộ khung chạy "round-robin, dùng chung tập CLOSE" cho các thuật toán greedy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]
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
        assert self.w.ndim == 2 and (self.w >= 0).all(), "w phải là ma trận không âm"
        assert 0 < self.eta <= 1
        for k, a in enumerate(self.uavs):
            assert 0 <= a.depot[0] < self.m and 0 <= a.depot[1] < self.n, f"depot UAV {k} ngoài lưới"

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
    walks: Optional[List[List[Cell]]] = None       # đường bay thực lúc lập kế hoạch (gồm ô chỉ bay qua)


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


def evaluate(inst: Instance, routes: List[List[Cell]]) -> Dict:
    """Kiểm tra (C1)-(C3) và tính các chỉ số: f, tỉ lệ rủi ro phủ, cover rate σ, ΣR, pin, thời gian."""
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
        per_uav.append(dict(cells=len(r), time=T, energy=E, battery_use=E / cap if cap > 0 else 0.0))
        for c in r:
            covered[c] = True
        total_visits += len(r)
    f = float(inst.w[covered].sum())
    total = float(inst.w.sum())
    return dict(
        feasible=not violations, violations=violations,
        f=f, risk_ratio=f / total if total > 0 else 0.0,
        cover_rate=float(covered.sum()) / covered.size,     # σ
        sum_R=f,                                            # ΣR (cùng mảng w)
        cells=int(covered.sum()), duplicate_visits=total_visits - int(covered.sum()),
        per_uav=per_uav,
    )


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


# --------------------------------------------------------------------------- #
# Khung chạy nhiều UAV dùng chung tập CLOSE (các UAV thay phiên đi từng bước)
# --------------------------------------------------------------------------- #
Action = Optional[Tuple[str, Cell]]            # ("observe", cell) | ("transit", cell) | None
StepFn = Callable[[Instance, UAVState, np.ndarray], Action]


def run_shared_closed(inst: Instance, step_fn: StepFn, name: str) -> Solution:
    """Mọi UAV dùng chung CLOSE (ô đã được quan sát), nên các lộ trình rời nhau (khớp mục 5).
    Mỗi vòng, mỗi UAV còn hoạt động đi đúng 1 bước do `step_fn` quyết định; trả None thì UAV dừng."""
    closed = np.zeros((inst.m, inst.n), dtype=bool)
    states = [UAVState(inst, k) for k in range(inst.K)]
    max_rounds = 50 * inst.m * inst.n + 100          # chặn an toàn; thực tế pin đã giới hạn số bước
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
        w += rng.uniform(0.5, 1.0) * np.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2 * s * s))
    w += 0.05 * rng.random((m, n))
    w = np.clip(w / w.max(), 0, 1)
    max_cells = flight_budget_m / u
    B = eta ** -1 * (e_h * flight_budget_m + eps * max_cells)      # pin đủ cho ~flight_budget_m mét
    D = flight_budget_m / v_h + tau * max_cells
    uavs = [UAV(depot=(int(rng.integers(m)), int(rng.integers(n))), B=B, v_h=v_h, e_h=e_h, tau=tau, eps=eps)
            for _ in range(K)]
    return Instance(w=w, u=u, D=D, eta=eta, uavs=uavs)
