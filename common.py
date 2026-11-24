"""
common.py - Mô hình bài toán Deadline-constrained Multi-UAV Risk Coverage
(theo tài liệu mô hình hoá: mỗi quan sát nhìn đúng 1 ô, độ cao không đưa vào mô hình).

Chứa: Instance/UAV, tính thời gian-năng lượng của lộ trình, kiểm tra ràng buộc
(C1) deadline, (C2) pin, (C3) lộ trình hợp lệ không lặp ô, các chỉ số đánh giá (mục 10),
bộ khung chạy "round-robin, dùng chung tập CLOSE" cho các thuật toán, và đọc/ghi JSON.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

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
    name: str = "synthetic"
    source: Optional[Dict[str, Any]] = None
    rho: Optional[Dict[str, float]] = None

    def __post_init__(self):
        self.w = np.asarray(self.w, dtype=float)
        assert self.w.ndim == 2 and not np.isnan(self.w).any() and (self.w >= 0).all(), "w phải là ma trận không âm"
        assert self.u > 0 and self.D > 0 and 0 < self.eta <= 1 and len(self.uavs) > 0
        for k, a in enumerate(self.uavs):
            assert 0 <= a.depot[0] < self.m and 0 <= a.depot[1] < self.n, f"depot UAV {k} ngoài lưới"
            assert a.B > 0 and a.v_h > 0 and a.e_h > 0 and a.tau >= 0 and a.eps >= 0

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
    runtime_sec: Optional[float] = None


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
    if len(routes) != inst.K:
        violations.append(f"Số lộ trình ({len(routes)}) khác K ({inst.K})")
    covered = np.zeros((inst.m, inst.n), dtype=bool)
    per_uav = []
    total_visits = 0
    for k, r in enumerate(routes[:inst.K]):
        r_tuples = [tuple(c) for c in r]
        if len(set(r_tuples)) != len(r_tuples):
            violations.append(f"UAV {k}: lặp ô trong lộ trình (C3)")
        valid_cells = True
        for c in r_tuples:
            if not (0 <= c[0] < inst.m and 0 <= c[1] < inst.n):
                violations.append(f"UAV {k}: ô {c} ngoài lưới (C3)")
                valid_cells = False
        T, E = route_metrics(inst, k, r_tuples) if valid_cells else (0.0, 0.0)
        cap = inst.eta * inst.uavs[k].B
        if T > inst.D + TOL:
            violations.append(f"UAV {k}: T_L={T:.2f} > D={inst.D:.2f} (C1)")
        if E > cap + TOL:
            violations.append(f"UAV {k}: E={E:.2f} > eta*B={cap:.2f} (C2)")
        per_uav.append(dict(cells=len(r_tuples), time=T, energy=E, battery_use=E / cap if cap > 0 else 0.0))
        if valid_cells:
            for c in r_tuples:
                covered[c] = True
        total_visits += len(r_tuples)
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
# Trạng thái 1 UAV trong lúc lập kế hoạch cục bộ (greedy / ACO)
# --------------------------------------------------------------------------- #
class UAVState:
    """Theo dõi quãng đường đã bay `flight`, danh sách ô quan sát `route` và đường bay `walk`."""

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
        """Quan sát `cell` mà vẫn thoả (C1) và (C2) kể cả chặng về depot."""
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
# Đọc / Ghi file JSON theo đúng docs/instance_format.md
# --------------------------------------------------------------------------- #
def load_instance(path: Union[str, Path], check_reachability: bool = False) -> Instance:
    """Đọc file instance *.json theo định dạng docs/instance_format.md."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    grid = data["grid"]
    m, n, u = int(grid["m"]), int(grid["n"]), float(grid["u"])
    w = np.asarray(grid["w"], dtype=float)
    assert w.shape == (m, n) and int(data["K"]) == len(data["uavs"])
    uavs = [
        UAV((int(a["depot"][0]), int(a["depot"][1])), float(a["B"]), float(a["v_h"]),
            float(a["e_h"]), float(a["tau"]), float(a["eps"]))
        for a in data["uavs"]
    ]
    inst = Instance(w=w, u=u, D=float(data["D"]), eta=float(data["eta"]), uavs=uavs,
                    name=str(data.get("name", p.stem)), source=data.get("source"), rho=data.get("rho"))
    if check_reachability:
        for k in range(inst.K):
            assert any(
                UAVState(inst, k).can_observe((i, j)) for i in range(m) for j in range(n)
            ), f"UAV {k} không bay được ô nào"
    return inst


def save_instance(inst: Instance, path: Union[str, Path]) -> None:
    """Ghi đối tượng Instance ra file *.json theo định dạng docs/instance_format.md."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "name": inst.name,
        "source": inst.source or {"dataset": "synthetic"},
        "grid": {"m": inst.m, "n": inst.n, "u": inst.u, "w": inst.w.tolist()},
        "K": inst.K, "D": inst.D, "eta": inst.eta,
        "uavs": [{"depot": list(a.depot), "B": a.B, "v_h": a.v_h, "e_h": a.e_h, "tau": a.tau, "eps": a.eps}
                 for a in inst.uavs],
    }
    if inst.rho is not None:
        payload["rho"] = inst.rho
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_solution(sol: Solution, path: Union[str, Path], instance_name: str = "synthetic",
                  runtime_sec: Optional[float] = None) -> None:
    """Ghi nghiệm ra file *.solution.json theo định dạng docs/instance_format.md."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rt = runtime_sec if runtime_sec is not None else sol.runtime_sec
    payload: Dict[str, Any] = {"instance": instance_name, "algorithm": sol.name}
    if rt is not None:
        payload["runtime_sec"] = round(float(rt), 6)
    payload["routes"] = [[[int(c[0]), int(c[1])] for c in r] for r in sol.routes]
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_solution(path: Union[str, Path]) -> Solution:
    """Đọc file *.solution.json."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Solution(
        name=str(data.get("algorithm", "unknown")),
        routes=[[(int(c[0]), int(c[1])) for c in r] for r in data["routes"]],
        runtime_sec=data.get("runtime_sec"),
    )


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
    B = eta ** -1 * (e_h * flight_budget_m + eps * max_cells)
    D = flight_budget_m / v_h + tau * max_cells
    uavs = [UAV(depot=(int(rng.integers(m)), int(rng.integers(n))), B=B, v_h=v_h, e_h=e_h, tau=tau, eps=eps)
            for _ in range(K)]
    return Instance(w=w, u=u, D=D, eta=eta, uavs=uavs,
                    name=f"synthetic_{m}x{n}_K{K}_s{seed}", source={"dataset": "synthetic", "seed": seed})
