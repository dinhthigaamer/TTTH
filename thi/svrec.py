"""
svrec.py - SVReC = SVeC (Skip Vertices Coverage, Algorithm 3) + VwRR (Vertex weight Resolution
Reinforcement, Algorithm 4) trên nền Na-Greedy. Điều chỉnh theo mô hình:

  * Eff-weight r = w_c. Ngưỡng r_th mặc định = trung bình w toàn lưới (tham số `r_th`).
  * Bước cục bộ: xét các ô lân cận CHƯA phủ và còn khả thi (C1, C2 có chặng về depot).
  * SVeC (giải Weight Trap / Interference / Sparse): nếu không còn ô lân cận khả thi, hoặc trung bình
    w của chúng < r_th, UAV "nhảy" tới ô chưa phủ gần nhất có w > r_th mà còn khả thi. Bước nhảy là
    chuyến bay thẳng (khoảng cách Euclid) giữa hai ô quan sát, đúng như mô hình. Nếu không có đích
    khả thi thì quay lại bước cục bộ; nếu cũng không có thì UAV dừng.
  * VwRR (giải Weight Redundancy): trong các ô lân cận có w lớn nhất bằng nhau, chọn ô có trung bình
    w của các ô chưa phủ xung quanh nó cao nhất; hoà tiếp thì chọn ô gần hơn, rồi theo chỉ số.
  * Nhiều UAV: dùng chung CLOSE, thay phiên đi từng bước. Không có bước "bay qua", mọi chặng đều
    là chặng giữa hai ô quan sát nên kế hoạch khớp hoàn toàn với mô hình.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common import Action, Cell, Instance, Solution, UAVState, neighbors8, run_shared_closed


def _avg_neighbor_r(inst: Instance, c: Cell, closed: np.ndarray) -> float:
    vals = [inst.w[x] for x in neighbors8(c, inst.m, inst.n) if not closed[x]]
    return float(np.mean(vals)) if vals else 0.0


def _svec_target(inst: Instance, s: UAVState, closed: np.ndarray, r_th: float) -> Optional[Cell]:
    idx = np.argwhere((~closed) & (inst.w > r_th))
    if len(idx) == 0:
        return None
    d = np.hypot(idx[:, 0] - s.cur[0], idx[:, 1] - s.cur[1])
    for o in np.lexsort((idx[:, 1], idx[:, 0], d)):     # gần nhất, hoà theo (i, j)
        c = (int(idx[o, 0]), int(idx[o, 1]))
        if s.can_observe(c):
            return c
    return None


def _vwrr(inst: Instance, s: UAVState, closed: np.ndarray, nbrs: list) -> Cell:
    rmax = max(inst.w[c] for c in nbrs)
    tied = [c for c in nbrs if abs(inst.w[c] - rmax) <= 1e-12]
    return min(tied, key=lambda c: (-_avg_neighbor_r(inst, c, closed), inst.dist(s.cur, c), c))


def make_svrec_step(r_th: Optional[float] = None):
    def step(inst: Instance, s: UAVState, closed: np.ndarray) -> Action:
        th = r_th if r_th is not None else float(inst.w.mean())
        nbrs = [c for c in neighbors8(s.cur, inst.m, inst.n) if not closed[c] and s.can_observe(c)]
        if not nbrs or float(np.mean([inst.w[c] for c in nbrs])) < th:
            t = _svec_target(inst, s, closed, th)
            if t is not None:
                return ("observe", t)
        if not nbrs:
            return None
        return ("observe", _vwrr(inst, s, closed, nbrs))
    return step


def svrec(inst: Instance, r_th: Optional[float] = None) -> Solution:
    return run_shared_closed(inst, make_svrec_step(r_th), "SVReC")


if __name__ == "__main__":
    from common import evaluate, make_random_instance
    inst = make_random_instance()
    sol = svrec(inst)
    r = evaluate(inst, sol.routes)
    print(sol.name, "feasible =", r["feasible"], "| risk_ratio = %.3f" % r["risk_ratio"])
