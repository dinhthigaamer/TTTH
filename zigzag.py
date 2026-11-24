"""
zigzag.py - Baseline Zig-Zag (quét liên tục kiểu boustrophedon), bỏ qua giá trị rủi ro.

Điều chỉnh so với bài báo để khớp mô hình:
  * Chia lưới thành K dải hàng liền kề (mỗi UAV một dải) để các UAV không chọn trùng ô.
    UAV được gán dải theo thứ tự hàng của depot (trên xuống dưới).
  * Mỗi UAV chọn 1 trong 4 kiểu quét (hướng hàng x hướng cột đầu tiên) sao cho ô đầu
    gần depot nhất, rồi đi theo thứ tự quét.
  * Dừng ở ô đầu tiên không còn thoả đồng thời deadline (C1) và pin có chặng về depot (C2).
Nếu K > m thì chỉ m UAV đầu có dải, các UAV còn lại giữ lộ trình rỗng.
"""
from __future__ import annotations

from typing import List

import numpy as np

from common import Cell, Instance, Solution, UAVState


def _sweep(rows: List[int], n: int, rows_desc: bool, first_ltr: bool) -> List[Cell]:
    seq: List[Cell] = []
    ltr = first_ltr
    for r in (reversed(rows) if rows_desc else rows):
        cols = range(n) if ltr else range(n - 1, -1, -1)
        seq.extend((r, c) for c in cols)
        ltr = not ltr
    return seq


def zigzag(inst: Instance) -> Solution:
    nb = min(inst.K, inst.m)
    bands = [list(map(int, b)) for b in np.array_split(np.arange(inst.m), nb)]
    order = sorted(range(inst.K), key=lambda k: (inst.uavs[k].depot[0], k))   # UAV theo hàng depot
    states = [UAVState(inst, k) for k in range(inst.K)]

    for band, k in zip(bands, order[:nb]):
        s = states[k]
        cands = [_sweep(band, inst.n, d, l) for d in (False, True) for l in (True, False)]
        seq = min(cands, key=lambda q: inst.dist(s.uav.depot, q[0]))
        for c in seq:
            if not s.can_observe(c):
                break
            s.observe(c)
    return Solution("Zig-Zag", [s.route for s in states], [s.walk for s in states])


if __name__ == "__main__":
    from common import evaluate, make_random_instance
    inst = make_random_instance()
    sol = zigzag(inst)
    r = evaluate(inst, sol.routes)
    print(sol.name, "feasible =", r["feasible"], "| risk_ratio = %.3f" % r["risk_ratio"])
