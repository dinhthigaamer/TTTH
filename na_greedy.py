"""
na_greedy.py - Naive Greedy (Algorithm 1 của bài báo) điều chỉnh theo mô hình.

Ý tưởng gốc: từ ô hiện tại, xét 8 ô lân cận, loại ô đã thuộc CLOSE, chọn ô có trọng số r lớn nhất.
Điều chỉnh:
  * Eff-weight r = w_c (mức rủi ro của ô), không dùng ánh xạ r = 10h - 2 của bài báo.
  * Điều kiện dừng "d_c <= d_m" được thay bằng kiểm tra đồng thời deadline (C1) và pin (C2) có
    chặng về depot cho từng ứng viên; ứng viên không thoả bị loại.
  * Nhiều UAV: dùng chung CLOSE, thay phiên đi từng bước, nên các lộ trình rời nhau.
  * Hoà điểm: chọn ô gần hơn (đi ngang/dọc thay vì chéo), rồi theo chỉ số (i, j).
UAV dừng khi mọi ô lân cận đã phủ hoặc không còn khả thi (Weight Trap / Weight Interference).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common import Action, Instance, Solution, UAVState, neighbors8, run_shared_closed


def na_greedy_step(inst: Instance, s: UAVState, closed: np.ndarray) -> Action:
    best, best_key = None, None
    for c in neighbors8(s.cur, inst.m, inst.n):
        if closed[c] or not s.can_observe(c):
            continue
        key = (-inst.w[c], inst.dist(s.cur, c), c)      # nhỏ nhất = tốt nhất
        if best_key is None or key < best_key:
            best, best_key = c, key
    return None if best is None else ("observe", best)


def na_greedy(inst: Instance) -> Solution:
    return run_shared_closed(inst, na_greedy_step, "Na-Greedy")


if __name__ == "__main__":
    from common import evaluate, make_random_instance
    inst = make_random_instance()
    sol = na_greedy(inst)
    r = evaluate(inst, sol.routes)
    print(sol.name, "feasible =", r["feasible"], "| risk_ratio = %.3f" % r["risk_ratio"])
