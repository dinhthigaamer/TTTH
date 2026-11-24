"""
h_first.py - Greedy "đi H trước": chọn ô theo  w_c [c in H] / t_c ,  t_c = t_k + d/v_h + tau  (thời điểm
hoàn tất nếu quan sát c ngay), kèm bước tiến gần H:
  1. Có ô lân cận thuộc H khả thi -> chọn ô có w_c/t_c lớn nhất.
  2. Không có -> nhắm ô H chưa phủ gần nhất (khả thi từ vị trí hiện tại): đi sang ô lân cận làm giảm
     khoảng cách tới đích; nếu không ô nào làm giảm (bị chặn bởi CLOSE) thì bay thẳng tới đích.
  3. Hết ô H khả thi -> dùng ngân sách còn lại theo Na-Greedy (tăng fcov), kẹt thì nhảy tới ô w cao gần nhất.
Là baseline thiên về fben và là seed cho NSGA-II / archive MO-ACO.
"""
from __future__ import annotations

import numpy as np

from common import Action, Instance, Solution, UAVState, neighbors8, run_shared_closed
from common import clock, jump_target, nearest_uncovered_H


def make_h_first_step(H: np.ndarray):
    def step(inst: Instance, s: UAVState, closed: np.ndarray) -> Action:
        a, tk = s.uav, clock(s)
        nbrs = [c for c in neighbors8(
            s.cur, inst.m, inst.n) if not closed[c] and s.can_observe(c)]

        # (1) ô H kề bên
        hn = [c for c in nbrs if H[c]]
        if hn:
            best = min(hn, key=lambda c: (-inst.w[c] / (tk + inst.dist(s.cur, c) / a.v_h + a.tau),
                                          inst.dist(s.cur, c), c))
            return ("observe", best)

        # (2) tiến gần ô H chưa phủ gần nhất mà còn khả thi
        idx = np.argwhere(H & ~closed)
        if len(idx):
            d = np.hypot(idx[:, 0] - s.cur[0], idx[:, 1] - s.cur[1])
            tgt = None
            for o in np.lexsort((idx[:, 1], idx[:, 0], d)):
                c = (int(idx[o, 0]), int(idx[o, 1]))
                if s.can_observe(c):
                    tgt = c
                    break
            if tgt is not None:
                d0 = inst.dist(s.cur, tgt)
                closer = [c for c in nbrs if inst.dist(c, tgt) < d0 - 1e-12]
                if closer:
                    return ("observe", min(closer, key=lambda c: (inst.dist(c, tgt), -inst.w[c], c)))
                return ("observe", tgt)

        # (3) hết ô H khả thi: lấp thêm độ phủ
        if nbrs:
            return ("observe", min(nbrs, key=lambda c: (-inst.w[c], inst.dist(s.cur, c), c)))
        t = jump_target(inst, s, closed, inst.w)
        return None if t is None else ("observe", t)
    return step


def h_first(inst: Instance, H: np.ndarray) -> Solution:
    return run_shared_closed(inst, make_h_first_step(H), "H-First")
