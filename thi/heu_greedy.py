"""
heu_greedy.py - Heuristic Greedy (Algorithm 2 của bài báo) điều chỉnh theo mô hình.

Ý tưởng gốc: giống Na-Greedy nhưng ô lân cận đã thuộc CLOSE không bị loại, mà bị phạt r <- r - 10
để UAV vẫn có thể đi qua ô đã phủ khi bị kẹt, nhưng không cộng thêm trọng số.
Điều chỉnh:
  * Eff-weight r = w_c; mức phạt `penalty` mặc định = max(w) (tương ứng 10 = r lớn nhất của bài báo),
    nên ô đã phủ có r <= 0 và chỉ được chọn khi không còn ô mới tốt hơn.
  * Bước "bay qua ô đã phủ" (transit) tốn thời gian bay và năng lượng nhưng KHÔNG tốn tau, eps và KHÔNG
    nằm trong R_k (R_k chỉ gồm các ô quan sát, nên không lặp ô: (C3)). Kế hoạch dùng quãng bay thực
    nên bảo thủ; theo mô hình (bay thẳng giữa các ô quan sát) chi phí thực chỉ nhỏ hơn hoặc bằng.
  * Mọi bước đều phải còn đủ pin để về depot; quan sát phải thoả thêm deadline (C1).
  * Nhiều UAV: dùng chung CLOSE, thay phiên đi từng bước.
  * Hoà điểm: ưu tiên quan sát ô mới hơn bay qua, rồi ô gần hơn, rồi theo chỉ số.
Điểm yếu giữ nguyên như bài báo: Weight Trap động (dao động giữa các ô đã phủ) khi r giảm dần.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common import Action, Instance, Solution, UAVState, neighbors8, run_shared_closed


def make_heu_greedy_step(penalty: Optional[float] = None):
    def step(inst: Instance, s: UAVState, closed: np.ndarray) -> Action:
        pen = penalty if penalty is not None else (float(inst.w.max()) or 1.0)
        best, best_key = None, None
        for c in neighbors8(s.cur, inst.m, inst.n):
            if closed[c]:
                if not s.can_transit(c):
                    continue
                kind, r = "transit", inst.w[c] - pen
            else:
                if not s.can_observe(c):
                    continue
                kind, r = "observe", inst.w[c]
            key = (-r, kind != "observe", inst.dist(s.cur, c), c)
            if best_key is None or key < best_key:
                best, best_key = (kind, c), key
        return best
    return step


def heu_greedy(inst: Instance, penalty: Optional[float] = None) -> Solution:
    return run_shared_closed(inst, make_heu_greedy_step(penalty), "Heu-Greedy")


if __name__ == "__main__":
    from common import evaluate, make_random_instance
    inst = make_random_instance()
    sol = heu_greedy(inst)
    r = evaluate(inst, sol.routes)
    print(sol.name, "feasible =", r["feasible"], "| risk_ratio = %.3f" % r["risk_ratio"])
