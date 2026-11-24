"""
ga.py - Thuật toán Di truyền (Genetic Algorithm - GA) điều chỉnh theo mô hình.

Tận dụng trực tiếp `UAVState`, `neighbors8`, `evaluate` và `run_shared_closed` từ common.py:
  * Cá thể (Chromosome): ma trận trọng số ưu tiên mở rộng `prio` (hoặc lộ trình K UAV được
    giải mã bằng `run_shared_closed` để luôn đảm bảo (C1)-(C3) và các lộ trình rời nhau).
  * Lai ghép (Uniform / Blend Crossover) và Đột biến (Gaussian perturbation) trên bản đồ ưu tiên,
    kết hợp toán tử sửa lộ trình cục bộ để tìm thứ tự ghé thăm tối ưu.
"""
from __future__ import annotations

from typing import List

import numpy as np

from common import Action, Cell, Instance, Solution, UAVState, evaluate, neighbors8, run_shared_closed
from na_greedy import na_greedy
from svrec import svrec


def _decode(inst: Instance, prio: np.ndarray, r_th: float) -> List[List[Cell]]:
    """Giải mã bản đồ gen ưu tiên `prio` thành lộ trình K UAV hợp lệ qua `run_shared_closed`."""
    def step(inst: Instance, s: UAVState, closed: np.ndarray) -> Action:
        cands = [c for c in neighbors8(s.cur, inst.m, inst.n) if not closed[c] and s.can_observe(c)]
        if not cands or float(np.mean([inst.w[c] for c in cands])) < r_th:
            idx = np.argwhere((~closed) & (inst.w > r_th))
            if len(idx) > 0:
                d = np.hypot(idx[:, 0] - s.cur[0], idx[:, 1] - s.cur[1])
                for o in np.argsort(d)[:6]:
                    c = (int(idx[o, 0]), int(idx[o, 1]))
                    if c not in cands and s.can_observe(c):
                        cands.append(c)
        if not cands:
            return None
        best = max(cands, key=lambda c: prio[c] / (inst.dist(s.cur, c) / inst.u))
        return ("observe", best)

    return run_shared_closed(inst, step, "GA").routes


def ga(
    inst: Instance,
    pop_size: int = 12,
    n_generations: int = 12,
    crossover_prob: float = 0.85,
    mutation_prob: float = 0.40,
    seed: int = 42,
) -> Solution:
    rng = np.random.default_rng(seed)
    r_th = float(inst.w.mean())

    # Khởi tạo quần thể gen ưu tiên xoay quanh mảng rủi ro gốc w
    pop = [inst.w.copy()]
    for _ in range(pop_size - 1):
        noise = rng.uniform(0.7, 1.3, size=inst.w.shape)
        pop.append(inst.w * noise)

    best_routes = max([svrec(inst).routes, na_greedy(inst).routes], key=lambda r: evaluate(inst, r)["f"])
    best_f = float(evaluate(inst, best_routes)["f"])

    # Bơm kinh nghiệm từ nghiệm khởi tạo vào cá thể đầu tiên
    for r in best_routes:
        for c in r:
            pop[0][c] *= 1.35

    fit = []
    for ind in pop:
        rds = _decode(inst, ind, r_th)
        f_val = float(evaluate(inst, rds)["f"])
        fit.append(f_val)
        if f_val > best_f:
            best_f, best_routes = f_val, rds

    for _ in range(n_generations):
        order = np.argsort(fit)[::-1]
        new_pop = [pop[int(order[0])].copy(), pop[int(order[1])].copy()]
        new_fit = [fit[int(order[0])], fit[int(order[1])]]

        while len(new_pop) < pop_size:
            i1, i2 = rng.choice(pop_size, size=2, replace=False)
            p1 = pop[int(i1 if fit[i1] >= fit[i2] else i2)]
            j1, j2 = rng.choice(pop_size, size=2, replace=False)
            p2 = pop[int(j1 if fit[j1] >= fit[j2] else j2)]

            if rng.random() < crossover_prob:
                mask = rng.random(inst.w.shape) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()

            if rng.random() < mutation_prob:
                mut_mask = rng.random(inst.w.shape) < 0.15
                child = np.where(mut_mask, child * rng.uniform(0.6, 1.5, size=inst.w.shape), child)

            rds = _decode(inst, child, r_th)
            f_c = float(evaluate(inst, rds)["f"])
            new_pop.append(child)
            new_fit.append(f_c)
            if f_c > best_f:
                best_f, best_routes = f_c, rds

        pop, fit = new_pop, new_fit

    return Solution("GA", best_routes, best_routes)


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    from common import make_random_instance

    inst = make_random_instance()
    sol = ga(inst)
    r = evaluate(inst, sol.routes)
    print(sol.name, "feasible =", r["feasible"], "| risk_ratio = %.3f" % r["risk_ratio"])
