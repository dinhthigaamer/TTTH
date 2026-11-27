"""
ga.py - Thuật toán Di truyền (Genetic Algorithm - GA) điều chỉnh theo mô hình.

Phần mở rộng đa mục tiêu (theo "Đề xuất mở rộng đa mục tiêu"): thêm gen gamma in [0, 2] nhân vào prio tại
các ô H, `ga_scalar` (GA đơn mục tiêu theo tổng có trọng số của fcov, fben chuẩn hoá) và `nsga2`
(rank không bị chi phối + crowding, giải đấu (rank, crowding), sống sót từ gộp cha+con). `_decode` giữ
nguyên (H=None, gamma=1 cho kết quả y như cũ) nên (C1)-(C3) vẫn được đảm bảo; hàm `ga` gốc không đổi.

Tận dụng trực tiếp `UAVState`, `neighbors8`, `evaluate` và `run_shared_closed` từ common.py:
  * Cá thể (Chromosome): ma trận trọng số ưu tiên mở rộng `prio` (hoặc lộ trình K UAV được
    giải mã bằng `run_shared_closed` để luôn đảm bảo (C1)-(C3) và các lộ trình rời nhau).
  * Lai ghép (Uniform / Blend Crossover) và Đột biến (Gaussian perturbation) trên bản đồ ưu tiên,
    kết hợp toán tử sửa lộ trình cục bộ để tìm thứ tự ghé thăm tối ưu.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from common import (Action, Cell, Instance, ParetoArchive, Solution, UAVState, crowding, evaluate,
                    fast_nds, hypervolume_2d, neighbors8, run_shared_closed)
from h_first import h_first
from heu_greedy import heu_greedy
from na_greedy import na_greedy
from svrec import svrec


def _decode(inst: Instance, prio: np.ndarray, r_th: float, H: Optional[np.ndarray] = None,
            gamma: float = 1.0) -> List[List[Cell]]:
    """Giải mã bản đồ gen ưu tiên `prio` thành lộ trình K UAV hợp lệ qua `run_shared_closed`.
    Nếu có H: prio' = prio * gamma tại các ô H (gen gamma của bài toán đa mục tiêu)."""
    P = prio if H is None else prio * np.where(H, gamma, 1.0)

    def step(inst: Instance, s: UAVState, closed: np.ndarray) -> Action:
        cands = [c for c in neighbors8(
            s.cur, inst.m, inst.n) if not closed[c] and s.can_observe(c)]
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
        best = max(cands, key=lambda c: P[c] / (inst.dist(s.cur, c) / inst.u))
        return ("observe", best)

    return run_shared_closed(inst, step, "GA").routes


def _over_budget(
    t0: float,
    evals: int,
    time_limit: Optional[float],
    max_evals: Optional[int],
) -> bool:
    return evals >= 1 and (
        (time_limit is not None and time.perf_counter() - t0 >= time_limit)
        or (max_evals is not None and evals >= max_evals)
    )


def ga(
    inst: Instance,
    pop_size: int = 12,
    n_generations: int = 12,
    crossover_prob: float = 0.85,
    mutation_prob: float = 0.40,
    seed: int = 42,
    time_limit: Optional[float] = None,
    max_evals: Optional[int] = None,
    trace: Optional[list] = None,
) -> Solution:
    t0 = time.perf_counter()
    evals = 0
    rng = np.random.default_rng(seed)
    r_th = float(inst.w.mean())

    # Khởi tạo quần thể gen ưu tiên xoay quanh mảng rủi ro gốc w
    pop = [inst.w.copy()]
    for _ in range(pop_size - 1):
        noise = rng.uniform(0.7, 1.3, size=inst.w.shape)
        pop.append(inst.w * noise)

    best_routes = max([svrec(inst).routes, na_greedy(
        inst).routes], key=lambda r: evaluate(inst, r)["f"])
    best_f = float(evaluate(inst, best_routes)["f"])
    if trace is not None:
        trace.append((time.perf_counter() - t0, [list(r) for r in best_routes]))

    # Bơm kinh nghiệm từ nghiệm khởi tạo vào cá thể đầu tiên
    for r in best_routes:
        for c in r:
            pop[0][c] *= 1.35

    fit = []
    stop = False
    for ind in pop:
        if _over_budget(t0, evals, time_limit, max_evals):
            stop = True
            break
        rds = _decode(inst, ind, r_th)
        evals += 1
        f_val = float(evaluate(inst, rds)["f"])
        fit.append(f_val)
        if f_val > best_f:
            best_f, best_routes = f_val, rds
            if trace is not None:
                trace.append((time.perf_counter() - t0, [list(r) for r in best_routes]))

    if not stop and len(fit) >= 2:
        for _ in range(n_generations):
            order = np.argsort(fit)[::-1]
            new_pop = [pop[int(order[0])].copy(), pop[int(order[1])].copy()]
            new_fit = [fit[int(order[0])], fit[int(order[1])]]

            while len(new_pop) < pop_size:
                if _over_budget(t0, evals, time_limit, max_evals):
                    stop = True
                    break
                i1, i2 = rng.choice(len(fit), size=2, replace=False)
                p1 = pop[int(i1 if fit[i1] >= fit[i2] else i2)]
                j1, j2 = rng.choice(len(fit), size=2, replace=False)
                p2 = pop[int(j1 if fit[j1] >= fit[j2] else j2)]

                if rng.random() < crossover_prob:
                    mask = rng.random(inst.w.shape) < 0.5
                    child = np.where(mask, p1, p2)
                else:
                    child = p1.copy()

                if rng.random() < mutation_prob:
                    mut_mask = rng.random(inst.w.shape) < 0.15
                    child = np.where(
                        mut_mask, child * rng.uniform(0.6, 1.5, size=inst.w.shape), child)

                rds = _decode(inst, child, r_th)
                evals += 1
                f_c = float(evaluate(inst, rds)["f"])
                new_pop.append(child)
                new_fit.append(f_c)
                if f_c > best_f:
                    best_f, best_routes = f_c, rds
                    if trace is not None:
                        trace.append((time.perf_counter() - t0, [list(r) for r in best_routes]))

            if stop:
                break
            pop, fit = new_pop, new_fit

    return Solution("GA", best_routes, best_routes)


# =========================================================================== #
# Mở rộng đa mục tiêu: gen gamma, GA tổng có trọng số, NSGA-II
# =========================================================================== #
class _Ind:
    __slots__ = ("prio", "gamma", "routes", "objs", "rank", "cd")

    def __init__(self, prio: np.ndarray, gamma: float = 1.0):
        self.prio, self.gamma = prio, float(gamma)
        self.routes, self.objs, self.rank, self.cd = None, None, 0, 0.0


def _eval_ind(inst: Instance, H: np.ndarray, r_th: float, ind: _Ind) -> _Ind:
    """Một lần _decode + evaluate cho cả hai mục tiêu: objs = (fcov chuẩn hoá, fben chuẩn hoá)."""
    ind.routes = _decode(inst, ind.prio, r_th, H, ind.gamma)
    ind.objs = evaluate(inst, ind.routes, H)["objs"]
    return ind


def _default_seeds(inst: Instance, H: np.ndarray) -> List[List[List[Cell]]]:
    return [svrec(inst).routes, na_greedy(inst).routes, heu_greedy(inst).routes, h_first(inst, H).routes]


def _init_pop(
    inst,
    H,
    r_th,
    rng,
    pop_size,
    seeds,
    t0: Optional[float] = None,
    time_limit: Optional[float] = None,
    max_evals: Optional[int] = None,
) -> Tuple[List[_Ind], int, bool]:
    """Quần thể xoay quanh w (như ga gốc); mỗi nghiệm seed bơm kinh nghiệm x1.35 vào một cá thể."""
    pop = [_Ind(inst.w.copy(), 1.0)]
    for r in seeds:
        p = inst.w.copy()
        for route in r:
            for c in route:
                p[c] *= 1.35
        pop.append(_Ind(p, 1.0))
    while len(pop) < pop_size:
        pop.append(_Ind(inst.w * rng.uniform(0.7, 1.3,
                   size=inst.w.shape), float(rng.uniform(0.5, 1.5))))
    out: List[_Ind] = []
    evals = 0
    stop = False
    t_start = time.perf_counter() if t0 is None else t0
    for i in pop[:pop_size]:
        if _over_budget(t_start, evals, time_limit, max_evals):
            stop = True
            break
        out.append(_eval_ind(inst, H, r_th, i))
        evals += 1
    return out, evals, stop


def _vary(inst, H, rng, p1: _Ind, p2: _Ind, crossover_prob, mutation_prob,
          block_xover: bool, corridor_mut: bool) -> _Ind:
    """Uniform crossover (+ crossover theo khối), đột biến nhân (+ đột biến theo hành lang tới H)."""
    m, n = inst.w.shape
    prio, gamma = p1.prio.copy(), p1.gamma
    if rng.random() < crossover_prob:
        if block_xover and rng.random() < 0.5:
            i0, i1 = sorted(int(x) for x in rng.integers(0, m + 1, 2))
            j0, j1 = sorted(int(x) for x in rng.integers(0, n + 1, 2))
            prio[i0:i1, j0:j1] = p2.prio[i0:i1, j0:j1]
        else:
            prio = np.where(rng.random((m, n)) < 0.5, p1.prio, p2.prio)
        gamma = p1.gamma if rng.random() < 0.5 else p2.gamma
    if rng.random() < mutation_prob:
        mut_mask = rng.random((m, n)) < 0.15
        prio = np.where(mut_mask, prio *
                        rng.uniform(0.6, 1.5, size=(m, n)), prio)
        gamma = float(np.clip(gamma + rng.normal(0, 0.2), 0.0, 2.0))
    if corridor_mut and rng.random() < 0.3:        # nâng prio dọc đường thẳng từ depot/ô bất kỳ tới một ô H
        hc = np.argwhere(H)
        if len(hc):
            h = hc[int(rng.integers(len(hc)))]
            start = np.array(inst.uavs[int(rng.integers(inst.K))].depot if rng.random() < 0.5
                             else (rng.integers(m), rng.integers(n)))
            q = float(np.quantile(prio, 0.9))
            for t in np.linspace(0, 1, int(np.abs(h - start).max()) + 1):
                c = np.rint(start + t * (h - start)).astype(int)
                prio[c[0], c[1]] = max(prio[c[0], c[1]], q)
    return _Ind(prio, gamma)


def ga_scalar(inst: Instance, H: np.ndarray, w_cov: float = 1.0, w_ben: float = 0.0, pop_size: int = 12,
              n_generations: int = 12, crossover_prob: float = 0.85, mutation_prob: float = 0.40,
              seed: int = 42, seeds: Optional[Sequence[List[List[Cell]]]] = None,
              block_xover: bool = True, corridor_mut: bool = True,
              time_limit: Optional[float] = None, max_evals: Optional[int] = None,
              trace: Optional[list] = None) -> _Ind:
    """GA đơn mục tiêu cực đại hoá w_cov*fcov_norm + w_ben*fben_norm (giữ khung ga gốc: elitism 2,
    giải đấu 2). (1,0)/(0,1): tối ưu riêng từng mục tiêu cho kiểm tra xung đột; (a,b): GA có trọng số.
    Trả về cá thể tốt nhất (có .routes, .objs)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    r_th = float(inst.w.mean())
    seeds = _default_seeds(inst, H) if seeds is None else seeds
    def fitf(i): return w_cov * i.objs[0] + w_ben * i.objs[1]
    pop, evals, stop = _init_pop(inst, H, r_th, rng, pop_size, seeds, t0, time_limit, max_evals)
    best_ind = max(pop, key=fitf)
    if trace is not None:
        trace.append((time.perf_counter() - t0, best_ind.objs))
    if not stop and len(pop) >= 2:
        for _ in range(n_generations):
            pop.sort(key=fitf, reverse=True)
            new_pop = pop[:2]
            while len(new_pop) < pop_size:
                if _over_budget(t0, evals, time_limit, max_evals):
                    stop = True
                    break
                ps = []
                for _ in range(2):
                    a, b = rng.choice(len(pop), size=2, replace=False)
                    ps.append(pop[int(a)] if fitf(pop[int(a)]) >=
                              fitf(pop[int(b)]) else pop[int(b)])
                child = _vary(
                    inst, H, rng, ps[0], ps[1], crossover_prob, mutation_prob, block_xover, corridor_mut)
                c_ind = _eval_ind(inst, H, r_th, child)
                new_pop.append(c_ind)
                evals += 1
                if fitf(c_ind) > fitf(best_ind):
                    best_ind = c_ind
                    if trace is not None:
                        trace.append((time.perf_counter() - t0, best_ind.objs))
            pop = new_pop
            if stop:
                break
    return best_ind


def _rank_crowd(pop: List[_Ind]) -> List[List[int]]:
    objs = np.array([i.objs for i in pop])
    fronts = fast_nds(objs)
    for r, f in enumerate(fronts):
        cd = crowding(objs[f])
        for idx, c in zip(f, cd):
            pop[idx].rank, pop[idx].cd = r, c
    return fronts


def nsga2(inst: Instance, H: np.ndarray, pop_size: int = 30, n_generations: int = 30,
          crossover_prob: float = 0.85, mutation_prob: float = 0.40, seed: int = 42,
          seeds: Optional[Sequence[List[List[Cell]]]] = None, block_xover: bool = True,
          corridor_mut: bool = True, archive_cap: int = 40,
          time_limit: Optional[float] = None, max_evals: Optional[int] = None,
          trace: Optional[list] = None) -> Tuple[ParetoArchive, List[float]]:
    """NSGA-II cho (fcov, fben). Seed quần thể (và archive) bằng svrec, na_greedy, heu_greedy, h_first.
    Trả về (archive Pareto của mọi nghiệm đã gặp, lịch sử Hypervolume theo thế hệ). Gợi ý: P=30-40, G=30-50."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    r_th = float(inst.w.mean())
    seeds = _default_seeds(inst, H) if seeds is None else seeds
    arch, hist = ParetoArchive(archive_cap), []
    for r in seeds:
        arch.add(evaluate(inst, r, H)["objs"], r)
    if trace is not None:
        trace.append((0.0, arch.points().copy()))
    pop, evals, stop = _init_pop(inst, H, r_th, rng, pop_size, seeds, t0, time_limit, max_evals)
    for i in pop:
        if arch.add(i.objs, i.routes) and trace is not None:
            trace.append((time.perf_counter() - t0, arch.points().copy()))
    if stop or len(pop) < 2:
        hist.append(hypervolume_2d(arch.points()))
        return arch, hist
    _rank_crowd(pop)

    def tour() -> _Ind:
        a, b = (pop[int(j)]
                for j in rng.choice(len(pop), size=2, replace=False))
        return a if (a.rank, -a.cd) < (b.rank, -b.cd) else b

    for _ in range(n_generations):
        kids = []
        for _ in range(pop_size):
            if _over_budget(t0, evals, time_limit, max_evals):
                stop = True
                break
            k = _eval_ind(inst, H, r_th, _vary(inst, H, rng, tour(), tour(), crossover_prob,
                                               mutation_prob, block_xover, corridor_mut))
            evals += 1
            if arch.add(k.objs, k.routes) and trace is not None:
                trace.append((time.perf_counter() - t0, arch.points().copy()))
            kids.append(k)
        if stop:
            hist.append(hypervolume_2d(arch.points()))
            break
        merged = pop + kids
        fronts = _rank_crowd(merged)
        nxt: List[_Ind] = []
        for f in fronts:
            if len(nxt) + len(f) <= pop_size:
                nxt += [merged[i] for i in f]
            else:
                f = sorted(f, key=lambda i: -merged[i].cd)
                nxt += [merged[i] for i in f[:pop_size - len(nxt)]]
                break
        pop = nxt
        _rank_crowd(pop)
        hist.append(hypervolume_2d(arch.points()))
    return arch, hist


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    from common import make_random_instance

    inst = make_random_instance()
    sol = ga(inst)
    r = evaluate(inst, sol.routes)
    print(sol.name, "feasible =", r["feasible"],
          "| risk_ratio = %.3f" % r["risk_ratio"])
