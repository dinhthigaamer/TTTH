"""
aco.py - Ant Colony Optimization (Algorithm 3-2, Eq. 18-23 của Cho et al., 2026)
điều chỉnh cho bài toán Deadline-constrained Multi-UAV Risk Coverage.

Ánh xạ trực tiếp từ bài báo sang mô hình (common.py):
  * Tập ứng viên Π_a: các ô 8-lân cận chưa thăm thoả (C1)-(C2); khi kẹt (Adj = ∅,
    Assumption 3 / Alg. 3-2 dòng 14), cho phép nhảy tới các ô ưu tiên cao gần nhất.
  * Eq. (20) Khởi tạo Pheromone τ_0 và Heuristic η(i, j) = w_j / T_{i,j} (ưu tiên rủi ro
    cao và thời gian bay T_{i,j} ngắn).
  * Eq. (18)-(19) Quy tắc chọn ô tiếp theo với ngưỡng q_0 (khai thác argmax hoặc
    Roulette Wheel Selection).
  * Eq. (21) Cập nhật Pheromone cục bộ sau mỗi bước đi: τ <- (1 - ρ)*τ + ρ*τ_0.
  * Eq. (22)-(23) Cập nhật Pheromone toàn cục sau mỗi thế hệ theo nghiệm tốt nhất P~:
    τ <- (1 - ε)*τ + ε*Δτ.
  * Bộ tham số chuẩn theo Mục 5.1 của bài báo: G=10, A=20, q0=0.7, alpha=1, beta=2, rho=0.1, eps=0.5.

Mở rộng MO-ACO (hàm `mo_aco`, `make_moaco_step`; `aco` gốc không đổi), theo "Đề xuất mở rộng đa mục tiêu":
  * Giữ nguyên run_shared_closed, tập ứng viên Π_a, xử lý kẹt, quy tắc q0 và cập nhật pheromone cục bộ.
    Đồng hồ t_k của UAV lấy từ common.clock.
  * Kiến thứ a mang lam_a = a/(A-1):  η = η_cov^lam · η_ben^(1-lam),
      η_cov = (w_c + 1e-3)/T_{i,c}                   (như aco gốc, T_{i,c} = d/v_h + tau)
      η_ben = w_c [c in H] / (t_k + d/v_h + tau).
    Ô c không thuộc H: dùng mức tiến gần ô H chưa phủ gần nhất h*:
      η_ben = w_h* / (t_k + (d(cur,c) + d(c,h*))/v_h + 2 tau)  (thời điểm ước tính tới h* khi đi qua c).
  * Pheromone: tau_cov[c] cùng tau_ben[b, c] theo khung thời gian b của t_k; trộn bằng cùng lam.
  * Archive Pareto (cắt bằng crowding) thay best_routes; cập nhật toàn cục bồi đắp cả hai ma trận từ archive.
"""
from __future__ import annotations

import numpy as np

from typing import List, Optional, Sequence, Tuple

from common import (Action, Cell, Instance, ParetoArchive, Solution, UAVState, cell_times, clock, evaluate,
                    hypervolume_2d, nearest_uncovered_H, neighbors8, run_shared_closed)


def make_aco_step(
    tau: np.ndarray,
    tau0: float,
    alpha: float,
    beta: float,
    q0: float,
    rho: float,
    rng: np.random.Generator,
):
    def step(inst: Instance, s: UAVState, closed: np.ndarray) -> Action:
        # Tập ứng viên lân cận Π_a (Alg. 3-2 dòng 13-14)
        cands = [c for c in neighbors8(
            s.cur, inst.m, inst.n) if not closed[c] and s.can_observe(c)]
        if not cands:
            # Xử lý Deadlock (Assumption 3): nhảy sang ô có rủi ro cao gần nhất còn khả thi
            idx = np.argwhere((~closed) & (inst.w > float(inst.w.mean())))
            if len(idx) == 0:
                return None
            d = np.hypot(idx[:, 0] - s.cur[0], idx[:, 1] - s.cur[1])
            for o in np.argsort(d):
                c = (int(idx[o, 0]), int(idx[o, 1]))
                if s.can_observe(c):
                    cands.append(c)
                    if len(cands) >= 8:
                        break
        if not cands:
            return None

        # Eq. (18)-(20): τ(i,j)^α * η(i,j)^β với η(i,j) = w_j / T_{i,j}
        uav = s.uav
        desirability = np.array([
            (tau[c] ** alpha) * (((inst.w[c] + 1e-3) /
                                  (inst.dist(s.cur, c) / uav.v_h + uav.tau)) ** beta)
            for c in cands
        ])
        if rng.random() <= q0 or len(cands) == 1:
            chosen = cands[int(np.argmax(desirability))]
        else:
            probs = desirability / desirability.sum()
            chosen = cands[int(rng.choice(len(cands), p=probs))]

        # Eq. (21): Cập nhật Pheromone cục bộ τ <- (1 - ρ)τ + ρ*τ_0
        tau[chosen] = (1.0 - rho) * tau[chosen] + rho * tau0
        return ("observe", chosen)

    return step


def aco(
    inst: Instance,
    G: int = 10,
    A: int = 20,
    q0: float = 0.7,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.1,
    eps_decay: float = 0.5,
    seed: int = 42,
) -> Solution:
    rng = np.random.default_rng(seed)
    tau0 = 1.0
    tau = np.full((inst.m, inst.n), tau0, dtype=float)
    best_routes = [[] for _ in range(inst.K)]
    best_f = -1.0

    for _ in range(G):
        for _ in range(A):
            sol = run_shared_closed(inst, make_aco_step(
                tau, tau0, alpha, beta, q0, rho, rng), "ACO")
            f_val = float(evaluate(inst, sol.routes)["f"])
            if f_val > best_f:
                best_f, best_routes = f_val, sol.routes

        # Eq. (22)-(23): Cập nhật Pheromone toàn cục theo nghiệm tốt nhất P~
        tau *= (1.0 - eps_decay)
        delta_tau = tau0 * (1.0 + best_f / max(float(inst.w.sum()), 1e-6))
        for r in best_routes:
            for c in r:
                tau[c] += eps_decay * delta_tau

    return Solution("ACO", best_routes, best_routes)


# =========================================================================== #
# Mở rộng đa mục tiêu: MO-ACO
# =========================================================================== #
def make_moaco_step(H: np.ndarray, tau_cov: np.ndarray, tau_ben: np.ndarray, tau0: float, alpha: float,
                    beta: float, q0: float, rho: float, lam: float, rng: np.random.Generator, nb: int):
    def step(inst: Instance, s: UAVState, closed: np.ndarray) -> Action:
        uav, tk = s.uav, clock(s)
        cands = [c for c in neighbors8(
            s.cur, inst.m, inst.n) if not closed[c] and s.can_observe(c)]
        if not cands:                                      # xử lý kẹt: giữ nguyên như aco gốc
            idx = np.argwhere((~closed) & (inst.w > float(inst.w.mean())))
            if len(idx) == 0:
                return None
            d = np.hypot(idx[:, 0] - s.cur[0], idx[:, 1] - s.cur[1])
            for o in np.argsort(d):
                c = (int(idx[o, 0]), int(idx[o, 1]))
                if s.can_observe(c):
                    cands.append(c)
                    if len(cands) >= 8:
                        break
        if not cands:
            return None

        b = min(nb - 1, int(tk / inst.D * nb))
        tgt = nearest_uncovered_H(s.cur, closed, H)
        des = []
        for c in cands:
            d = inst.dist(s.cur, c)
            e_cov = (inst.w[c] + 1e-3) / (d / uav.v_h + uav.tau)
            if H[c]:
                e_ben = inst.w[c] / (tk + d / uav.v_h + uav.tau)
            elif tgt is not None:
                e_ben = inst.w[tgt] / \
                    (tk + (d + inst.dist(c, tgt)) / uav.v_h + 2 * uav.tau)
            else:
                e_ben = 1e-6
            eta = (e_cov ** lam) * (max(e_ben, 1e-9) ** (1.0 - lam))
            ph = (tau_cov[c] ** lam) * (tau_ben[b][c] ** (1.0 - lam))
            des.append((ph ** alpha) * (eta ** beta))
        des = np.array(des)
        if rng.random() <= q0 or len(cands) == 1:
            chosen = cands[int(np.argmax(des))]
        else:
            chosen = cands[int(rng.choice(len(cands), p=des / des.sum()))]

        # cục bộ (Eq. 21) cho cả hai ma trận
        tau_cov[chosen] = (1.0 - rho) * tau_cov[chosen] + rho * tau0
        tau_ben[b][chosen] = (1.0 - rho) * tau_ben[b][chosen] + rho * tau0
        return ("observe", chosen)

    return step


def mo_aco(inst: Instance, H: np.ndarray, G: int = 10, A: int = 20, q0: float = 0.7, alpha: float = 1.0,
           beta: float = 2.0, rho: float = 0.1, eps_decay: float = 0.5, nb: int = 10, seed: int = 42,
           seeds: Optional[Sequence[List[List[Cell]]]] = None, archive_cap: int = 40
           ) -> Tuple[ParetoArchive, List[float]]:
    """Trả về (archive Pareto, lịch sử Hypervolume theo thế hệ). `seeds`: lộ trình greedy nạp sẵn vào archive."""
    rng = np.random.default_rng(seed)
    tau0 = 1.0
    tau_cov = np.full((inst.m, inst.n), tau0, dtype=float)
    tau_ben = np.full((nb, inst.m, inst.n), tau0, dtype=float)
    arch, hist = ParetoArchive(archive_cap), []
    for r in (seeds or []):
        arch.add(evaluate(inst, r, H)["objs"], r)

    for _ in range(G):
        for ai in range(A):
            lam = ai / (A - 1) if A > 1 else 0.5
            step = make_moaco_step(H, tau_cov, tau_ben,
                                   tau0, alpha, beta, q0, rho, lam, rng, nb)
            sol = run_shared_closed(inst, step, "MO-ACO")
            arch.add(evaluate(inst, sol.routes, H)["objs"], sol.routes)

        # cập nhật toàn cục: bay hơi rồi bồi đắp cả hai ma trận từ archive (chia đều theo số nghiệm)
        tau_cov *= (1.0 - eps_decay)
        tau_ben *= (1.0 - eps_decay)
        for (f_cov, f_ben), routes in arch.items:
            for k, r in enumerate(routes):
                for c, t in zip(r, cell_times(inst, k, r)):
                    b = min(nb - 1, int(t / inst.D * nb))
                    tau_cov[c] += eps_decay * tau0 * \
                        (1.0 + f_cov) / len(arch.items)
                    tau_ben[b][c] += eps_decay * tau0 * \
                        (1.0 + f_ben) / len(arch.items)
        hist.append(hypervolume_2d(arch.points()))
    return arch, hist


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    from common import make_random_instance

    inst = make_random_instance()
    sol = aco(inst)
    r = evaluate(inst, sol.routes)
    print(sol.name, "feasible =", r["feasible"],
          "| risk_ratio = %.3f" % r["risk_ratio"])
