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
"""
from __future__ import annotations

import numpy as np

from common import Action, Instance, Solution, UAVState, evaluate, neighbors8, run_shared_closed


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
        cands = [c for c in neighbors8(s.cur, inst.m, inst.n) if not closed[c] and s.can_observe(c)]
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
            (tau[c] ** alpha) * (((inst.w[c] + 1e-3) / (inst.dist(s.cur, c) / uav.v_h + uav.tau)) ** beta)
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
            sol = run_shared_closed(inst, make_aco_step(tau, tau0, alpha, beta, q0, rho, rng), "ACO")
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


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    from common import make_random_instance

    inst = make_random_instance()
    sol = aco(inst)
    r = evaluate(inst, sol.routes)
    print(sol.name, "feasible =", r["feasible"], "| risk_ratio = %.3f" % r["risk_ratio"])
