"""demo_moo.py - Chấm lại baseline, kiểm tra xung đột (mục 3), chạy NSGA-II và MO-ACO, báo cáo HV / IGD."""
import argparse
import os
import time

import numpy as np

from common import evaluate, make_random_instance
from h_first import h_first
from heu_greedy import heu_greedy
from aco import mo_aco
from common import (evaluate, hypervolume_2d, high_risk_mask,
                    igd, pareto_filter, save_routes)
from na_greedy import na_greedy
from ga import ga_scalar, nsga2
from svrec import svrec
from zigzag import zigzag


def conflict_check(inst, H, P, G, seed, seeds):
    """Mục 3: GA đơn mục tiêu cho từng mục tiêu rồi đánh giá chéo; ngưỡng 95%."""
    a = ga_scalar(inst, H, 1, 0, pop_size=P, n_generations=G,
                  seed=seed, seeds=seeds)    # tối ưu fcov
    b = ga_scalar(inst, H, 0, 1, pop_size=P, n_generations=G,
                  seed=seed, seeds=seeds)    # tối ưu fben
    # nghiệm cov đạt bao nhiêu % fben tối ưu
    r_ab = a.objs[1] / b.objs[1] if b.objs[1] > 0 else 1.0
    # nghiệm ben đạt bao nhiêu % fcov tối ưu
    r_ba = b.objs[0] / a.objs[0] if a.objs[0] > 0 else 1.0
    narrow = r_ab >= 0.95 and r_ba >= 0.95
    print(
        f"  GA-fcov: fcov={a.objs[0]:.3f} fben={a.objs[1]:.3f} | GA-fben: fcov={b.objs[0]:.3f} fben={b.objs[1]:.3f}")
    print(f"  nghiệm-fcov đạt {r_ab:.1%} fben tối ưu; nghiệm-fben đạt {r_ba:.1%} fcov tối ưu "
          f"=> {'mặt trận HẸP (báo cáo như một kết quả)' if narrow else 'xung đột CÓ THẬT'}")
    return a, b


def run(inst, p, args, tag=""):
    H = high_risk_mask(inst, p)
    print(
        f"\n=== p={p:.0%}: |H|={int(H.sum())}, tổng w_H={inst.w[H].sum():.1f} {tag}")
    os.makedirs(args.out, exist_ok=True)
    sols, pts = {}, {}
    for algo in (zigzag, na_greedy, heu_greedy, svrec):
        sol = algo(inst)
        sols[sol.name] = sol.routes
    sols["H-First"] = h_first(inst, H).routes
    print(f"{'Thuật toán':<12}{'khả thi':<9}{'fcov':<8}{'fben':<8}{'|H∩U|':<7}")
    for name, routes in sols.items():
        r = evaluate(inst, routes, H)
        pts[name] = np.array([r["objs"]])
        save_routes(
            f"{args.out}/{name}_p{int(p*100)}_s{args.seed}.json", name, routes)
        print(
            f"{name:<12}{str(r['feasible']):<9}{r['f_cov_norm']:<8.3f}{r['f_ben_norm']:<8.3f}{r['H_covered']:<7}")

    seeds = [sols[k] for k in ("SVReC", "Na-Greedy", "Heu-Greedy", "H-First")]
    print("Kiểm tra xung đột (GA đơn mục tiêu, đánh giá chéo):")
    a, b = conflict_check(inst, H, args.P, args.G, args.seed, seeds)
    ws = ga_scalar(inst, H, 0.5, 0.5, pop_size=args.P,
                   n_generations=args.G, seed=args.seed, seeds=seeds)
    pts["GA-weighted"] = np.array([ws.objs])

    t0 = time.time()
    arch_n, _ = nsga2(inst, H, pop_size=args.P,
                      n_generations=args.G, seed=args.seed, seeds=seeds)
    tn = time.time() - t0
    t0 = time.time()
    arch_a, _ = mo_aco(inst, H, G=args.iters, A=args.A,
                       seed=args.seed, seeds=seeds)
    ta = time.time() - t0
    pts["NSGA-II"], pts["MO-ACO"] = arch_n.points(), arch_a.points()
    for objs, routes in arch_n.items:
        assert evaluate(inst, routes, H)["feasible"]
    ref = pareto_filter(np.vstack(list(pts.values())))
    print(
        f"\n{'Thuật toán':<12}{'HV':<9}{'IGD':<9}{'|front|':<9}{'thời gian(s)':<12}")
    times = {"NSGA-II": tn, "MO-ACO": ta}
    for name, P_ in pts.items():
        print(f"{name:<12}{hypervolume_2d(P_):<9.4f}{igd(pareto_filter(P_), ref):<9.4f}{len(pareto_filter(P_)):<9}"
              f"{times.get(name, 0.0):<12.2f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        for name, P_ in pts.items():
            P_ = P_[np.argsort(P_[:, 0])]
            plt.plot(P_[:, 0], P_[:, 1], "o-" if len(P_)
                     > 1 else "s", label=name, ms=5)
        plt.xlabel("fcov (chuẩn hoá)")
        plt.ylabel("fben (chuẩn hoá)")
        plt.legend()
        plt.grid(alpha=.3)
        plt.savefig(f"{args.out}/front_p{int(p*100)}_s{args.seed}.png",
                    dpi=130, bbox_inches="tight")
    except Exception as e:
        print("(bỏ qua vẽ hình:", e, ")")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.10)
    ap.add_argument("--sens", action="store_true",
                    help="độ nhạy theo p in {5%%,10%%,20%%}")
    ap.add_argument("--P", type=int, default=30)
    ap.add_argument("--G", type=int, default=30)
    ap.add_argument("--A", type=int, default=20)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    inst = make_random_instance(m=args.m, n=args.m, K=args.K, seed=args.seed)
    print(
        f"Lưới {inst.m}x{inst.n}, K={inst.K}, D={inst.D:.0f}s, tổng rủi ro={inst.w.sum():.1f}")
    for p in ((0.05, 0.10, 0.20) if args.sens else (args.p,)):
        run(inst, p, args)
