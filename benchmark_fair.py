"""benchmark_fair.py - Benchmark cong bang cho ca 10 thuat toan (6 don muc tieu + 4 da muc tieu)
ho tro 3 che do gioi han cong bang:
  --mode time  : gioi han cung thoi gian chay (--time-limit, mac dinh 5.0s)
  --mode evals : gioi han cung so lan danh gia lo trinh NFEs (--max-evals, mac dinh 200)
  --mode gens  : gioi han so the he co dinh (--pop-size, --n-gens, --aco-G, --aco-A)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from aco import aco, mo_aco
from common import (
    Instance,
    UAV,
    evaluate,
    high_risk_mask,
    hypervolume_2d,
    igd,
    pareto_filter,
)
from ga import ga, ga_scalar, nsga2
from h_first import h_first
from heu_greedy import heu_greedy
from na_greedy import na_greedy
from svrec import svrec
from zigzag import zigzag

GREEDY_BASELINES = [
    ("Zig-Zag", zigzag),
    ("Na-Greedy", na_greedy),
    ("Heu-Greedy", heu_greedy),
    ("SVReC", svrec),
]

ALL_NAMES = [
    "Zig-Zag",
    "Na-Greedy",
    "Heu-Greedy",
    "SVReC",
    "ACO",
    "GA",
    "H-First",
    "GA-weighted",
    "MO-ACO",
    "NSGA-II",
]


def _instance_from_dict(data: dict) -> Instance:
    grid = data["grid"]
    u = float(grid["u"])
    w = np.asarray(grid["w"], dtype=float)
    uavs = [
        UAV(
            (int(a["depot"][0]), int(a["depot"][1])),
            float(a["B"]),
            float(a["v_h"]),
            float(a["e_h"]),
            float(a["tau"]),
            float(a["eps"]),
        )
        for a in data["uavs"]
    ]
    return Instance(w=w, u=u, D=float(data["D"]), eta=float(data["eta"]), uavs=uavs)


def _run_one(task: tuple[str, str, str, str, dict]) -> dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    zip_path, member, size, budget, cfg = task
    with zipfile.ZipFile(zip_path) as zf:
        data = json.loads(zf.read(member))
    inst = _instance_from_dict(data)
    H = high_risk_mask(inst, p=cfg["p"])

    row: dict = {
        "member": member,
        "img_id": re.search(r"rescuenet_(\d+)_", member).group(1),
        "Size": size,
        "Budget": budget,
        "Config": f"{size}_{budget}",
        "K": inst.K,
        "D": round(inst.D, 2),
        "Total_W": round(float(inst.w.sum()), 4),
        "Total_WH": round(float(inst.w[H].sum()), 4),
        "H_cells": int(H.sum()),
    }

    pts_map: dict[str, np.ndarray] = {}
    sol_routes: dict[str, list] = {}

    # 1. 4 greedy baselines + H-First (khong gioi han, chay ~ms)
    for algo_name, fn in GREEDY_BASELINES:
        t0 = time.perf_counter()
        sol = fn(inst)
        dt = time.perf_counter() - t0
        ev = evaluate(inst, sol.routes, H)
        sol_routes[algo_name] = sol.routes
        pts_map[algo_name] = np.array([ev["objs"]], dtype=float)
        row[f"{algo_name}_fcov"] = round(float(ev["f_cov"]), 4)
        row[f"{algo_name}_fcov_norm"] = round(float(ev["f_cov_norm"]), 6)
        row[f"{algo_name}_fben"] = round(float(ev["f_ben"]), 4)
        row[f"{algo_name}_fben_norm"] = round(float(ev["f_ben_norm"]), 6)
        row[f"{algo_name}_H_cov"] = int(ev["H_covered"])
        row[f"{algo_name}_front"] = 1
        row[f"{algo_name}_time"] = round(dt, 4)
        row[f"{algo_name}_feasible"] = bool(ev["feasible"])

    t0 = time.perf_counter()
    h_sol = h_first(inst, H)
    dt_hf = time.perf_counter() - t0
    ev_hf = evaluate(inst, h_sol.routes, H)
    sol_routes["H-First"] = h_sol.routes
    pts_map["H-First"] = np.array([ev_hf["objs"]], dtype=float)
    row["H-First_fcov"] = round(float(ev_hf["f_cov"]), 4)
    row["H-First_fcov_norm"] = round(float(ev_hf["f_cov_norm"]), 6)
    row["H-First_fben"] = round(float(ev_hf["f_ben"]), 4)
    row["H-First_fben_norm"] = round(float(ev_hf["f_ben_norm"]), 6)
    row["H-First_H_cov"] = int(ev_hf["H_covered"])
    row["H-First_front"] = 1
    row["H-First_time"] = round(dt_hf, 4)
    row["H-First_feasible"] = bool(ev_hf["feasible"])

    seeds = [
        sol_routes["SVReC"],
        sol_routes["Na-Greedy"],
        sol_routes["Heu-Greedy"],
        sol_routes["H-First"],
    ]

    mode = cfg["mode"]
    seed = cfg["seed"]
    if mode == "time":
        tl, me = cfg["time_limit"], None
        aco_G, aco_A = 1000, cfg["aco_A"]
        pop_size, n_gens = cfg["pop_size"], 1000
    elif mode == "evals":
        tl, me = None, cfg["max_evals"]
        aco_G, aco_A = 1000, cfg["aco_A"]
        pop_size, n_gens = cfg["pop_size"], 1000
    else:
        tl, me = None, None
        aco_G, aco_A = cfg["aco_G"], cfg["aco_A"]
        pop_size, n_gens = cfg["pop_size"], cfg["n_gens"]

    # 2. ACO, GA, GA-weighted (1 nghiem)
    single_meta = [
        ("ACO", lambda: aco(inst, G=aco_G, A=aco_A, seed=seed, time_limit=tl, max_evals=me).routes),
        ("GA", lambda: ga(inst, pop_size=pop_size, n_generations=n_gens, seed=seed, time_limit=tl, max_evals=me).routes),
        (
            "GA-weighted",
            lambda: ga_scalar(
                inst, H, 0.5, 0.5, pop_size=pop_size, n_generations=n_gens,
                seed=seed, seeds=seeds, time_limit=tl, max_evals=me
            ).routes,
        ),
    ]
    for mname, fn in single_meta:
        t0 = time.perf_counter()
        routes = fn()
        dt = time.perf_counter() - t0
        ev = evaluate(inst, routes, H)
        pts_map[mname] = np.array([ev["objs"]], dtype=float)
        row[f"{mname}_fcov"] = round(float(ev["f_cov"]), 4)
        row[f"{mname}_fcov_norm"] = round(float(ev["f_cov_norm"]), 6)
        row[f"{mname}_fben"] = round(float(ev["f_ben"]), 4)
        row[f"{mname}_fben_norm"] = round(float(ev["f_ben_norm"]), 6)
        row[f"{mname}_H_cov"] = int(ev["H_covered"])
        row[f"{mname}_front"] = 1
        row[f"{mname}_time"] = round(dt, 4)
        row[f"{mname}_feasible"] = bool(ev["feasible"])

    # 3. MO-ACO, NSGA-II (mat tran Pareto)
    multi_meta = [
        ("MO-ACO", lambda: mo_aco(inst, H, G=aco_G, A=aco_A, seed=seed, seeds=seeds, time_limit=tl, max_evals=me)),
        (
            "NSGA-II",
            lambda: nsga2(
                inst, H, pop_size=pop_size, n_generations=n_gens, seed=seed,
                seeds=seeds, time_limit=tl, max_evals=me
            ),
        ),
    ]
    for mname, mfn in multi_meta:
        t0 = time.perf_counter()
        arch, _ = mfn()
        dt_m = time.perf_counter() - t0
        evs = [evaluate(inst, r, H) for _, r in arch.items]
        pts = arch.points()
        pts_map[mname] = pts
        row[f"{mname}_fcov"] = round(max(e["f_cov"] for e in evs), 4) if evs else 0.0
        row[f"{mname}_fcov_norm"] = round(max(e["f_cov_norm"] for e in evs), 6) if evs else 0.0
        row[f"{mname}_fben"] = round(max(e["f_ben"] for e in evs), 4) if evs else 0.0
        row[f"{mname}_fben_norm"] = round(max(e["f_ben_norm"] for e in evs), 6) if evs else 0.0
        row[f"{mname}_H_cov"] = max(e["H_covered"] for e in evs) if evs else 0
        row[f"{mname}_front"] = int(len(pareto_filter(pts)))
        row[f"{mname}_time"] = round(dt_m, 4)
        row[f"{mname}_feasible"] = all(bool(e["feasible"]) for e in evs)

    ref_front = pareto_filter(np.vstack(list(pts_map.values())))
    for aname in ALL_NAMES:
        P_ = pts_map[aname]
        row[f"{aname}_HV"] = round(float(hypervolume_2d(P_)), 6)
        row[f"{aname}_IGD"] = round(float(igd(pareto_filter(P_), ref_front)), 6)

    return row


def _safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> str:
    if len(a) < 2 or np.allclose(a, b):
        return "1.0"
    return f"{wilcoxon(a, b).pvalue:.2e}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark cong bang cho 10 thuat toan tren RescueNet")
    ap.add_argument("--mode", choices=["time", "evals", "gens"], default="time")
    ap.add_argument("--time-limit", type=float, default=5.0)
    ap.add_argument("--max-evals", type=int, default=200)
    ap.add_argument("--pop-size", type=int, default=12)
    ap.add_argument("--n-gens", type=int, default=12)
    ap.add_argument("--aco-G", type=int, default=10)
    ap.add_argument("--aco-A", type=int, default=20)
    ap.add_argument("--n-imgs", type=int, default=30)
    ap.add_argument(
        "--configs",
        type=str,
        default="50x50_blow,50x50_bmedium,50x50_bhigh,100x100_blow,100x100_bmedium,100x100_bhigh",
    )
    ap.add_argument("--p", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out-prefix", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    workers = args.workers if args.workers is not None else (1 if args.mode == "time" else 8)
    if args.out_prefix:
        prefix = args.out_prefix
    elif args.mode == "time":
        tl_str = f"{args.time_limit:g}".replace(".", "p")
        prefix = f"benchmark_fair_time{tl_str}s"
    elif args.mode == "evals":
        prefix = f"benchmark_fair_evals{args.max_evals}"
    else:
        prefix = f"benchmark_fair_gens_P{args.pop_size}G{args.n_gens}"

    z50 = CODE_DIR / "rescuenet_50x50.zip"
    z100 = CODE_DIR / "rescuenet_100x100.zip"
    budgets_all = ["blow", "bmedium", "bhigh"]
    seed_tag = f"_s{args.seed}.json"

    def get_map(zpath: Path) -> dict[tuple[str, str], str]:
        mapping: dict[tuple[str, str], str] = {}
        with zipfile.ZipFile(zpath) as zf:
            for n in zf.namelist():
                if "/test/" in n and n.endswith(seed_tag):
                    m = re.search(r"test/K6_(\w+)/rescuenet_(\d+)_", n)
                    if m:
                        mapping[(m.group(1), m.group(2))] = n
        return mapping

    m50 = get_map(z50)
    m100 = get_map(z100)

    common_imgs = sorted(
        set.intersection(
            *[set(img for (b, img) in m50 if b == bg) for bg in budgets_all],
            *[set(img for (b, img) in m100 if b == bg) for bg in budgets_all],
        ),
        key=int,
    )[: args.n_imgs]

    cfg_list = [c.strip() for c in args.configs.split(",") if c.strip()]
    cfg_dict = {
        "mode": args.mode,
        "time_limit": args.time_limit,
        "max_evals": args.max_evals,
        "pop_size": args.pop_size,
        "n_gens": args.n_gens,
        "aco_G": args.aco_G,
        "aco_A": args.aco_A,
        "p": args.p,
        "seed": args.seed,
    }

    tasks = []
    for cfg_name in cfg_list:
        size, bg = cfg_name.split("_", 1)
        zpath, mp = (z50, m50) if size == "50x50" else (z100, m100)
        for img in common_imgs:
            tasks.append((str(zpath), mp[(bg, img)], size, bg, cfg_dict))

    print(
        f"Running mode={args.mode} on {len(tasks)} instances "
        f"({len(common_imgs)} imgs x {len(cfg_list)} configs) with {workers} worker(s)..."
    )
    t_start = time.perf_counter()
    rows = []
    if workers <= 1:
        for idx, t in enumerate(tasks, 1):
            rows.append(_run_one(t))
            if idx % 10 == 0 or idx == len(tasks):
                print(f"  Completed {idx}/{len(tasks)} in {time.perf_counter() - t_start:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one, t) for t in tasks]
            for idx, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if idx % 15 == 0 or idx == len(tasks):
                    print(f"  Completed {idx}/{len(tasks)} in {time.perf_counter() - t_start:.1f}s", flush=True)

    rows.sort(
        key=lambda r: (
            0 if r["Size"] == "50x50" else 1,
            budgets_all.index(r["Budget"]),
            int(r["img_id"]),
        )
    )

    raw_path = CODE_DIR / f"{prefix}_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_rows = []
    for cfg_name in cfg_list:
        sub = [r for r in rows if r["Config"] == cfg_name]
        if not sub:
            continue
        srow: dict = {
            "Config": cfg_name,
            "Mode": args.mode,
            "N_imgs": len(sub),
            "Mean_Total_W": round(float(np.mean([r["Total_W"] for r in sub])), 2),
            "Mean_Total_WH": round(float(np.mean([r["Total_WH"] for r in sub])), 2),
            "All_Feasible": all(r[f"{a}_feasible"] for r in sub for a in ALL_NAMES),
        }
        for a in ALL_NAMES:
            fcov = np.array([r[f"{a}_fcov"] for r in sub], dtype=float)
            fcov_pct = np.array([r[f"{a}_fcov_norm"] for r in sub], dtype=float) * 100.0
            fben = np.array([r[f"{a}_fben"] for r in sub], dtype=float)
            fben_pct = np.array([r[f"{a}_fben_norm"] for r in sub], dtype=float) * 100.0
            hv = np.array([r[f"{a}_HV"] for r in sub], dtype=float)
            ig = np.array([r[f"{a}_IGD"] for r in sub], dtype=float)
            fr = np.array([r[f"{a}_front"] for r in sub], dtype=float)
            tm = np.array([r[f"{a}_time"] for r in sub], dtype=float)

            srow[f"{a}_fcov_mean"] = round(float(fcov.mean()), 2)
            srow[f"{a}_fcov_pct_mean"] = round(float(fcov_pct.mean()), 2)
            srow[f"{a}_fben_mean"] = round(float(fben.mean()), 2)
            srow[f"{a}_fben_pct_mean"] = round(float(fben_pct.mean()), 2)
            srow[f"{a}_HV_mean"] = round(float(hv.mean()), 5)
            srow[f"{a}_IGD_mean"] = round(float(ig.mean()), 5)
            srow[f"{a}_front_mean"] = round(float(fr.mean()), 2)
            srow[f"{a}_time_mean"] = round(float(tm.mean()), 3)

        srow["p_NSGA2_vs_GA_fcov"] = _safe_wilcoxon(
            np.array([r["NSGA-II_fcov"] for r in sub]),
            np.array([r["GA_fcov"] for r in sub]),
        )
        srow["p_NSGA2_vs_GA_HV"] = _safe_wilcoxon(
            np.array([r["NSGA-II_HV"] for r in sub]),
            np.array([r["GA_HV"] for r in sub]),
        )
        srow["p_NSGA2_vs_MOACO_HV"] = _safe_wilcoxon(
            np.array([r["NSGA-II_HV"] for r in sub]),
            np.array([r["MO-ACO_HV"] for r in sub]),
        )
        srow["p_MOACO_vs_ACO_fcov"] = _safe_wilcoxon(
            np.array([r["MO-ACO_fcov"] for r in sub]),
            np.array([r["ACO_fcov"] for r in sub]),
        )
        summary_rows.append(srow)

    sum_path = CODE_DIR / f"{prefix}_summary.csv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Done in {time.perf_counter() - t_start:.1f}s! Saved {raw_path.name} and {sum_path.name}")


if __name__ == "__main__":
    main()
