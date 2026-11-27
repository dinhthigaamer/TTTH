"""plot_convergence_120s.py - Chay cac thuat toan lap lien tuc tu 0 den 120 giay
tren 3 anh RescueNet 100x100_bmedium (10778, 10779, 10780) va ve bieu do hoi tu.
"""
from __future__ import annotations

import csv
import json
import sys
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

CODE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = CODE_DIR / "plots"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from aco import aco, mo_aco
from common import (
    Instance,
    ParetoArchive,
    UAV,
    evaluate,
    high_risk_mask,
    hypervolume_2d,
)
from ga import _decode, ga, ga_scalar, nsga2
from h_first import h_first
from heu_greedy import heu_greedy
from na_greedy import na_greedy
from svrec import svrec

T_MAX = 120.0
IMG_IDS = ["10778", "10779", "10780"]
GRID_T = sorted(set([0.0, 1.0, 5.0, 15.0] + list(range(0, 121, 2))))
MILESTONES = [5, 15, 30, 60, 120]
TABLE_TIMES = [1, 2, 5, 10, 15, 30, 60, 90, 120]
ALGO_ORDER = [
    "SVReC",
    "H-First",
    "Random-Restart",
    "ACO",
    "GA",
    "GA-weighted",
    "MO-ACO",
    "NSGA-II",
]


def _load_img(img_id: str) -> Instance:
    zpath = CODE_DIR / "rescuenet_100x100.zip"
    member = f"rescuenet_100x100/test/K6_bmedium/rescuenet_{img_id}_100x100_K6_bmedium_s42.json"
    with zipfile.ZipFile(zpath) as zf:
        data = json.loads(zf.read(member))
    g = data["grid"]
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
    return Instance(w=np.asarray(g["w"], dtype=float), u=float(g["u"]), D=float(data["D"]), eta=float(data["eta"]), uavs=uavs)


def _run_random_restart_120s(inst: Instance, H: np.ndarray, seeds: list, time_limit: float = T_MAX, seed: int = 42) -> list:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    r_th = float(inst.w.mean())
    arch = ParetoArchive(40)
    for r in seeds:
        arch.add(evaluate(inst, r, H)["objs"], r)
    trace = [(0.0, arch.points().copy())]
    evals = 0
    while evals == 0 or time.perf_counter() - t0 < time_limit:
        prio = inst.w * rng.uniform(0.7, 1.3, size=inst.w.shape)
        gamma = float(rng.uniform(0.5, 1.5))
        rds = _decode(inst, prio, r_th, H, gamma)
        objs = evaluate(inst, rds, H)["objs"]
        evals += 1
        if arch.add(objs, rds):
            trace.append((time.perf_counter() - t0, arch.points().copy()))
    return trace


def _step_sample_archive(trace: list, grid: list[float]) -> tuple[np.ndarray, np.ndarray]:
    hvs, fcs = [], []
    j = 0
    cur_pts = trace[0][1] if trace else np.zeros((0, 2))
    for t in grid:
        while j + 1 < len(trace) and trace[j + 1][0] <= t:
            j += 1
            cur_pts = trace[j][1]
        if len(cur_pts) > 0:
            hvs.append(float(hypervolume_2d(cur_pts)))
            fcs.append(float(cur_pts[:, 0].max()) * 100.0)
        else:
            hvs.append(0.0)
            fcs.append(0.0)
    return np.asarray(hvs, dtype=float), np.asarray(fcs, dtype=float)


def _step_sample_single(trace: list, grid: list[float]) -> tuple[np.ndarray, np.ndarray]:
    hvs, fcs = [], []
    j = -1
    for t in grid:
        while j + 1 < len(trace) and trace[j + 1][0] <= t:
            j += 1
        if j >= 0:
            objs = trace[j][1]
            hvs.append(float(hypervolume_2d(np.array([objs], dtype=float))))
            fcs.append(float(objs[0]) * 100.0)
        else:
            hvs.append(0.0)
            fcs.append(0.0)
    return np.asarray(hvs, dtype=float), np.asarray(fcs, dtype=float)


def _run_one_image(img_id: str) -> dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t_img0 = time.perf_counter()
    inst = _load_img(img_id)
    H = high_risk_mask(inst, p=0.10)

    t0 = time.perf_counter()
    sv_sol = svrec(inst)
    dt_sv = time.perf_counter() - t0
    sv_objs = evaluate(inst, sv_sol.routes, H)["objs"]

    t0 = time.perf_counter()
    hf_sol = h_first(inst, H)
    dt_hf = time.perf_counter() - t0
    hf_objs = evaluate(inst, hf_sol.routes, H)["objs"]

    na_routes = na_greedy(inst).routes
    heu_routes = heu_greedy(inst).routes
    seeds = [sv_sol.routes, na_routes, heu_routes, hf_sol.routes]

    seed_arch = ParetoArchive(40)
    for r in seeds:
        seed_arch.add(evaluate(inst, r, H)["objs"], r)
    hv_seed = float(hypervolume_2d(seed_arch.points()))

    res_hv: dict[str, np.ndarray] = {}
    res_fc: dict[str, np.ndarray] = {}

    res_hv["SVReC"], res_fc["SVReC"] = _step_sample_single([(dt_sv, sv_objs)], GRID_T)
    res_hv["H-First"], res_fc["H-First"] = _step_sample_single([(dt_hf, hf_objs)], GRID_T)

    tr_rr = _run_random_restart_120s(inst, H, seeds, time_limit=T_MAX, seed=42)
    res_hv["Random-Restart"], res_fc["Random-Restart"] = _step_sample_archive(tr_rr, GRID_T)

    tr_aco: list = []
    aco(inst, G=10000, A=20, seed=42, time_limit=T_MAX, trace=tr_aco)
    tr_aco_objs = [(ts, evaluate(inst, rds, H)["objs"]) for ts, rds in tr_aco]
    res_hv["ACO"], res_fc["ACO"] = _step_sample_single(tr_aco_objs, GRID_T)

    tr_ga: list = []
    ga(inst, pop_size=12, n_generations=10000, seed=42, time_limit=T_MAX, trace=tr_ga)
    tr_ga_objs = [(ts, evaluate(inst, rds, H)["objs"]) for ts, rds in tr_ga]
    res_hv["GA"], res_fc["GA"] = _step_sample_single(tr_ga_objs, GRID_T)

    tr_gw: list = []
    ga_scalar(inst, H, 0.5, 0.5, pop_size=12, n_generations=10000, seed=42, seeds=seeds, time_limit=T_MAX, trace=tr_gw)
    res_hv["GA-weighted"], res_fc["GA-weighted"] = _step_sample_single(tr_gw, GRID_T)

    tr_ma: list = []
    mo_aco(inst, H, G=10000, A=20, seed=42, seeds=seeds, time_limit=T_MAX, trace=tr_ma)
    res_hv["MO-ACO"], res_fc["MO-ACO"] = _step_sample_archive(tr_ma, GRID_T)

    tr_ns: list = []
    nsga2(inst, H, pop_size=12, n_generations=10000, seed=42, seeds=seeds, time_limit=T_MAX, trace=tr_ns)
    res_hv["NSGA-II"], res_fc["NSGA-II"] = _step_sample_archive(tr_ns, GRID_T)

    delta_hv = {}
    for a in ("Random-Restart", "MO-ACO", "NSGA-II"):
        delta_hv[a] = (res_hv[a] - hv_seed) / hv_seed * 100.0 if hv_seed > 0 else np.zeros_like(res_hv[a])

    print(
        f"  Finished img {img_id} in {time.perf_counter() - t_img0:.1f}s | "
        f"NSGA-II: {hv_seed:.5f} -> {res_hv['NSGA-II'][-1]:.5f} (+{delta_hv['NSGA-II'][-1]:.2f}%, {len(tr_ns)} upd) | "
        f"MO-ACO: {res_hv['MO-ACO'][-1]:.5f} ({len(tr_ma)} upd)",
        flush=True,
    )
    return {"img_id": img_id, "hv": res_hv, "fc": res_fc, "delta": delta_hv, "hv_seed": hv_seed}


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running 120s convergence on {len(IMG_IDS)} images {IMG_IDS} (100x100_bmedium) in parallel...", flush=True)
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(_run_one_image, IMG_IDS))

    n = float(len(results))
    hv_mean = {a: sum(r["hv"][a] for r in results) / n for a in ALGO_ORDER}
    fc_mean = {a: sum(r["fc"][a] for r in results) / n for a in ALGO_ORDER}
    delta_mean = {a: sum(r["delta"][a] for r in results) / n for a in ("Random-Restart", "MO-ACO", "NSGA-II")}

    styles = {
        "SVReC": ("#2980b9", "--", 1.9),
        "H-First": ("#8e44ad", ":", 2.0),
        "Random-Restart": ("#5d6d7e", "-.", 2.0),
        "ACO": ("#e67e22", "-", 1.9),
        "GA": ("#27ae60", "-", 2.0),
        "GA-weighted": ("#16a085", "-", 2.1),
        "MO-ACO": ("#d35400", "-", 2.2),
        "NSGA-II": ("#c0392b", "-", 2.8),
    }

    # --- Hinh 1: convergence_120s_100x100_bmedium.png ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150)
    for a in ALGO_ORDER:
        col, ls, lw = styles[a]
        ax1.plot(GRID_T, hv_mean[a], label=f"{a} ({hv_mean[a][-1]:.4f})", color=col, linestyle=ls, linewidth=lw)
        ax2.plot(GRID_T, fc_mean[a], label=f"{a} ({fc_mean[a][-1]:.2f}%)", color=col, linestyle=ls, linewidth=lw)

    for m in MILESTONES:
        ax1.axvline(m, color="gray", linestyle="--", linewidth=0.9, alpha=0.55)
        ax2.axvline(m, color="gray", linestyle="--", linewidth=0.9, alpha=0.55)

    ax1.set_title("Hypervolume theo thoi gian (0 -> 120s)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Thoi gian chay thuc te (giay)", fontsize=10)
    ax1.set_ylabel("Hypervolume TB (3 anh)", fontsize=10)
    ax1.set_xlim(0, 120)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8.2, loc="lower right", ncol=2, framealpha=0.92)

    ax2.set_title("fcov (%) theo thoi gian (0 -> 120s)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Thoi gian chay thuc te (giay)", fontsize=10)
    ax2.set_ylabel("fcov (%) TB (3 anh)", fontsize=10)
    ax2.set_xlim(0, 120)
    ax2.set_ylim(22.0, max(fc_mean["NSGA-II"][-1], fc_mean["GA-weighted"][-1]) + 1.5)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=8.2, loc="lower right", ncol=2, framealpha=0.92)

    fig.suptitle("Qua trinh hoi tu 0 -> 120s cua 8 thuat toan — RescueNet 100x100_bmedium (TB 3 anh)", fontsize=12.8, fontweight="bold")
    fig.tight_layout()
    p1 = PLOTS_DIR / "convergence_120s_100x100_bmedium.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # --- Hinh 2: convergence_120s_zoom_top.png ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150)
    top_algos = ["SVReC", "H-First", "Random-Restart", "MO-ACO", "NSGA-II"]
    for a in top_algos:
        col, ls, lw = styles[a]
        axL.plot(GRID_T, hv_mean[a], label=f"{a} ({hv_mean[a][-1]:.5f})", color=col, linestyle=ls, linewidth=lw)

    y_bot = 0.95 * float(hv_mean["SVReC"][-1])
    y_top = float(hv_mean["NSGA-II"][-1])
    axL.set_xlim(0, 120)
    axL.set_ylim(y_bot, y_top + 0.14 * (y_top - y_bot))
    for m in MILESTONES:
        axL.axvline(m, color="gray", linestyle="--", linewidth=0.9, alpha=0.55)
    axL.set_title("Zoom Hypervolume vung top (tu 0.95 x HV(SVReC))", fontsize=11, fontweight="bold")
    axL.set_xlabel("Thoi gian chay thuc te (giay)", fontsize=10)
    axL.set_ylabel("Hypervolume TB (3 anh)", fontsize=10)
    axL.grid(True, linestyle="--", alpha=0.35)
    axL.legend(fontsize=8.5, loc="lower right", framealpha=0.92)

    for a in ("Random-Restart", "MO-ACO", "NSGA-II"):
        col, ls, lw = styles[a]
        axR.plot(GRID_T, delta_mean[a], label=f"{a} (+{delta_mean[a][-1]:.2f}%)", color=col, linestyle=ls, linewidth=lw)
    for m in MILESTONES:
        axR.axvline(m, color="gray", linestyle="--", linewidth=0.9, alpha=0.55)
    axR.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    axR.set_xlim(0, 120)
    axR.set_title("% cai thien Hypervolume vuot tren khoi tao seeds (+Delta HV %)", fontsize=11, fontweight="bold")
    axR.set_xlabel("Thoi gian chay thuc te (giay)", fontsize=10)
    axR.set_ylabel("+Delta HV so voi seeds (%)", fontsize=10)
    axR.grid(True, linestyle="--", alpha=0.35)
    axR.legend(fontsize=8.8, loc="upper left", framealpha=0.92)

    fig.suptitle("Toc do tien hoa vung top (0 -> 120s) — RescueNet 100x100_bmedium (TB 3 anh)", fontsize=12.8, fontweight="bold")
    fig.tight_layout()
    p2 = PLOTS_DIR / "convergence_120s_zoom_top.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)

    # --- Luu bang cac moc thoi gian ---
    table_rows = []
    idx_map = {t: i for i, t in enumerate(GRID_T)}
    for t in TABLE_TIMES:
        ti = idx_map[float(t)]
        for a in ALGO_ORDER:
            dhv = round(float(delta_mean[a][ti]), 3) if a in delta_mean else ""
            table_rows.append(
                {
                    "t_s": t,
                    "Algo": a,
                    "HV_mean": round(float(hv_mean[a][ti]), 5),
                    "fcov_pct_mean": round(float(fc_mean[a][ti]), 2),
                    "delta_HV_pct": dhv,
                }
            )
    csv_path = PLOTS_DIR / "convergence_120s_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)

    print(f"Done in {time.perf_counter() - t_start:.1f}s! Saved {p1.name}, {p2.name}, {csv_path.name}", flush=True)


if __name__ == "__main__":
    main()
