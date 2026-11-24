"""
runner.py - Công cụ chạy thực nghiệm hàng loạt trên các file test (*.json)
theo đúng định dạng docs/instance_format.md và xuất file nghiệm (*.solution.json).

Cách dùng:
  1. Chạy trên 1 file JSON cụ thể:
     python project/code/runner.py path/to/instance.json --out-dir solutions/

  2. Chạy toàn bộ các file *.json trong 1 thư mục:
     python project/code/runner.py path/to/test_folder/ --out-dir solutions/ --csv summary.csv

  3. Sinh nhanh bộ test mẫu chuẩn JSON (gồm toy_3x3_K2 trong instance_format.md và lưới 30x30, 50x50)
     rồi chạy kiểm tra toàn bộ pipeline:
     python project/code/runner.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from aco import aco
from common import (
    Instance,
    Solution,
    UAV,
    evaluate,
    load_instance,
    load_solution,
    make_random_instance,
    save_instance,
    save_solution,
)
from ga import ga
from heu_greedy import heu_greedy
from na_greedy import na_greedy
from svrec import svrec
from zigzag import zigzag

ALGORITHMS: Dict[str, Callable[[Instance], Solution]] = {
    "zigzag": zigzag,
    "na_greedy": na_greedy,
    "heu_greedy": heu_greedy,
    "svrec": svrec,
    "aco": aco,
    "ga": ga,
}


def run_on_instance(
    inst: Instance,
    algos: List[str],
    out_dir: Path | None = None,
) -> List[Dict]:
    rows: List[Dict] = []
    print(
        f"\n=== Instance: {inst.name} ({inst.m}x{inst.n}, K={inst.K}, D={inst.D:.1f}s, "
        f"sum_w={inst.w.sum():.2f}) ==="
    )
    print(
        f"{'Thuật toán':<12}{'Khả thi':<9}{'Tỉ lệ phủ':<12}{'ΣR (f)':<10}"
        f"{'Cover σ':<10}{'Số ô':<7}{'Thời gian (s)':<14}"
    )
    print("-" * 74)

    for key in algos:
        fn = ALGORITHMS[key]
        t0 = time.perf_counter()
        sol = fn(inst)
        dt = time.perf_counter() - t0
        sol.runtime_sec = dt

        res = evaluate(inst, sol.routes)
        print(
            f"{sol.name:<12}{str(res['feasible']):<9}{res['risk_ratio']:<12.4f}"
            f"{res['sum_R']:<10.2f}{res['cover_rate']:<10.4f}{res['cells']:<7}{dt:<14.4f}"
        )
        if not res["feasible"]:
            print(f"   [!] Vi phạm: {res['violations']}")

        if out_dir is not None:
            sol_path = out_dir / f"{inst.name}.{key}.solution.json"
            save_solution(sol, sol_path, instance_name=inst.name, runtime_sec=dt)
            # Kiểm tra đọc ngược lại từ file .solution.json
            loaded_sol = load_solution(sol_path)
            chk = evaluate(inst, loaded_sol.routes)
            assert chk["feasible"] == res["feasible"] and abs(chk["f"] - res["f"]) < 1e-9

        rows.append(
            {
                "instance": inst.name,
                "m": inst.m,
                "n": inst.n,
                "K": inst.K,
                "algorithm": sol.name,
                "feasible": res["feasible"],
                "f": round(res["f"], 4),
                "risk_ratio": round(res["risk_ratio"], 4),
                "cover_rate": round(res["cover_rate"], 4),
                "cells": res["cells"],
                "runtime_sec": round(dt, 4),
            }
        )
    return rows


def run_self_test(base_dir: Path) -> None:
    """Tạo các file JSON mẫu (bao gồm ví dụ toy_3x3_K2 trong docs/instance_format.md) và kiểm thử."""
    sample_dir = base_dir / "sample_instances"
    sol_dir = base_dir / "sample_solutions"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sol_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ví dụ toy_3x3_K2 từ docs/instance_format.md
    toy = Instance(
        name="toy_3x3_K2",
        source={"dataset": "synthetic", "seed": 1},
        w=[[0.0, 0.2, 0.9], [0.1, 0.8, 0.7], [0.0, 0.1, 0.3]],
        u=40.0,
        D=600.0,
        eta=0.9,
        uavs=[
            UAV(depot=(0, 0), B=300.0, v_h=15.0, e_h=1.0, tau=2.0, eps=0.0),
            UAV(depot=(0, 0), B=280.0, v_h=15.0, e_h=1.0, tau=2.0, eps=0.0),
        ],
        rho={"intact": 0.0, "flooded": 0.5, "collapsed": 1.0},
    )
    toy_path = sample_dir / "toy_3x3_K2.json"
    save_instance(toy, toy_path)

    # 2. Instance 30x30 và 50x50
    inst30 = make_random_instance(m=30, n=30, K=3, seed=1)
    path30 = sample_dir / f"{inst30.name}.json"
    save_instance(inst30, path30)

    inst50 = make_random_instance(m=50, n=50, K=3, seed=0)
    path50 = sample_dir / f"{inst50.name}.json"
    save_instance(inst50, path50)

    all_rows: List[Dict] = []
    for p in (toy_path, path30, path50):
        loaded = load_instance(p, check_reachability=True)
        all_rows.extend(run_on_instance(loaded, list(ALGORITHMS.keys()), out_dir=sol_dir))

    csv_path = sol_dir / "summary_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[OK] Đã lưu các file instance mẫu tại: {sample_dir}")
    print(f"[OK] Đã lưu các file *.solution.json và bảng tổng hợp tại: {sol_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Runner cho các thuật toán Multi-UAV Risk Coverage")
    parser.add_argument("target", nargs="?", help="Đường dẫn tới file *.json hoặc thư mục chứa *.json")
    parser.add_argument("--algos", nargs="+", default=list(ALGORITHMS.keys()), choices=list(ALGORITHMS.keys()))
    parser.add_argument("--out-dir", type=str, default=None, help="Thư mục xuất file *.solution.json")
    parser.add_argument("--csv", type=str, default=None, help="Đường dẫn file CSV tổng hợp kết quả")
    parser.add_argument("--self-test", action="store_true", help="Sinh test mẫu theo instance_format.md và chạy thử")
    args = parser.parse_args()

    if args.self_test or not args.target:
        run_self_test(Path(__file__).resolve().parent)
        return

    target = Path(args.target)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = [target] if target.is_file() else sorted(p for p in target.glob("*.json") if not p.name.endswith(".solution.json"))
    all_rows: List[Dict] = []
    for fpath in files:
        inst = load_instance(fpath)
        all_rows.extend(run_on_instance(inst, args.algos, out_dir=out_dir))

    if args.csv and all_rows:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[OK] Đã xuất kết quả CSV ra: {csv_path}")


if __name__ == "__main__":
    main()
