"""demo.py - Chạy cả 6 thuật toán (4 Baseline + ACO + GA) trên instance giả lập hoặc file JSON và in các chỉ số (mục 10)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from aco import aco
from common import evaluate, load_instance, make_random_instance
from ga import ga
from heu_greedy import heu_greedy
from na_greedy import na_greedy
from svrec import svrec
from zigzag import zigzag

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    if arg.endswith(".json") and Path(arg).exists():
        inst = load_instance(arg)
    else:
        seed = int(arg)
        inst = make_random_instance(m=50, n=50, K=3, seed=seed)

    print(f"Instance: {inst.name} | Lưới {inst.m}x{inst.n}, K={inst.K}, D={inst.D:.0f}s, tổng rủi ro={inst.w.sum():.1f}")
    print(f"{'Thuật toán':<12}{'khả thi':<9}{'rủi ro phủ':<12}{'ΣR':<10}{'cover σ':<10}{'số ô':<7}{'thời gian chạy':<14}")
    print("-" * 74)
    for algo in (zigzag, na_greedy, heu_greedy, svrec, aco, ga):
        t0 = time.time()
        sol = algo(inst)
        dt = time.time() - t0
        r = evaluate(inst, sol.routes)
        print(
            f"{sol.name:<12}{str(r['feasible']):<9}{r['risk_ratio']:<12.3f}"
            f"{r['sum_R']:<10.2f}{r['cover_rate']:<10.3f}{r['cells']:<7}{dt:<14.3f}"
        )
        if not r["feasible"]:
            print("   vi phạm:", r["violations"])
