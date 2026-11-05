"""demo.py - Chạy cả 4 thuật toán trên instance giả lập và in các chỉ số (mục 10)."""
import sys
import time

from common import evaluate, make_random_instance
from heu_greedy import heu_greedy
from na_greedy import na_greedy
from svrec import svrec
from zigzag import zigzag

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    inst = make_random_instance(m=50, n=50, K=3, seed=seed)
    print(f"Lưới {inst.m}x{inst.n}, K={inst.K}, D={inst.D:.0f}s, tổng rủi ro={inst.w.sum():.1f}")
    print(f"{'Thuật toán':<12}{'khả thi':<9}{'rủi ro phủ':<12}{'cover σ':<10}{'số ô':<7}{'thời gian chạy':<14}")
    for algo in (zigzag, na_greedy, heu_greedy, svrec):
        t0 = time.time()
        sol = algo(inst)
        dt = time.time() - t0
        r = evaluate(inst, sol.routes)
        print(f"{sol.name:<12}{str(r['feasible']):<9}{r['risk_ratio']:<12.3f}{r['cover_rate']:<10.3f}"
              f"{r['cells']:<7}{dt:<14.3f}")
        if not r["feasible"]:
            print("   vi phạm:", r["violations"])
