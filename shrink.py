"""Post-hoc decode tuning on a saved OOF file (CPU, cheap)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import NAX, NPROBE, PARK_GRID, final_score, decode_park_batch, decode_plan
from train2 import marg_blend
from scipy.ndimage import gaussian_filter

f = np.load(sys.argv[1])
m = f["used"]
SEL, POS, HEAT, PM = f["sel"][m], f["pos"][m], f["heat"][m], f["pm"][m]
ypos, ypark = f["ypos"][m], f["ypark"][m]
ypos_all, ypark_all = f["ypos"], f["ypark"]
tr = ~m
n = int(m.sum())

# prior marginal from train rows
cnt = np.zeros(NPROBE)
for r in ypos_all[tr]:
    cnt[r] += 1
prior_m = cnt / tr.sum()                      # sums to 2 per axis
prior_pos = np.zeros((4, NPROBE))
for r in ypos_all[tr]:
    for j in range(4):
        prior_pos[j, r[j]] += 1
prior_pos /= tr.sum()

c = (np.arange(PARK_GRID) + 0.5) / PARK_GRID
ph = np.zeros((PARK_GRID, PARK_GRID))
for u, v in ypark_all[tr]:
    ph[np.argmin(np.abs(c - v)), np.argmin(np.abs(c - u))] += 1
ph = gaussian_filter(ph, 2.0); ph /= ph.sum()
park = decode_park_batch(np.repeat(ph[None], n, 0))

base = marg_blend(SEL, POS, PM, (.4, .2, .4))
print("beta  gamma   sel     order   plan    total")
best = None
for beta in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
    M = base + beta * prior_m[None]
    for gamma in (0.0, 0.1, 0.25, 0.5):
        Q = POS + gamma * prior_pos[None]
        pl = np.stack([decode_plan(M[i], Q[i]) for i in range(n)])
        sel = np.mean([len(set(pl[i]) & set(ypos[i])) / 4 for i in range(n)])
        order = np.mean(((pl == ypos) * np.array([.4, .3, .2, .1])).sum(1))
        tot, ps, pk = final_score(pl, park, ypos, ypark)
        if best is None or tot > best[0]:
            best = (tot, beta, gamma)
        print(f"{beta:4.1f}  {gamma:5.2f}  {sel:.4f}  {order:.4f}  {ps:6.3f}  {tot:6.3f}")
print("BEST", best)

# per-position temperature on q (sharpen/flatten the order term)
print("\ntemperature on position marginals (beta=best):")
M = base + best[1] * prior_m[None]
for T in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    Q = POS ** (1.0 / T)
    Q /= Q.sum(-1, keepdims=True)
    pl = np.stack([decode_plan(M[i], Q[i]) for i in range(n)])
    order = np.mean(((pl == ypos) * np.array([.4, .3, .2, .1])).sum(1))
    tot, ps, pk = final_score(pl, park, ypos, ypark)
    print(f"  T={T}: order {order:.4f} plan {ps:.3f} total {tot:.3f}")
