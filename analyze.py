"""Diagnostics on a saved OOF file."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import NAX, NPROBE, final_score, decode_park_batch
from train2 import marg_blend, decode_all
from sklearn.metrics import roc_auc_score

f = np.load(sys.argv[1])
m = f["used"]
SEL, POS, HEAT = f["sel"][m], f["pos"][m], f["heat"][m]
PM = f["pm"][m] if "pm" in f else None
ypos, ypark = f["ypos"][m], f["ypark"][m]
n = m.sum()
Y = np.zeros((n, NPROBE)); Y[np.arange(n)[:, None], ypos] = 1

print(f"rows {n}")
cands = {"sel": 2 * SEL, "possum": POS.sum(1)}
if PM is not None:
    cands["pair"] = PM
for name, M in cands.items():
    for a in (0, 1):
        s = slice(a * NAX, (a + 1) * NAX)
        print(f"  AUC {name} axis_{a}: {roc_auc_score(Y[:, s].ravel(), M[:, s].ravel()):.4f}")

park = decode_park_batch(HEAT)
d = np.sqrt(((park - ypark) ** 2).sum(1))
print(f"park: mean dist {d.mean():.4f} median {np.median(d):.4f} score {np.maximum(0,1-d/0.35).mean():.4f}")
print(f"const park score {np.maximum(0,1-np.sqrt(((ypark-np.array([0.49,0.5]))**2).sum(1))/0.35).mean():.4f}")

best = None
for w in [(1,0,0),(0,1,0),(0,0,1),(.5,0,.5),(.4,.2,.4),(.34,.33,.33),(.6,.2,.2),(.2,.2,.6),(.5,.25,.25),(.25,.25,.5)]:
    if PM is None and w[2] > 0:
        continue
    M = marg_blend(SEL, POS, PM if PM is not None else np.zeros_like(SEL), w)
    pl = decode_all(M, POS)
    tot, ps, pk = final_score(pl, park, ypos, ypark)
    sel_frac = np.mean([len(set(pl[i]) & set(ypos[i]))/4 for i in range(n)])
    order = np.mean((pl == ypos) * np.array([.4,.3,.2,.1]))*4
    if best is None or tot > best[0]: best = (tot, w)
    print(f"  w={w} total {tot:.3f} plan {ps:.3f} park {pk:.3f} | sel {sel_frac:.4f} order {order:.4f}")
print("BEST", best)

# oracle-order: what if ordering were perfect given our set?
M = marg_blend(SEL, POS, PM if PM is not None else np.zeros_like(SEL), best[1])
pl = decode_all(M, POS)
sel_frac = np.mean([len(set(pl[i]) & set(ypos[i]))/4 for i in range(n)])
oracle_plan = 0.8*sel_frac + 0.2*np.mean([sum(.4*(0),) if False else 0 for _ in range(1)])
print(f"\nselection fraction {sel_frac:.4f} -> plan w/ random order approx {0.8*sel_frac+0.2*0.25*sel_frac:.4f}")
print(f"total with perfect park: {100*(0.7*(0.8*sel_frac)+0.3):.2f}")
