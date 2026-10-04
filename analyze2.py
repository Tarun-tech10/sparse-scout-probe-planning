import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import NAX, NPROBE, PARK_GRID, final_score, decode_park_batch
from train2 import marg_blend, decode_all
from sklearn.metrics import roc_auc_score

f = np.load(sys.argv[1])
m = f["used"]
SEL, POS, HEAT, PM = f["sel"][m], f["pos"][m], f["heat"][m], f["pm"][m]
ypos, ypark = f["ypos"][m], f["ypark"][m]
ypos_all, ypark_all = f["ypos"], f["ypark"]
tr_mask = ~m
n = int(m.sum())
Y = np.zeros((n, NPROBE)); Y[np.arange(n)[:, None], ypos] = 1
print(f"val rows {n}, train rows {tr_mask.sum()}")

for name, M in (("sel", 2 * SEL), ("possum", POS.sum(1)), ("pair", PM),
                ("blend", marg_blend(SEL, POS, PM, (.4, .2, .4)))):
    a0 = roc_auc_score(Y[:, :NAX].ravel(), M[:, :NAX].ravel())
    a1 = roc_auc_score(Y[:, NAX:].ravel(), M[:, NAX:].ravel())
    print(f"  AUC {name:7s} axis0 {a0:.4f} axis1 {a1:.4f}")

# ---- parking -------------------------------------------------------------
c = (np.arange(PARK_GRID) + 0.5) / PARK_GRID
# empirical prior heat-map from the TRAIN rows of this split
prior = np.zeros((PARK_GRID, PARK_GRID))
for u, v in ypark_all[tr_mask]:
    iu = np.argmin(np.abs(c - u)); iv = np.argmin(np.abs(c - v))
    prior[iv, iu] += 1
from scipy.ndimage import gaussian_filter
prior = gaussian_filter(prior, 2.0); prior /= prior.sum()

print("\npark:")
for lam in (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0):
    H = lam * HEAT + (1 - lam) * prior[None]
    H /= H.sum((1, 2), keepdims=True)
    p = decode_park_batch(H)
    d = np.sqrt(((p - ypark) ** 2).sum(1))
    print(f"  lam={lam:.2f}: park {100*np.maximum(0,1-d/0.35).mean():.3f}  meandist {d.mean():.4f}")
# constant
best = None
g = np.arange(0, 33)/32
for u in g:
    for v in g:
        s = np.maximum(0, 1-np.sqrt((ypark[:,0]-u)**2+(ypark[:,1]-v)**2)/0.35).mean()
        if best is None or s > best[0]: best = (s, u, v)
print(f"  best CONSTANT on this fold: {100*best[0]:.3f} at {best[1:]}")
# does the heat carry any signal? log-lik vs prior
iu = np.clip(np.round(ypark[:,0]*PARK_GRID-0.5).astype(int),0,PARK_GRID-1)
iv = np.clip(np.round(ypark[:,1]*PARK_GRID-0.5).astype(int),0,PARK_GRID-1)
print(f"  mean log p(true cell): model {np.log(HEAT[np.arange(n),iv,iu]+1e-12).mean():.4f}"
      f"  prior {np.log(prior[iv,iu]+1e-12).mean():.4f}")

# ---- plan ---------------------------------------------------------------
print("\nplan:")
park = decode_park_batch(0.3*HEAT + 0.7*prior[None])
for w in [(1,0,0),(0,0,1),(.4,.2,.4),(.5,.25,.25)]:
    M = marg_blend(SEL, POS, PM, w)
    pl = decode_all(M, POS)
    sel_frac = np.mean([len(set(pl[i]) & set(ypos[i]))/4 for i in range(n)])
    order = np.mean(((pl == ypos) * np.array([.4,.3,.2,.1])).sum(1))
    tot, ps, pk = final_score(pl, park, ypos, ypark)
    print(f"  w={w}: sel {sel_frac:.4f} order {order:.4f} plan {ps:.3f} total {tot:.3f}")
# prior-only plan
cnt = np.zeros(NPROBE)
for r in ypos_all[tr_mask]:
    cnt[r] += 1
cnt /= tr_mask.sum()
pl0 = np.array([sorted(np.argsort(-cnt[:NAX])[:2]) + sorted(NAX+np.argsort(-cnt[NAX:])[:2])]*n)
sel_frac = np.mean([len(set(pl0[i]) & set(ypos[i]))/4 for i in range(n)])
print(f"  PRIOR-only: sel {sel_frac:.4f}")
