"""Training / CV harness v2 (FPN + pair head)."""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (NPROBE, NAX, PARK_GRID, DIHEDRAL, load_meta, catalog_arrays,
                  load_images, encode_labels, apply_dihedral_probe, decode_plan,
                  decode_park_batch, final_score)
from model2 import ScoutNet2, sample_taps, IU

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PAIR_ID = np.full((NAX, NAX), -1, np.int64)
for _t, (_a, _b) in enumerate(zip(*IU)):
    PAIR_ID[_a, _b] = PAIR_ID[_b, _a] = _t
PAIR_MEMBER = np.zeros((120, NAX), np.float32)
for _t, (_a, _b) in enumerate(zip(*IU)):
    PAIR_MEMBER[_t, _a] = PAIR_MEMBER[_t, _b] = 1.0


def build_perms(device):
    base = np.arange(NPROBE)
    P = np.stack([apply_dihedral_probe(base, t) for t in DIHEDRAL])
    return torch.tensor(P, dtype=torch.long, device=device)


def dihedral_img(x, t):
    fu, fv, tp = t
    if fu:
        x = torch.flip(x, dims=[3])
    if fv:
        x = torch.flip(x, dims=[2])
    if tp:
        x = x.transpose(2, 3)
    return x


def inv_dihedral_map(x, t):
    fu, fv, tp = t
    if tp:
        x = x.transpose(-2, -1)
    if fv:
        x = torch.flip(x, dims=[-2])
    if fu:
        x = torch.flip(x, dims=[-1])
    return x


def dihedral_park(uv, t):
    fu, fv, tp = t
    u, v = uv[:, 0], uv[:, 1]
    if fu:
        u = 1.0 - u
    if fv:
        v = 1.0 - v
    if tp:
        u, v = v, u
    return torch.stack([u, v], 1)


def prep_batch(xb, mean, std, train=True, aug=True):
    x = xb.float() / 255.0
    if train and aug:
        B = x.shape[0]
        mask = x[:, 2:3]
        flip = (torch.rand(B, 1, 1, 1, device=x.device) < 0.5).float()
        rg = x[:, :2]
        rg = flip * (1.0 - rg) + (1 - flip) * rg
        a = 1.0 + 0.35 * (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1)
        b = 0.10 * (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1)
        rg = ((rg - 0.5) * a + 0.5 + b).clamp(0, 1)
        g = rg[:, 1:2] * mask + 0.5 * (1 - mask)
        x = torch.cat([rg[:, 0:1], g, x[:, 2:3]], 1)
        x = x + 0.02 * torch.randn_like(x)
    return (x - mean) / std


def park_target(uv, G=PARK_GRID, sigma=0.03):
    dev = uv.device
    c = (torch.arange(G, device=dev).float() + 0.5) / G
    dv = (c[None, :] - uv[:, 1:2]) ** 2
    du = (c[None, :] - uv[:, 0:1]) ** 2
    t = torch.exp(-0.5 * (dv[:, :, None] + du[:, None, :]) / sigma ** 2)
    return t / t.sum((1, 2), keepdim=True).clamp_min(1e-12)


def compute_loss(sel, pos, heat, pair, ypos, ypark, pair_id, ls=0.03, w=(1.0, .6, 1.0, .6)):
    B = sel.shape[0]
    tsel = torch.zeros(B, NPROBE, device=sel.device)
    tsel.scatter_(1, ypos, 0.5)
    loss_sel = 0.0
    for a in range(2):
        tt = tsel[:, a * NAX:(a + 1) * NAX] * (1 - ls) + ls / NAX
        loss_sel = loss_sel - (tt * F.log_softmax(sel[:, a * NAX:(a + 1) * NAX], 1)).sum(1).mean()
    loss_sel = loss_sel / 2
    loss_pos = sum(F.cross_entropy(pos[:, j], ypos[:, j], label_smoothing=ls) for j in range(4)) / 4
    tgt = park_target(ypark)
    loss_park = -(tgt.flatten(1) * F.log_softmax(heat.flatten(1), 1)).sum(1).mean()
    # pair target
    k = ypos % NAX
    ax = ypos // NAX
    loss_pair = 0.0
    for a in range(2):
        m = (ax == a)
        kk = k[m].view(B, 2)
        tid = pair_id[kk[:, 0], kk[:, 1]]
        loss_pair = loss_pair + F.cross_entropy(pair[:, a], tid, label_smoothing=ls)
    loss_pair = loss_pair / 2
    tot = w[0] * loss_sel + w[1] * loss_pos + w[2] * loss_park + w[3] * loss_pair
    return tot, (loss_sel.item(), loss_pos.item(), loss_park.item(), loss_pair.item())


@torch.no_grad()
def predict(model, X, mean, std, taps, perms, device, bs=64, tta=8, amp=True):
    model.eval()
    N = X.shape[0]
    SEL = np.zeros((N, NPROBE)); POS = np.zeros((N, 4, NPROBE))
    HEAT = np.zeros((N, PARK_GRID, PARK_GRID)); PM = np.zeros((N, NPROBE))
    pm_t = torch.tensor(PAIR_MEMBER, device=device)
    for s in range(0, N, bs):
        x0 = prep_batch(X[s:s + bs].to(device), mean, std, train=False)
        for ti in range(tta):
            t = DIHEDRAL[ti]
            with torch.autocast("cuda", torch.float16, enabled=amp):
                sel, pos, heat, pair = model(dihedral_img(x0, t), taps)
            sel = sel.float(); pos = pos.float(); heat = heat.float(); pair = pair.float()
            p = perms[ti]
            sp = torch.softmax(sel.view(-1, 2, NAX), -1).view(-1, NPROBE)
            SEL[s:s + bs] += sp[:, p].double().cpu().numpy()
            POS[s:s + bs] += torch.softmax(pos, -1)[:, :, p].double().cpu().numpy()
            marg = torch.softmax(pair, -1) @ pm_t                    # (B,2,16)
            PM[s:s + bs] += marg.reshape(-1, NPROBE)[:, p].double().cpu().numpy()
            h = torch.softmax(heat.flatten(1), 1).view(-1, PARK_GRID, PARK_GRID)
            HEAT[s:s + bs] += inv_dihedral_map(h, t).double().cpu().numpy()
    for A in (SEL, POS, HEAT, PM):
        A /= tta
    HEAT /= HEAT.sum((1, 2), keepdims=True)
    return SEL, POS, HEAT, PM


def marg_blend(SEL, POS, PM, wts=(0.4, 0.2, 0.4)):
    return wts[0] * 2.0 * SEL + wts[1] * POS.sum(1) + wts[2] * PM


def decode_all(M, POS):
    return np.stack([decode_plan(M[i], POS[i]) for i in range(len(M))])


def train_one(Xtr, ytr_pos, ytr_park, Xva, cfg, device, seed=0,
              log=lambda *a, **k: print(*a, **{**k, "flush": True}), Xte=None):
    torch.manual_seed(seed); np.random.seed(seed)
    mean = IMNET_MEAN.to(device); std = IMNET_STD.to(device)
    model = ScoutNet2(cfg["arch"], pretrained=cfg["pretrained"], width=cfg.get("width", 192),
                      prof_stride=cfg.get("prof_stride", 8)).to(device)
    taps = sample_taps(cfg["fixed16"], 64, device)
    perms = build_perms(device)
    pair_id = torch.tensor(PAIR_ID, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    N = Xtr.shape[0]
    spe = max(1, N // cfg["bs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                                total_steps=spe * cfg["epochs"], pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    ed = cfg.get("ema", 0.997)
    ypos_t = torch.tensor(ytr_pos, device=device)
    ypark_t = torch.tensor(ytr_park, device=device)
    step = 0; t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(N)
        agg = np.zeros(4); nb = 0
        for b in range(spe):
            ix = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
            xb = Xtr[ix].to(device, non_blocking=True)
            ixd = ix.to(device)
            ti = int(torch.randint(0, 8, (1,)).item())
            t = DIHEDRAL[ti]
            x = dihedral_img(prep_batch(xb, mean, std, True, cfg.get("aug", True)), t)
            yp2 = perms[ti][ypos_t[ixd]]
            yk2 = dihedral_park(ypark_t[ixd], t)
            with torch.autocast("cuda", torch.float16, enabled=cfg["amp"]):
                sel, pos, heat, pair = model(x, taps)
                loss, parts = compute_loss(sel.float(), pos.float(), heat.float(), pair.float(),
                                           yp2, yk2, pair_id, ls=cfg["ls"], w=cfg["lossw"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                d = min(ed, (1 + step) / (10 + step))
                for kk, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[kk].mul_(d).add_(v.detach().float(), alpha=1 - d)
                    else:
                        ema[kk].copy_(v)
            step += 1; agg += parts; nb += 1
        if (ep + 1) % max(1, cfg["epochs"] // 20) == 0:
            log(f"    ep {ep+1:3d} sel={agg[0]/nb:.4f} pos={agg[1]/nb:.4f} "
                f"park={agg[2]/nb:.4f} pair={agg[3]/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
    sd = model.state_dict()
    model.load_state_dict({k: v.to(sd[k].dtype) for k, v in ema.items()})
    outs = [predict(model, Xva, mean, std, taps, perms, device, tta=cfg["tta"], amp=cfg["amp"])]
    if Xte is not None:
        outs.append(predict(model, Xte, mean, std, taps, perms, device, tta=cfg["tta"], amp=cfg["amp"]))
    return model, outs


def park_prior(ypark_train, sigma=2.0):
    """Empirical parking heat-map built from training rows only."""
    from scipy.ndimage import gaussian_filter
    c = (np.arange(PARK_GRID) + 0.5) / PARK_GRID
    h = np.zeros((PARK_GRID, PARK_GRID))
    for u, v in ypark_train:
        h[np.argmin(np.abs(c - v)), np.argmin(np.abs(c - u))] += 1
    h = gaussian_filter(h, sigma)
    return h / h.sum()


def report(SEL, POS, HEAT, PM, ypos, ypark, prefix="", prior_heat=None):
    if prior_heat is not None:
        print(f"  {prefix}park lambda sweep:")
        bestl = None
        for lam in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            H = lam * HEAT + (1 - lam) * prior_heat[None]
            H = H / H.sum((1, 2), keepdims=True)
            p = decode_park_batch(H)
            d = np.sqrt(((p - ypark) ** 2).sum(1))
            sc = 100 * np.maximum(0, 1 - d / 0.35).mean()
            print(f"    lam={lam:.2f} park {sc:.3f}")
            if bestl is None or sc > bestl[0]:
                bestl = (sc, lam)
        print(f"  {prefix}best lambda {bestl[1]} -> park {bestl[0]:.3f}")
        HEAT = bestl[1] * HEAT + (1 - bestl[1]) * prior_heat[None]
        HEAT = HEAT / HEAT.sum((1, 2), keepdims=True)
    park = decode_park_batch(HEAT)
    best = None
    for wts in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (.5, .0, .5), (.4, .2, .4), (.34, .33, .33),
                (.6, .2, .2), (.2, .2, .6)]:
        M = marg_blend(SEL, POS, PM, wts)
        pl = decode_all(M, POS)
        tot, ps, pk = final_score(pl, park, ypos, ypark)
        if best is None or tot > best[0]:
            best = (tot, ps, pk, wts)
        print(f"  {prefix}wts={wts} total {tot:.3f} plan {ps:.3f} park {pk:.3f}")
    print(f"  {prefix}BEST {best[3]} -> total {best[0]:.3f} (plan {best[1]:.3f} park {best[2]:.3f})")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=r"c:\Users\tarun\OneDrive\Desktop\ERIS\sparse_scout")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ls", type=float, default=0.03)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-folds", type=int, default=0)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--prof-stride", type=int, default=8)
    ap.add_argument("--lossw", default="1.0,0.6,1.0,0.6")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = "cuda"
    tr, te, cat = load_meta(args.public)
    codes, axis, fixed, idx = catalog_arrays(cat)
    ypos, ypark = encode_labels(tr, idx)
    Ximg = load_images(args.public, list(tr.survey_image), size=args.size)
    X = torch.from_numpy(Ximg).permute(0, 3, 1, 2).contiguous()

    cfg = dict(arch=args.arch, epochs=args.epochs, bs=args.bs, lr=args.lr, wd=args.wd,
               ls=args.ls, amp=True, tta=args.tta, fixed16=fixed[:NAX], width=args.width,
               prof_stride=args.prof_stride, aug=not args.no_aug,
               pretrained=not args.no_pretrained,
               lossw=tuple(float(v) for v in args.lossw.split(",")))
    print(json.dumps({k: (list(v) if isinstance(v, np.ndarray) else v)
                      for k, v in cfg.items() if k != "fixed16"}), flush=True)

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=args.folds)
    groups = tr.session_id.values
    n = len(tr)
    SEL = np.zeros((n, NPROBE)); POS = np.zeros((n, 4, NPROBE))
    HEAT = np.zeros((n, PARK_GRID, PARK_GRID)); PM = np.zeros((n, NPROBE))
    used = np.zeros(n, bool)
    nf = args.only_folds or args.folds
    for f, (itr, iva) in enumerate(gkf.split(np.arange(n), groups=groups)):
        if f >= nf:
            break
        print(f"  fold {f}: {len(itr)}/{len(iva)}", flush=True)
        _, outs = train_one(X[itr], ypos[itr], ypark[itr], X[iva], cfg, device,
                            seed=f + 100 * args.seed)
        S, P, H, M = outs[0]
        SEL[iva] = S; POS[iva] = P; HEAT[iva] = H; PM[iva] = M; used[iva] = True
    m = used
    print(f"\n== OOF {m.sum()} rows ==", flush=True)
    report(SEL[m], POS[m], HEAT[m], PM[m], ypos[m], ypark[m],
           prior_heat=park_prior(ypark[~m] if (~m).sum() > 100 else ypark))
    outd = os.path.dirname(os.path.abspath(__file__))
    np.savez_compressed(os.path.join(outd, f"oof_{args.tag}.npz"),
                        sel=SEL, pos=POS, heat=HEAT, pm=PM, used=used,
                        ypos=ypos, ypark=ypark)


if __name__ == "__main__":
    main()
