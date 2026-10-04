"""Training / CV harness for Sparse Scout Probe Planning."""
import os, sys, json, time, math, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (IMG, NPROBE, NAX, WPOS, PARK_GRID, CAND, DIHEDRAL,
                  load_meta, catalog_arrays, load_images, encode_labels,
                  apply_dihedral_probe, decode_plan, decode_park_batch,
                  final_score, ScoutNet, sample_taps)

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- augmentation
def build_perms(device):
    """perm[t] : (32,) transformed probe index for each original index."""
    base = np.arange(NPROBE)
    P = np.stack([apply_dihedral_probe(base, t) for t in DIHEDRAL])
    return torch.tensor(P, dtype=torch.long, device=device)


def dihedral_img(x, t):
    """x (B,C,H,W) with H=v, W=u."""
    fu, fv, tp = t
    if fu:
        x = torch.flip(x, dims=[3])
    if fv:
        x = torch.flip(x, dims=[2])
    if tp:
        x = x.transpose(2, 3)
    return x


def inv_dihedral_map(x, t):
    """Undo dihedral_img on a spatial map (B,*,H,W)."""
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


def prep_batch(xb, mean, std, train=True, gen=None):
    """xb uint8 (B,3,H,W) -> normalised float, with intensity augmentation."""
    x = xb.float() / 255.0
    if train:
        B = x.shape[0]
        mask = x[:, 2:3]
        # polarity flip (label-invariant: transition strength & flatness are)
        flip = (torch.rand(B, 1, 1, 1, device=x.device, generator=gen) < 0.5).float()
        rg = x[:, :2]
        rg = flip * (1.0 - rg) + (1 - flip) * rg
        # gain / offset
        a = 1.0 + 0.35 * (torch.rand(B, 1, 1, 1, device=x.device, generator=gen) * 2 - 1)
        b = 0.10 * (torch.rand(B, 1, 1, 1, device=x.device, generator=gen) * 2 - 1)
        rg = (rg - 0.5) * a + 0.5 + b
        rg = rg.clamp(0, 1)
        # keep the "unmeasured = neutral grey" convention intact on the G channel
        g = rg[:, 1:2] * mask + 0.5 * (1 - mask)
        x = torch.cat([rg[:, 0:1], g, x[:, 2:3]], 1)
        x = x + 0.02 * torch.randn(x.shape, device=x.device, generator=gen)
    return (x - mean) / std


# ---------------------------------------------------------------- targets
def park_target(uv, G=PARK_GRID, sigma=0.03):
    """uv (B,2) -> (B,G,G) normalised gaussian, image layout (v,u)."""
    dev = uv.device
    c = (torch.arange(G, device=dev).float() + 0.5) / G
    dv = (c[None, :] - uv[:, 1:2]) ** 2
    du = (c[None, :] - uv[:, 0:1]) ** 2
    d2 = dv[:, :, None] + du[:, None, :]
    t = torch.exp(-0.5 * d2 / sigma ** 2)
    return t / t.sum((1, 2), keepdim=True).clamp_min(1e-12)


def compute_loss(sel, pos, heat, ypos, ypark, ls=0.03, w=(1.0, 0.6, 1.0)):
    B = sel.shape[0]
    # ---- selection: list-wise CE within each axis (target 0.5 on each chosen)
    tsel = torch.zeros(B, NPROBE, device=sel.device)
    tsel.scatter_(1, ypos, 0.5)
    loss_sel = 0.0
    for a in range(2):
        sl = sel[:, a * NAX:(a + 1) * NAX]
        tt = tsel[:, a * NAX:(a + 1) * NAX]
        tt = tt * (1 - ls) + ls / NAX
        loss_sel = loss_sel + -(tt * F.log_softmax(sl, 1)).sum(1).mean()
    loss_sel = loss_sel / 2
    # ---- ordered positions
    loss_pos = 0.0
    for j in range(4):
        loss_pos = loss_pos + F.cross_entropy(pos[:, j], ypos[:, j], label_smoothing=ls)
    loss_pos = loss_pos / 4
    # ---- parking heat-map
    tgt = park_target(ypark)
    lh = F.log_softmax(heat.flatten(1), 1)
    loss_park = -(tgt.flatten(1) * lh).sum(1).mean()
    return w[0] * loss_sel + w[1] * loss_pos + w[2] * loss_park, \
        (loss_sel.item(), loss_pos.item(), loss_park.item())


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, X, mean, std, taps, perms, device, bs=64, tta=8, amp=True):
    model.eval()
    N = X.shape[0]
    SEL = np.zeros((N, NPROBE), dtype=np.float64)
    POS = np.zeros((N, 4, NPROBE), dtype=np.float64)
    HEAT = np.zeros((N, PARK_GRID, PARK_GRID), dtype=np.float64)
    tl = DIHEDRAL[:tta]
    for s in range(0, N, bs):
        xb = X[s:s + bs].to(device, non_blocking=True)
        x0 = prep_batch(xb, mean, std, train=False)
        for ti, t in enumerate(tl):
            xt = dihedral_img(x0, t)
            with torch.autocast("cuda", torch.float16, enabled=amp):
                sel, pos, heat = model(xt, taps)
            sel = sel.float(); pos = pos.float(); heat = heat.float()
            p = perms[ti]                     # original index -> transformed index
            sp = torch.softmax(sel.view(-1, 2, NAX), -1).view(-1, NPROBE)
            SEL[s:s + bs] += sp[:, p].double().cpu().numpy()
            POS[s:s + bs] += torch.softmax(pos, -1)[:, :, p].double().cpu().numpy()
            h = torch.softmax(heat.flatten(1), 1).view(-1, PARK_GRID, PARK_GRID)
            HEAT[s:s + bs] += inv_dihedral_map(h, t).double().cpu().numpy()
    SEL /= len(tl); POS /= len(tl); HEAT /= len(tl)
    HEAT /= HEAT.sum((1, 2), keepdims=True)
    return SEL, POS, HEAT


def decode_all(SEL, POS, alpha=0.5):
    """SEL: (N,32) within-axis softmax; POS: (N,4,32). alpha blends the two
    estimates of the membership marginal."""
    N = SEL.shape[0]
    M = alpha * (2.0 * SEL) + (1 - alpha) * POS.sum(1)
    plans = np.zeros((N, 4), dtype=np.int64)
    for i in range(N):
        plans[i] = decode_plan(M[i], POS[i])
    return plans


# ---------------------------------------------------------------- training
def train_one(Xtr, ytr_pos, ytr_park, Xva, cfg, device, seed=0, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    mean = IMNET_MEAN.to(device); std = IMNET_STD.to(device)
    model = ScoutNet(cfg["arch"], in_ch=3, pretrained=cfg["pretrained"]).to(device)
    if cfg.get("channels_last"):
        model = model.to(memory_format=torch.channels_last)
    taps = sample_taps(cfg["fixed16"], 64, device)
    perms = build_perms(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    N = Xtr.shape[0]
    steps_per_epoch = max(1, N // cfg["bs"])
    total = steps_per_epoch * cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total,
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    ed = cfg.get("ema", 0.995)
    ypos_t = torch.tensor(ytr_pos, device=device)
    ypark_t = torch.tensor(ytr_park, device=device)
    step = 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(N)
        agg = np.zeros(3); nb = 0
        for b in range(steps_per_epoch):
            ix = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
            xb = Xtr[ix].to(device, non_blocking=True)
            ix = ix.to(device)
            yp = ypos_t[ix]; yk = ypark_t[ix]
            ti = int(torch.randint(0, 8, (1,)).item())
            t = DIHEDRAL[ti]
            x = prep_batch(xb, mean, std, train=True)
            x = dihedral_img(x, t)
            if cfg.get("channels_last"):
                x = x.contiguous(memory_format=torch.channels_last)
            yp2 = perms[ti][yp]
            yk2 = dihedral_park(yk, t)
            with torch.autocast("cuda", torch.float16, enabled=cfg["amp"]):
                sel, pos, heat = model(x, taps)
                loss, parts = compute_loss(sel.float(), pos.float(), heat.float(),
                                           yp2, yk2, ls=cfg["ls"], w=cfg["lossw"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                d = min(ed, (1 + step) / (10 + step))
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
                    else:
                        ema[k].copy_(v)
            step += 1
            agg += parts; nb += 1
        if (ep + 1) % max(1, cfg["epochs"] // 6) == 0:
            log(f"    ep {ep+1:3d}/{cfg['epochs']} loss sel={agg[0]/nb:.4f} "
                f"pos={agg[1]/nb:.4f} park={agg[2]/nb:.4f}  ({time.time()-t0:.0f}s)")
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ema.items()})
    out = predict(model, Xva, mean, std, taps, perms, device, tta=cfg["tta"], amp=cfg["amp"])
    return model, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=r"c:\Users\tarun\OneDrive\Desktop\ERIS\sparse_scout")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ls", type=float, default=0.03)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-folds", type=int, default=0)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--lossw", default="1.0,0.6,1.0")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = "cuda"
    tr, te, cat = load_meta(args.public)
    codes, axis, fixed, idx = catalog_arrays(cat)
    fixed16 = fixed[:NAX]
    ypos, ypark = encode_labels(tr, idx)
    print("loading images...")
    t0 = time.time()
    Ximg = load_images(args.public, list(tr.survey_image), size=args.size)
    X = torch.from_numpy(Ximg).permute(0, 3, 1, 2).contiguous()
    print(f"  {X.shape} in {time.time()-t0:.0f}s")

    cfg = dict(arch=args.arch, epochs=args.epochs, bs=args.bs, lr=args.lr, wd=args.wd,
               ls=args.ls, amp=True, tta=args.tta, fixed16=fixed16,
               pretrained=not args.no_pretrained,
               lossw=tuple(float(v) for v in args.lossw.split(",")))
    print(json.dumps({k: v for k, v in cfg.items() if k != "fixed16"}))

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=args.folds)
    groups = tr.session_id.values
    oof_sel = np.zeros((len(tr), NPROBE))
    oof_pos = np.zeros((len(tr), 4, NPROBE))
    oof_heat = np.zeros((len(tr), PARK_GRID, PARK_GRID))
    nf = args.only_folds or args.folds
    used = np.zeros(len(tr), bool)
    for f, (itr, iva) in enumerate(gkf.split(np.arange(len(tr)), groups=groups)):
        if f >= nf:
            break
        print(f"  fold {f}: train {len(itr)} val {len(iva)}")
        _, (S, P, H) = train_one(X[itr], ypos[itr], ypark[itr], X[iva], cfg, device,
                                 seed=f + 100 * args.seed)
        oof_sel[iva] = S; oof_pos[iva] = P; oof_heat[iva] = H; used[iva] = True

    m = used
    plans = decode_all(oof_sel[m], oof_pos[m], alpha=0.5)
    park = decode_park_batch(oof_heat[m])
    tot, ps, pk = final_score(plans, park, ypos[m], ypark[m])
    print(f"\n== OOF ({m.sum()} rows) total {tot:.3f} | plan {ps:.3f} | park {pk:.3f}")
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        pl = decode_all(oof_sel[m], oof_pos[m], alpha=a)
        t2, p2, _ = final_score(pl, park, ypos[m], ypark[m])
        print(f"   alpha={a}: total {t2:.3f} plan {p2:.3f}")
    outd = os.path.dirname(os.path.abspath(__file__))
    np.savez_compressed(os.path.join(outd, f"oof_{args.tag}.npz"),
                        sel=oof_sel, pos=oof_pos, heat=oof_heat, used=used,
                        ypos=ypos, ypark=ypark)


if __name__ == "__main__":
    main()
