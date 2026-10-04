#!/usr/bin/env python3
"""Sparse Scout Probe Planning.

Fine-tunes an ImageNet-pretrained ResNet backbone on the sparse scout images and
predicts, for every test row, the ordered four-probe sweep plan and the parking
coordinate.

Usage:
    python3 solution.py <public_dir> <submission_out>

Defaults (when run with no arguments) are ``dataset/public`` and
``working/submission.csv``.

Design
------
* The catalogue splits into 16 ``axis_0`` probes whose fixed coordinate is the
  vertical image coordinate v (a horizontal line sweep) and 16 ``axis_1`` probes
  whose fixed coordinate is u (a vertical line sweep).  This was established
  empirically from the training labels.
* The network therefore predicts two 1-D problems.  A feature-pyramid map is
  pooled along the width to give a per-v profile and along the height to give a
  per-u profile; one shared 1-D convolutional head reads both, which makes the
  two axes symmetric.  Probe logits are sampled from the profile at the exact
  catalogue coordinates by linear interpolation.
* Heads: (a) list-wise "which two of the 16" logits per axis, (b) four ordered
  position soft-maxes over all 32 probes, (c) a soft-max over the 120 unordered
  probe pairs per axis, (d) a 64x64 parking heat-map.
* The label distribution is symmetric under the full dihedral group (flip u,
  flip v, transpose), which is used both as training augmentation and as
  eight-fold test-time augmentation.
* Decoding maximises the expected challenge score exactly.  The plan score is
  linear in the per-probe marginals and in the per-position marginals, so the
  optimum over all valid plans is found by enumerating the six axis patterns and
  the 240 ordered pairs per axis.  The parking point is the lattice point that
  maximises the expected parking score under the predicted heat-map.
"""
import os
import sys
import json
import time
import math
import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
SIZE = 256               # network input resolution
ARCH = "resnet18"
EPOCHS = 70
BATCH = 32
LR = 4e-4
WD = 1e-4
LABEL_SMOOTH = 0.03
LOSS_W = (1.0, 0.6, 1.0, 0.6)     # selection / position / parking / pair
N_SEEDS = 5
TTA = 8
MARG_W = (0.4, 0.2, 0.4)          # blend of the three marginal estimates
TIME_BUDGET = 45 * 60             # seconds of training before we stop adding seeds

NAX = 16
NPROBE = 32
PARK_GRID = 64
PARK_R = 0.35
WPOS = np.array([0.40, 0.30, 0.20, 0.10])
CAND = np.arange(0, 33) / 32.0    # parking labels live on a 1/32 lattice
IU = np.triu_indices(NAX, 1)      # the 120 unordered pairs
DIHEDRAL = [(fu, fv, tp) for tp in (0, 1) for fv in (0, 1) for fu in (0, 1)]
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

PAIR_ID = np.full((NAX, NAX), -1, np.int64)
PAIR_MEMBER = np.zeros((120, NAX), np.float32)
for _t, (_a, _b) in enumerate(zip(*IU)):
    PAIR_ID[_a, _b] = PAIR_ID[_b, _a] = _t
    PAIR_MEMBER[_t, _a] = PAIR_MEMBER[_t, _b] = 1.0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_images(public_dir, rel_paths, size=SIZE, workers=8):
    out = np.zeros((len(rel_paths), size, size, 3), dtype=np.uint8)

    def one(i):
        im = Image.open(os.path.join(public_dir, rel_paths[i].replace("/", os.sep)))
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.uint8)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(len(rel_paths))))
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


def encode_labels(df, code_to_idx):
    pos = np.zeros((len(df), 4), dtype=np.int64)
    for i, s in enumerate(df.probe_order.values):
        for j, c in enumerate(json.loads(s)):
            pos[i, j] = code_to_idx[c]
    park = df[["park_u", "park_v"]].values.astype(np.float32)
    return pos, park


# --------------------------------------------------------------------------- #
# dihedral group
# --------------------------------------------------------------------------- #
def dihedral_probe(p, t):
    """Probe index under an image transform. axis_0 is fixed in v, axis_1 in u."""
    fu, fv, tp = t
    ax, k = p // NAX, p % NAX
    flip = np.where(ax == 0, fv, fu)
    k = np.where(flip == 1, NAX - 1 - k, k)
    if tp:
        ax = 1 - ax
    return ax * NAX + k


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


def build_perms(device):
    base = np.arange(NPROBE)
    P = np.stack([dihedral_probe(base, t) for t in DIHEDRAL])
    return torch.tensor(P, dtype=torch.long, device=device)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def sample_taps(fixed, up, device):
    x = np.clip(np.asarray(fixed, dtype=np.float64) * up - 0.5, 0, up - 1)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, up - 1)
    return (torch.tensor(i0, device=device), torch.tensor(i1, device=device),
            torch.tensor(x - i0, dtype=torch.float32, device=device))


class ProfileHead(nn.Module):
    """Shared by both axes: a 1-D tower over the pooled profile."""

    def __init__(self, cin, cg, width=192, emb=64, nout=5, up=64):
        super().__init__()
        self.up, self.nout, self.emb = up, nout, emb
        self.pre = nn.Sequential(
            nn.Conv1d(cin + cg, width, 5, padding=2), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU())
        self.post = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv1d(width, nout + emb, 1))
        self.pair_bias = nn.Linear(emb, 1)
        self.pair_w = nn.Linear(emb, emb, bias=False)

    def forward(self, prof, gctx, taps):
        B, _, L = prof.shape
        h = torch.cat([prof, gctx[:, :, None].expand(-1, -1, L)], 1)
        h = self.pre(h)
        h = F.interpolate(h, size=self.up, mode="linear", align_corners=False)
        h = self.post(h)
        i0, i1, w = taps
        h = h[:, :, i0] * (1 - w) + h[:, :, i1] * w
        logits = h[:, :self.nout]
        e = h[:, self.nout:].transpose(1, 2)
        s = torch.einsum("bik,bjk->bij", self.pair_w(e), e)
        s = 0.5 * (s + s.transpose(1, 2))
        b = self.pair_bias(e)[..., 0]
        s = s + b[:, :, None] + b[:, None, :]
        return logits, s[:, IU[0], IU[1]]


class ScoutNet(nn.Module):
    def __init__(self, arch=ARCH, pretrained=True, fpn=128, width=192):
        super().__init__()
        import torchvision
        fn = getattr(torchvision.models, arch)
        try:
            net = fn(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:                       # no cached / downloadable weights
            net = fn(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.l1, self.l2, self.l3, self.l4 = net.layer1, net.layer2, net.layer3, net.layer4
        with torch.no_grad():
            d = torch.zeros(1, 3, 128, 128)
            a = self.l1(self.stem(d)); b = self.l2(a); c = self.l3(b); e = self.l4(c)
        ch = [a.shape[1], b.shape[1], c.shape[1], e.shape[1]]
        self.lat = nn.ModuleList([nn.Conv2d(k, fpn, 1) for k in ch])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fpn, fpn, 3, padding=1), nn.GroupNorm(8, fpn), nn.SiLU())
            for _ in range(3)])
        self.gctx = nn.Sequential(nn.Linear(ch[3], 256), nn.SiLU())
        self.profile = ProfileHead(cin=2 * fpn, cg=256, width=width)
        self.park = nn.Sequential(
            nn.Conv2d(fpn, fpn, 3, padding=1), nn.GroupNorm(8, fpn), nn.SiLU(),
            nn.Conv2d(fpn, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 1, 1))

    def forward(self, x, taps):
        c2 = self.l1(self.stem(x))
        c3 = self.l2(c2); c4 = self.l3(c3); c5 = self.l4(c4)
        p5 = self.lat[3](c5)
        p4 = self.smooth[2](self.lat[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.smooth[1](self.lat[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        p2 = self.smooth[0](self.lat[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest"))
        g = self.gctx(F.adaptive_avg_pool2d(c5, 1).flatten(1))

        def prof(t):
            return torch.cat([t.mean(-1), t.amax(-1)], 1)

        lr, pr_r = self.profile(prof(p3), g, taps)                    # axis_0 (rows / v)
        lc, pr_c = self.profile(prof(p3.transpose(-1, -2)), g, taps)  # axis_1 (cols / u)
        logits = torch.cat([lr, lc], -1)
        heat = self.park(p2)
        heat = F.interpolate(heat, size=(PARK_GRID, PARK_GRID),
                             mode="bilinear", align_corners=False)[:, 0]
        return logits[:, 0], logits[:, 1:], heat, torch.stack([pr_r, pr_c], 1)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def prep_batch(xb, mean, std, train):
    """uint8 (B,3,H,W) -> normalised float. Polarity / gain / offset jitter are
    label preserving: the reference decisions depend on transition geometry."""
    x = xb.float() / 255.0
    if train:
        B = x.shape[0]
        mask = x[:, 2:3]
        flip = (torch.rand(B, 1, 1, 1, device=x.device) < 0.5).float()
        rg = x[:, :2]
        rg = flip * (1.0 - rg) + (1 - flip) * rg
        a = 1.0 + 0.35 * (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1)
        b = 0.10 * (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1)
        rg = ((rg - 0.5) * a + 0.5 + b).clamp(0, 1)
        g = rg[:, 1:2] * mask + 0.5 * (1 - mask)     # keep "unmeasured = grey"
        x = torch.cat([rg[:, 0:1], g, x[:, 2:3]], 1)
        x = x + 0.02 * torch.randn_like(x)
    return (x - mean) / std


def park_target(uv, sigma=0.03):
    c = (torch.arange(PARK_GRID, device=uv.device).float() + 0.5) / PARK_GRID
    dv = (c[None, :] - uv[:, 1:2]) ** 2
    du = (c[None, :] - uv[:, 0:1]) ** 2
    t = torch.exp(-0.5 * (dv[:, :, None] + du[:, None, :]) / sigma ** 2)
    return t / t.sum((1, 2), keepdim=True).clamp_min(1e-12)


def compute_loss(sel, pos, heat, pair, ypos, ypark, pair_id, ls=LABEL_SMOOTH, w=LOSS_W):
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
    k, ax = ypos % NAX, ypos // NAX
    loss_pair = 0.0
    for a in range(2):
        kk = k[ax == a].view(B, 2)
        loss_pair = loss_pair + F.cross_entropy(pair[:, a], pair_id[kk[:, 0], kk[:, 1]],
                                                label_smoothing=ls)
    loss_pair = loss_pair / 2
    return w[0] * loss_sel + w[1] * loss_pos + w[2] * loss_park + w[3] * loss_pair


@torch.no_grad()
def predict(model, X, mean, std, taps, perms, device, bs=64, tta=TTA):
    model.eval()
    N = X.shape[0]
    SEL = np.zeros((N, NPROBE)); POS = np.zeros((N, 4, NPROBE))
    HEAT = np.zeros((N, PARK_GRID, PARK_GRID)); PM = np.zeros((N, NPROBE))
    pm_t = torch.tensor(PAIR_MEMBER, device=device)
    amp = device == "cuda"
    for s in range(0, N, bs):
        x0 = prep_batch(X[s:s + bs].to(device), mean, std, False)
        for ti in range(tta):
            t = DIHEDRAL[ti]
            with torch.autocast("cuda", torch.float16, enabled=amp):
                sel, pos, heat, pair = model(dihedral_img(x0, t), taps)
            sel, pos = sel.float(), pos.float()
            heat, pair = heat.float(), pair.float()
            p = perms[ti]
            SEL[s:s + bs] += torch.softmax(sel.view(-1, 2, NAX), -1
                                           ).view(-1, NPROBE)[:, p].double().cpu().numpy()
            POS[s:s + bs] += torch.softmax(pos, -1)[:, :, p].double().cpu().numpy()
            PM[s:s + bs] += (torch.softmax(pair, -1) @ pm_t
                             ).reshape(-1, NPROBE)[:, p].double().cpu().numpy()
            h = torch.softmax(heat.flatten(1), 1).view(-1, PARK_GRID, PARK_GRID)
            HEAT[s:s + bs] += inv_dihedral_map(h, t).double().cpu().numpy()
    for A in (SEL, POS, HEAT, PM):
        A /= tta
    HEAT /= HEAT.sum((1, 2), keepdims=True)
    return SEL, POS, HEAT, PM


def train_one(X, ypos, ypark, fixed16, device, seed, epochs=EPOCHS, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, std = IMNET_MEAN.to(device), IMNET_STD.to(device)
    model = ScoutNet().to(device)
    taps = sample_taps(fixed16, 64, device)
    perms = build_perms(device)
    pair_id = torch.tensor(PAIR_ID, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    N = X.shape[0]
    spe = max(1, N // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=spe * epochs,
                                                pct_start=0.15)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    ypos_t = torch.tensor(ypos, device=device)
    ypark_t = torch.tensor(ypark, device=device)
    step, t0 = 0, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        for b in range(spe):
            ix = perm[b * BATCH:(b + 1) * BATCH]
            xb = X[ix].to(device, non_blocking=True)
            ixd = ix.to(device)
            ti = int(torch.randint(0, 8, (1,)).item())
            t = DIHEDRAL[ti]
            x = dihedral_img(prep_batch(xb, mean, std, True), t)
            yp = perms[ti][ypos_t[ixd]]
            yk = dihedral_park(ypark_t[ixd], t)
            with torch.autocast("cuda", torch.float16, enabled=amp):
                sel, pos, heat, pair = model(x, taps)
                loss = compute_loss(sel.float(), pos.float(), heat.float(), pair.float(),
                                    yp, yk, pair_id)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                d = min(0.997, (1 + step) / (10 + step))
                for kk, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[kk].mul_(d).add_(v.detach().float(), alpha=1 - d)
                    else:
                        ema[kk].copy_(v)
            step += 1
        if (ep + 1) % 20 == 0:
            log(f"    epoch {ep + 1}/{epochs}  loss {float(loss):.4f}  "
                f"({time.time() - t0:.0f}s)")
    sd = model.state_dict()
    model.load_state_dict({k: v.to(sd[k].dtype) for k, v in ema.items()})
    return model, taps, perms


# --------------------------------------------------------------------------- #
# decoding (exact expected-score maximisation)
# --------------------------------------------------------------------------- #
_AXIS_PATTERNS = [p for p in itertools.product((0, 1), repeat=4) if sum(p) == 2]
_OP = np.array([(a, b) for a in range(NAX) for b in range(NAX) if a != b])


def decode_plan(m, q):
    """m: (32,) P(probe in plan); q: (4,32) P(t_j = probe)."""
    best_val, best_plan = -1e18, None
    A, B = _OP[:, 0], _OP[:, 1]
    for pat in _AXIS_PATTERNS:
        pat = np.asarray(pat)
        parts = []
        for a in (0, 1):
            j0, j1 = np.where(pat == a)[0]
            off = a * NAX
            val = (m[off + A] + m[off + B]
                   + WPOS[j0] * q[j0, off + A] + WPOS[j1] * q[j1, off + B])
            k = int(np.argmax(val))
            parts.append((val[k], (j0, off + A[k]), (j1, off + B[k])))
        tot = parts[0][0] + parts[1][0]
        if tot > best_val:
            best_val = tot
            best_plan = [None] * 4
            for _, (ja, pa), (jb, pb) in parts:
                best_plan[ja] = int(pa)
                best_plan[jb] = int(pb)
    return best_plan


def decode_park_batch(heat):
    """heat: (N, Gv, Gu) -> (N,2) lattice point maximising the expected score."""
    N, Gv, Gu = heat.shape
    cv = (np.arange(Gv) + 0.5) / Gv
    cu = (np.arange(Gu) + 0.5) / Gu
    du = (CAND[:, None] - cu[None, :]) ** 2
    dv = (CAND[:, None] - cv[None, :]) ** 2
    nc = len(CAND)
    d2 = dv[None, :, :, None] + du[:, None, None, :]
    s = np.maximum(0.0, 1.0 - np.sqrt(d2) / PARK_R).reshape(nc * nc, Gv * Gu).astype(np.float32)
    exp = heat.reshape(N, Gv * Gu).astype(np.float32) @ s.T
    k = exp.argmax(1)
    return np.stack([CAND[k // nc], CAND[k % nc]], 1)


def marg_blend(SEL, POS, PM, w=MARG_W):
    return w[0] * 2.0 * SEL + w[1] * POS.sum(1) + w[2] * PM


# --------------------------------------------------------------------------- #
# scoring helper (used for the optional held-out report)
# --------------------------------------------------------------------------- #
def score(pred_pos, pred_uv, true_pos, true_uv):
    sel = np.array([len(set(a) & set(b)) / 4.0 for a, b in zip(pred_pos, true_pos)])
    order = ((pred_pos == true_pos) * WPOS[None, :]).sum(1)
    plan = 0.8 * sel + 0.2 * order
    d = np.sqrt(((pred_uv - true_uv) ** 2).sum(-1))
    park = np.maximum(0.0, 1.0 - d / PARK_R)
    return 100 * (0.7 * plan + 0.3 * park).mean(), 100 * plan.mean(), 100 * park.mean()


# --------------------------------------------------------------------------- #
def write_submission(ids, plans, park, codes, path):
    rows = []
    for i, rid in enumerate(ids):
        pl = [codes[j] for j in plans[i]]
        assert len(set(pl)) == 4 and sum(1 for j in plans[i] if j < NAX) == 2
        u = float(min(max(park[i, 0], 0.0), 1.0))
        v = float(min(max(park[i, 1], 0.0), 1.0))
        rows.append((rid, json.dumps(pl), u, v))
    df = pd.DataFrame(rows, columns=["id", "probe_order", "park_u", "park_v"])
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main():
    public = sys.argv[1] if len(sys.argv) > 1 else os.path.join("dataset", "public")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("working", "submission.csv")
    t_start = time.time()
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"public={public} out={out} device={device}", flush=True)

    tr = pd.read_csv(os.path.join(public, "train.csv"))
    te = pd.read_csv(os.path.join(public, "test.csv"))
    cat = pd.read_csv(os.path.join(public, "probe_catalog.csv")).sort_values("probe_code")
    cat = cat.reset_index(drop=True)
    codes = list(cat.probe_code)
    code_to_idx = {c: i for i, c in enumerate(codes)}
    fixed16 = cat.fixed_coordinate.values[:NAX].astype(np.float64)
    assert (cat.sweep_axis.values[:NAX] == "axis_0").all()
    assert (cat.sweep_axis.values[NAX:] == "axis_1").all()
    assert np.allclose(cat.fixed_coordinate.values[NAX:], fixed16)   # shared taps

    # a format-valid fallback so that an interrupted run still produces a file
    n_te = len(te)
    prior = [c for c in ("q04", "q11", "q27", "q20")]
    write_submission(list(te.id),
                     np.array([[code_to_idx[c] for c in prior]] * n_te),
                     np.tile(np.array([[0.49, 0.50]]), (n_te, 1)), codes, out)

    ypos, ypark = encode_labels(tr, code_to_idx)
    Xtr = load_images(public, list(tr.survey_image))
    Xte = load_images(public, list(te.survey_image))
    print(f"train {tuple(Xtr.shape)}  test {tuple(Xte.shape)}  "
          f"({time.time() - t_start:.0f}s)", flush=True)

    mean, std = IMNET_MEAN.to(device), IMNET_STD.to(device)
    SEL = np.zeros((n_te, NPROBE)); POS = np.zeros((n_te, 4, NPROBE))
    HEAT = np.zeros((n_te, PARK_GRID, PARK_GRID)); PM = np.zeros((n_te, NPROBE))
    used = 0
    for s in range(N_SEEDS):
        if s > 0 and time.time() - t_start > TIME_BUDGET:
            print(f"time budget reached, stopping after {used} models", flush=True)
            break
        print(f"[model {s + 1}/{N_SEEDS}] {time.time() - t_start:.0f}s", flush=True)
        model, taps, perms = train_one(Xtr, ypos, ypark, fixed16, device, seed=1000 + s)
        S, P, H, M = predict(model, Xte, mean, std, taps, perms, device)
        SEL += S; POS += P; HEAT += H; PM += M
        used += 1
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    SEL /= used; POS /= used; HEAT /= used; PM /= used
    HEAT /= HEAT.sum((1, 2), keepdims=True)

    M = marg_blend(SEL, POS, PM)
    plans = np.stack([decode_plan(M[i], POS[i]) for i in range(n_te)])
    park = decode_park_batch(HEAT)
    df = write_submission(list(te.id), plans, park, codes, out)
    print(df.head().to_string(), flush=True)
    print(f"wrote {out}: {len(df)} rows, {used} models, "
          f"{time.time() - t_start:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
