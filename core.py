"""Core library for the Sparse Scout Probe Planning challenge.

Everything needed for training / decoding / scoring lives here so that the
notebook and solution.py stay thin wrappers.
"""
import os, json, math, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
IMG = 256
NPROBE = 32
NAX = 16
WPOS = np.array([0.40, 0.30, 0.20, 0.10], dtype=np.float64)
PARK_R = 0.35
PARK_GRID = 64          # heat-map resolution used by the model
CAND = np.arange(0, 33) / 32.0   # park labels live on a 1/32 lattice


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_meta(public_dir):
    tr = pd.read_csv(os.path.join(public_dir, "train.csv"))
    te = pd.read_csv(os.path.join(public_dir, "test.csv"))
    cat = pd.read_csv(os.path.join(public_dir, "probe_catalog.csv"))
    cat = cat.sort_values("probe_code").reset_index(drop=True)
    return tr, te, cat


def catalog_arrays(cat):
    codes = list(cat.probe_code)
    axis = (cat.sweep_axis.values == "axis_1").astype(np.int64)
    fixed = cat.fixed_coordinate.values.astype(np.float64)
    idx = {c: i for i, c in enumerate(codes)}
    return codes, axis, fixed, idx


def load_images(public_dir, rel_paths, size=IMG, workers=8):
    """Return uint8 array (N, size, size, 3)."""
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    out = np.zeros((len(rel_paths), size, size, 3), dtype=np.uint8)

    def one(i):
        p = os.path.join(public_dir, rel_paths[i].replace("/", os.sep))
        im = Image.open(p).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.uint8)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(len(rel_paths))))
    return out


# ----------------------------------------------------------------------------
# label encoding
# ----------------------------------------------------------------------------
def encode_labels(df, idx):
    """positions (N,4) int64 global probe index; park (N,2) float."""
    pos = np.zeros((len(df), 4), dtype=np.int64)
    for i, s in enumerate(df.probe_order.values):
        for j, c in enumerate(json.loads(s)):
            pos[i, j] = idx[c]
    park = df[["park_u", "park_v"]].values.astype(np.float32)
    return pos, park


# ----------------------------------------------------------------------------
# dihedral augmentation (image + label transform)
# ----------------------------------------------------------------------------
# a transform is (flip_u, flip_v, transpose)
DIHEDRAL = [(fu, fv, tp) for tp in (0, 1) for fv in (0, 1) for fu in (0, 1)]


def apply_dihedral_img(x, t):
    """x: (...,H,W,C) numpy or (...,C,H,W) torch? -> here numpy HWC."""
    fu, fv, tp = t
    if fu:
        x = x[:, ::-1]
    if fv:
        x = x[::-1, :]
    if tp:
        x = np.swapaxes(x, 0, 1)
    return np.ascontiguousarray(x)


def apply_dihedral_probe(p, t):
    """p: global probe index array. Returns transformed index."""
    fu, fv, tp = t
    ax = p // NAX          # 0 = axis_0 (horizontal line, fixed v) 1 = axis_1 (vertical, fixed u)
    k = p % NAX
    # axis_0 fixed coordinate is v -> flipped by fv ; axis_1 fixed is u -> flipped by fu
    flip = np.where(ax == 0, fv, fu)
    k = np.where(flip == 1, NAX - 1 - k, k)
    if tp:
        ax = 1 - ax
    return ax * NAX + k


def apply_dihedral_park(uv, t):
    fu, fv, tp = t
    u, v = uv[..., 0].copy(), uv[..., 1].copy()
    if fu:
        u = 1.0 - u
    if fv:
        v = 1.0 - v
    if tp:
        u, v = v, u
    return np.stack([u, v], -1)


# ----------------------------------------------------------------------------
# scoring (exact reproduction of the challenge metric)
# ----------------------------------------------------------------------------
def plan_score(pred_pos, true_pos):
    """pred_pos/true_pos: (N,4) int arrays of probe indices."""
    n = len(pred_pos)
    sel = np.zeros(n)
    for i in range(n):
        sel[i] = len(set(pred_pos[i]) & set(true_pos[i])) / 4.0
    order = ((pred_pos == true_pos) * WPOS[None, :]).sum(1)
    return 0.80 * sel + 0.20 * order


def parking_score(pred_uv, true_uv):
    d = np.sqrt(((pred_uv - true_uv) ** 2).sum(-1))
    return np.maximum(0.0, 1.0 - d / PARK_R)


def final_score(pred_pos, pred_uv, true_pos, true_uv):
    ps = plan_score(pred_pos, true_pos)
    pk = parking_score(pred_uv, true_uv)
    row = 0.70 * ps + 0.30 * pk
    return 100.0 * row.mean(), 100 * ps.mean(), 100 * pk.mean()


# ----------------------------------------------------------------------------
# exact plan decoder
# ----------------------------------------------------------------------------
_AXIS_PATTERNS = [p for p in itertools.product((0, 1), repeat=4) if sum(p) == 2]
_ORDERED_PAIRS = np.array([(a, b) for a in range(NAX) for b in range(NAX) if a != b])  # (240,2)


def decode_plan(m, q):
    """m: (32,) marginal P(probe in plan); q: (4,32) P(t_j = probe).

    Maximises  0.2*sum_{p in S} m_p + 0.2*sum_j w_j q_j(p_j)  exactly.
    Returns list of 4 global probe indices.
    """
    best_val, best_plan = -1e18, None
    A, B = _ORDERED_PAIRS[:, 0], _ORDERED_PAIRS[:, 1]
    for pat in _AXIS_PATTERNS:
        pat = np.asarray(pat)
        pos_ax = [np.where(pat == a)[0] for a in (0, 1)]
        parts = []
        for a in (0, 1):
            j0, j1 = pos_ax[a]
            off = a * NAX
            val = (m[off + A] + m[off + B]
                   + WPOS[j0] * q[j0, off + A] + WPOS[j1] * q[j1, off + B])
            k = int(np.argmax(val))
            parts.append((val[k], (j0, off + A[k]), (j1, off + B[k])))
        tot = 0.2 * (parts[0][0] + parts[1][0])
        if tot > best_val:
            best_val = tot
            plan = [None] * 4
            for _, (ja, pa), (jb, pb) in parts:
                plan[ja] = int(pa)
                plan[jb] = int(pb)
            best_plan = plan
    return best_plan


def decode_park_batch(heat, cand=CAND):
    """MBR decode of the parking point.

    heat: (N, Gv, Gu) probabilities over grid-cell centres ((i+.5)/G), image
    convention (first spatial axis = v, second = u).
    Returns (N,2) array of (u, v) taken from the 1/32 candidate lattice, each
    maximising the expected parking score under the predicted distribution.
    """
    N, Gv, Gu = heat.shape
    cv = (np.arange(Gv) + 0.5) / Gv
    cu = (np.arange(Gu) + 0.5) / Gu
    du = (cand[:, None] - cu[None, :]) ** 2             # (nc, Gu)
    dv = (cand[:, None] - cv[None, :]) ** 2             # (nc, Gv)
    nc = len(cand)
    d2 = dv[None, :, :, None] + du[:, None, None, :]    # (nc_u, nc_v, Gv, Gu)
    s = np.maximum(0.0, 1.0 - np.sqrt(d2) / PARK_R).reshape(nc * nc, Gv * Gu)
    exp = heat.reshape(N, Gv * Gu).astype(np.float32) @ s.astype(np.float32).T
    k = exp.argmax(1)
    return np.stack([cand[k // nc], cand[k % nc]], 1)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class ProbeHead(nn.Module):
    """Shared 1-D head applied to both the row profile and the column profile.

    Sharing weights makes the two axes symmetric, which matches the (verified)
    symmetry of the label distribution and pairs with transpose augmentation.
    """

    def __init__(self, cin, cg, width=256, nout=5, up=64):
        super().__init__()
        self.up = up
        self.net = nn.Sequential(
            nn.Conv1d(cin + cg, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(width, width // 2, 3, padding=1), nn.GroupNorm(8, width // 2), nn.SiLU(),
            nn.Conv1d(width // 2, nout, 1),
        )

    def forward(self, prof, gctx, samp):
        """prof (B,cin,L); gctx (B,cg); samp = (idx0, idx1, w) precomputed
        linear-interpolation taps for the 16 probe coordinates."""
        B, _, L = prof.shape
        h = torch.cat([prof, gctx[:, :, None].expand(-1, -1, L)], 1)
        h = self.net(h)
        h = F.interpolate(h, size=self.up, mode="linear", align_corners=False)
        h = self.net2(h)                                  # (B,nout,up)
        i0, i1, w = samp
        return h[:, :, i0] * (1 - w) + h[:, :, i1] * w    # (B,nout,16)


def sample_taps(fixed, up, device):
    """Linear-interpolation taps for sampling a length-`up` profile at the
    normalised probe coordinates."""
    x = np.clip(np.asarray(fixed, dtype=np.float64) * up - 0.5, 0, up - 1)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, up - 1)
    w = torch.tensor(x - i0, dtype=torch.float32, device=device)
    return (torch.tensor(i0, device=device), torch.tensor(i1, device=device), w)


class ScoutNet(nn.Module):
    def __init__(self, arch="resnet18", in_ch=3, pretrained=True, nheat=PARK_GRID):
        super().__init__()
        import torchvision
        fn = getattr(torchvision.models, arch)
        try:
            net = fn(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:
            net = fn(weights=None)
        if in_ch != 3:
            w = net.conv1.weight.data
            c = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
            c.weight.data.zero_()
            c.weight.data[:, :3] = w
            net.conv1 = c
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.l1, self.l2, self.l3, self.l4 = net.layer1, net.layer2, net.layer3, net.layer4
        c3 = self.l3[-1].conv2.out_channels if hasattr(self.l3[-1], "conv2") else 256
        c4 = self.l4[-1].conv2.out_channels if hasattr(self.l4[-1], "conv2") else 512
        if hasattr(self.l4[-1], "conv3"):
            c3 = self.l3[-1].conv3.out_channels
            c4 = self.l4[-1].conv3.out_channels
        self.c3, self.c4 = c3, c4
        cf = c3 + c4
        self.lat = nn.Sequential(nn.Conv2d(cf, 256, 1), nn.GroupNorm(8, 256), nn.SiLU())
        self.probe = ProbeHead(cin=512, cg=256, width=256, nout=5)
        self.park = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 1, 1),
        )
        self.nheat = nheat

    def forward(self, x, sample_pos):
        h = self.stem(x)
        h = self.l1(h); h = self.l2(h)
        f3 = self.l3(h)                        # stride 16
        f4 = self.l4(f3)                       # stride 32
        f4u = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        f = self.lat(torch.cat([f3, f4u], 1))  # (B,256,16,16)
        g = F.adaptive_avg_pool2d(f, 1).flatten(1)          # (B,256)

        def prof(t):        # t: (B,C,H,W) -> (B,2C,H) pooled over last dim
            return torch.cat([t.mean(-1), t.amax(-1)], 1)

        row = self.probe(prof(f), g, sample_pos)            # (B,5,16) axis_0
        col = self.probe(prof(f.transpose(-1, -2)), g, sample_pos)  # (B,5,16) axis_1
        pl = torch.cat([row, col], -1)                      # (B,5,32)
        heat = self.park(f)
        heat = F.interpolate(heat, size=(self.nheat, self.nheat),
                             mode="bilinear", align_corners=False)[:, 0]
        return pl[:, 0], pl[:, 1:], heat                    # sel(B,32), pos(B,4,32), heat(B,G,G)
