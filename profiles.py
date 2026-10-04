"""Hand-crafted 1-D line-profile features (CPU) — reference model / ensemble member."""
import os, sys, json
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d

SCALES = (1.0, 2.0, 4.0, 8.0)


def row_bank(R, mask):
    """Return dict name -> (256,) per-row (v) profile and per-col (u) profile."""
    out = {}
    for s in SCALES:
        Rs = gaussian_filter(R, s)
        gv, gu = np.gradient(Rs)
        av, au = np.abs(gv), np.abs(gu)
        for nm, A in (("gv", av), ("gu", au)):
            out[f"row_{nm}_{s}"] = A.mean(1)
            out[f"col_{nm}_{s}"] = A.mean(0)
            out[f"rowmax_{nm}_{s}"] = A.max(1)
            out[f"colmax_{nm}_{s}"] = A.max(0)
        out[f"rowstd_{s}"] = Rs.std(1)
        out[f"colstd_{s}"] = Rs.std(0)
        out[f"rowrng_{s}"] = Rs.max(1) - Rs.min(1)
        out[f"colrng_{s}"] = Rs.max(0) - Rs.min(0)
        # number of "transitions" crossed: local maxima of |grad| above a threshold
        thr = np.quantile(av, 0.85)
        out[f"rowcnt_{s}"] = (av > thr).mean(1)
        thr = np.quantile(au, 0.85)
        out[f"colcnt_{s}"] = (au > thr).mean(0)
    out["row_mask"] = mask.mean(1)
    out["col_mask"] = mask.mean(0)
    return out


def image_profiles(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    R = a[..., 0] / 255.0
    mask = (a[..., 2] > 127).astype(np.float32)
    R = (R - R.mean()) / (R.std() + 1e-6)
    return row_bank(R, mask)


def featurise(bank, fixed, axis):
    """axis 0 -> row profiles (fixed coord = v); axis 1 -> col profiles."""
    pre = "row" if axis == 0 else "col"
    names = [k for k in bank if k.startswith(pre)]
    L = 256
    x = fixed * (L - 1)
    feats, cols = [], []
    for k in sorted(names):
        p = bank[k].astype(np.float64)
        ps = gaussian_filter1d(p, 3, mode="nearest")
        mu, sd = ps.mean(), ps.std() + 1e-9
        z = (ps - mu) / sd
        r = np.argsort(np.argsort(ps)) / (L - 1)
        for tag, arr in (("z", z), ("r", r)):
            v = np.interp(x, np.arange(L), arr)
            feats.append(v); cols.append(f"{k}_{tag}")
            for off in (-15, 15, -31, 31):
                v2 = np.interp(np.clip(x + off, 0, L - 1), np.arange(L), arr)
                feats.append(v2 - v); cols.append(f"{k}_{tag}_d{off}")
        # local peak-ness
        feats.append(np.interp(x, np.arange(L), z) - np.interp(x, np.arange(L),
                     gaussian_filter1d(z, 12, mode="nearest")))
        cols.append(f"{k}_peak")
    F = np.stack(feats, 1)          # (16, nfeat)
    return F, cols


def build(public_dir, df, fixed16, cache=None):
    Xs, keys = [], None
    for i, rel in enumerate(df.survey_image.values):
        bank = image_profiles(os.path.join(public_dir, rel.replace("/", os.sep)))
        for a in (0, 1):
            F, cols = featurise(bank, fixed16, a)
            Xs.append(F)
            keys = cols
    X = np.concatenate(Xs, 0)       # (n*2*16, nfeat)
    return X, keys
