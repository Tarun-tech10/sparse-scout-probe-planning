"""Train on all training rows (n seeds) and write submission.csv."""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (NPROBE, NAX, PARK_GRID, load_meta, catalog_arrays, load_images,
                  encode_labels, decode_park_batch)
from train2 import train_one, marg_blend, decode_all


def write_submission(ids, plans, park, codes, path):
    rows = []
    for i, pid in enumerate(ids):
        pl = [codes[j] for j in plans[i]]
        assert len(set(pl)) == 4
        assert sum(1 for c in pl if codes.index(c) < NAX) == 2
        rows.append((pid, json.dumps(pl), float(park[i, 0]), float(park[i, 1])))
    df = pd.DataFrame(rows, columns=["id", "probe_order", "park_u", "park_v"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=r"c:\Users\tarun\OneDrive\Desktop\ERIS\sparse_scout")
    ap.add_argument("--out", default=r"c:\Users\tarun\OneDrive\Desktop\ERIS\sparse_scout_sol\working\submission.csv")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ls", type=float, default=0.03)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--prof-stride", type=int, default=8)
    ap.add_argument("--lossw", default="1.0,0.6,1.0,0.6")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--wts", default="0.4,0.2,0.4")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = "cuda"
    tr, te, cat = load_meta(args.public)
    codes, axis, fixed, idx = catalog_arrays(cat)
    ypos, ypark = encode_labels(tr, idx)
    Xtr = torch.from_numpy(load_images(args.public, list(tr.survey_image), size=args.size)
                           ).permute(0, 3, 1, 2).contiguous()
    Xte = torch.from_numpy(load_images(args.public, list(te.survey_image), size=args.size)
                           ).permute(0, 3, 1, 2).contiguous()
    cfg = dict(arch=args.arch, epochs=args.epochs, bs=args.bs, lr=args.lr, wd=args.wd,
               ls=args.ls, amp=True, tta=args.tta, fixed16=fixed[:NAX], width=args.width,
               prof_stride=args.prof_stride, aug=True, pretrained=True,
               lossw=tuple(float(v) for v in args.lossw.split(",")))
    n = len(te)
    SEL = np.zeros((n, NPROBE)); POS = np.zeros((n, 4, NPROBE))
    HEAT = np.zeros((n, PARK_GRID, PARK_GRID)); PM = np.zeros((n, NPROBE))
    t0 = time.time()
    for s in range(args.seeds):
        print(f"[seed {s}] {time.time()-t0:.0f}s", flush=True)
        _, outs = train_one(Xtr, ypos, ypark, Xte[:1], cfg, device, seed=1000 + s, Xte=Xte)
        S, P, H, M = outs[1]
        SEL += S; POS += P; HEAT += H; PM += M
    SEL /= args.seeds; POS /= args.seeds; HEAT /= args.seeds; PM /= args.seeds
    HEAT /= HEAT.sum((1, 2), keepdims=True)
    wts = tuple(float(v) for v in args.wts.split(","))
    plans = decode_all(marg_blend(SEL, POS, PM, wts), POS)
    park = decode_park_batch(HEAT)
    df = write_submission(list(te.id), plans, park, codes, args.out)
    np.savez_compressed(os.path.join(os.path.dirname(args.out), "test_preds.npz"),
                        sel=SEL, pos=POS, heat=HEAT, pm=PM, ids=np.array(list(te.id)))
    print(df.head())
    print(f"wrote {args.out} ({len(df)} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
