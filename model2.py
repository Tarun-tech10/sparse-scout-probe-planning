"""ScoutNet v2: FPN backbone + axis-shared 1-D profile head + pair head."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NAX = 16
NPROBE = 32
PARK_GRID = 64
IU = np.triu_indices(NAX, 1)          # 120 unordered pairs


def sample_taps(fixed, up, device):
    x = np.clip(np.asarray(fixed, dtype=np.float64) * up - 0.5, 0, up - 1)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, up - 1)
    w = torch.tensor(x - i0, dtype=torch.float32, device=device)
    return (torch.tensor(i0, device=device), torch.tensor(i1, device=device), w)


class ProfileHead(nn.Module):
    """Shared across the two axes: 1-D conv tower on the pooled profile."""

    def __init__(self, cin, cg, width=192, emb=64, nout=5, up=64):
        super().__init__()
        self.up = up
        self.pre = nn.Sequential(
            nn.Conv1d(cin + cg, width, 5, padding=2), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
        )
        self.post = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv1d(width, nout + emb, 1),
        )
        self.nout, self.emb = nout, emb
        self.pair_bias = nn.Linear(emb, 1)
        self.pair_w = nn.Linear(emb, emb, bias=False)

    def forward(self, prof, gctx, taps):
        B, _, L = prof.shape
        h = torch.cat([prof, gctx[:, :, None].expand(-1, -1, L)], 1)
        h = self.pre(h)
        h = F.interpolate(h, size=self.up, mode="linear", align_corners=False)
        h = self.post(h)
        i0, i1, w = taps
        h = h[:, :, i0] * (1 - w) + h[:, :, i1] * w      # (B, nout+emb, 16)
        logits = h[:, :self.nout]                        # (B,nout,16)
        e = h[:, self.nout:].transpose(1, 2)             # (B,16,emb)
        s = torch.einsum("bik,bjk->bij", self.pair_w(e), e)
        s = 0.5 * (s + s.transpose(1, 2))
        b = self.pair_bias(e)[..., 0]                    # (B,16)
        s = s + b[:, :, None] + b[:, None, :]
        pair = s[:, IU[0], IU[1]]                        # (B,120)
        return logits, pair


class ScoutNet2(nn.Module):
    def __init__(self, arch="resnet18", pretrained=True, fpn=128, width=192,
                 prof_stride=8, nheat=PARK_GRID):
        super().__init__()
        import torchvision
        fn = getattr(torchvision.models, arch)
        try:
            net = fn(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:
            net = fn(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.l1, self.l2, self.l3, self.l4 = net.layer1, net.layer2, net.layer3, net.layer4
        with torch.no_grad():
            d = torch.zeros(1, 3, 128, 128)
            h = self.stem(d); a = self.l1(h); b = self.l2(a); c = self.l3(b); e = self.l4(c)
        chans = [a.shape[1], b.shape[1], c.shape[1], e.shape[1]]
        self.lat = nn.ModuleList([nn.Conv2d(ch, fpn, 1) for ch in chans])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fpn, fpn, 3, padding=1), nn.GroupNorm(8, fpn), nn.SiLU())
            for _ in range(3)])
        self.gctx = nn.Sequential(nn.Linear(chans[3], 256), nn.SiLU())
        self.prof_stride = prof_stride
        self.profile = ProfileHead(cin=2 * fpn, cg=256, width=width)
        self.park = nn.Sequential(
            nn.Conv2d(fpn, fpn, 3, padding=1), nn.GroupNorm(8, fpn), nn.SiLU(),
            nn.Conv2d(fpn, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 1, 1))
        self.nheat = nheat

    def forward(self, x, taps):
        h = self.stem(x)
        c2 = self.l1(h); c3 = self.l2(c2); c4 = self.l3(c3); c5 = self.l4(c4)
        p5 = self.lat[3](c5)
        p4 = self.smooth[2](self.lat[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.smooth[1](self.lat[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        p2 = self.smooth[0](self.lat[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest"))
        g = self.gctx(F.adaptive_avg_pool2d(c5, 1).flatten(1))
        pf = p3 if self.prof_stride == 8 else p2

        def prof(t):
            return torch.cat([t.mean(-1), t.amax(-1)], 1)

        lr, pr_r = self.profile(prof(pf), g, taps)                      # axis_0 (rows)
        lc, pr_c = self.profile(prof(pf.transpose(-1, -2)), g, taps)    # axis_1 (cols)
        logits = torch.cat([lr, lc], -1)                                # (B,5,32)
        heat = self.park(p2)
        heat = F.interpolate(heat, size=(self.nheat, self.nheat),
                             mode="bilinear", align_corners=False)[:, 0]
        return logits[:, 0], logits[:, 1:], heat, torch.stack([pr_r, pr_c], 1)
