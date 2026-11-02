"""M3 self-supervised pretraining: masked signal modeling + contrastive (NT-Xent).

Data boundary: only the train split is read. Val/test records never enter the
pretraining loaders, and the script asserts that before training.

Encoder: the same 1D-conv patch tokenizer + Transformer encoder as the M2
downstream model, but without a CLS token (MAE style). Token positions are
transferred to the downstream model at index 1..T when a CLS is prepended.

MSM: 8 random spans of 50 samples (nominal 40% of the 1,000-sample axis) are
zeroed IN THE RAW SIGNAL before tokenization, and a linear decoder reconstructs
the 12x25 window of every token. Loss is MSE only on samples that fall inside a
masked span. Masking samples rather than tokens matters: patches overlap by 13
samples (patch 25, stride 12), so masking token embeddings would let the model
copy most of a masked window from its neighbours and turn MSM into a copy task.

Contrastive: two augmented views of the same record are positives, other records
in the batch are negatives, NT-Xent with tau=0.1 on the mean-pooled token state.
Augmentations are limited to amplitude scaling, circular time shift, additive
Gaussian noise and per-lead dropout; no time stretching that would change rhythm
semantics.

Usage:
  python src/m3_pretrain.py --mode joint --epochs 30 --tag m3_ssl_joint
  python src/m3_pretrain.py --mode msm   --epochs 30 --tag m3_ssl_msm
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from ecg_data import (PROC, RESULTS, ROOT, ECGDataset, build_labels,  # noqa: E402
                      label_columns, load_meta, load_split, split_rows)
from m1_train import set_seed  # noqa: E402

CKPT = ROOT / "models"
N_SPANS, SPAN = 8, 50
MASK_RATIO = N_SPANS * SPAN / 1000.0      # nominal; overlapping spans make the real ratio lower
TAU = 0.1
CL_WEIGHT = 0.2


class SSLEncoder(nn.Module):
    """Tokenizer + Transformer encoder over 82 patches; no CLS token."""

    def __init__(self, patch=25, stride=12, d_model=128, depth=4, nhead=4,
                 ff=512, drop=0.1, length=1000):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.tok = nn.Sequential(
            nn.Conv1d(12, d_model, patch, stride, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU())
        self.n_tokens = (length - patch) // stride + 1
        self.pos = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout=drop, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def tokens(self, x):
        t = self.tok(x).transpose(1, 2)                 # B, T, D
        return self.norm(self.enc(t + self.pos))


class SSLModel(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.enc = SSLEncoder(**kw)
        d = kw.get("d_model", 128)
        p = kw.get("patch", 25)
        self.dec = nn.Linear(d, 12 * p)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128))

    def forward(self, x):
        return self.enc.tokens(x)


def span_mask(b, length, n_spans, span, device):
    m = torch.zeros(b, length, dtype=torch.bool, device=device)
    starts = torch.randint(0, length - span + 1, (b, n_spans), device=device)
    rows = torch.arange(b, device=device)[:, None]
    off = torch.arange(span, device=device)
    for k in range(n_spans):
        m[rows, starts[:, k][:, None] + off] = True
    return m


def augment(x: torch.Tensor) -> torch.Tensor:
    b, nlead, length = x.shape
    dev = x.device
    x = x * torch.empty(b, 1, 1, device=dev).uniform_(0.8, 1.2)
    shifts = torch.randint(-10, 11, (b,), device=dev)
    idx = (torch.arange(length, device=dev)[None, :] - shifts[:, None]) % length
    x = torch.gather(x, 2, idx[:, None, :].expand(b, nlead, length))
    sigma = torch.empty(b, 1, 1, device=dev).uniform_(0.01, 0.05)
    x = x + torch.randn_like(x) * sigma
    keep = (torch.rand(b, nlead, 1, device=dev) >= 0.05).float()
    return x * keep


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    b = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.t() / tau
    eye = torch.eye(2 * b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    target = (torch.arange(2 * b, device=z.device) + b) % (2 * b)
    return F.cross_entropy(sim, target)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["msm", "joint"], default="joint")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patch", type=int, default=25)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"m3_ssl_{args.mode}"

    CKPT.mkdir(exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = load_meta()
    sup, rhy = label_columns(meta)
    ys, yr, ms, mr = build_labels(meta, sup, rhy)
    rows = split_rows(meta, load_split())
    train_rows = rows["train"]
    val_test = set(rows["val"].tolist()) | set(rows["test"].tolist())
    assert not (set(train_rows.tolist()) & val_test), "train overlaps val/test"
    print(f"pretrain records={len(train_rows)} patients="
          f"{meta.iloc[train_rows]['patient_id'].nunique()} (train split only)",
          flush=True)

    cache = PROC / "ptbxl_100hz_filt.npy"
    ds = ECGDataset(cache, train_rows, ys, yr, ms, mr)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)

    model = SSLModel(patch=args.patch, stride=args.stride, d_model=args.d_model,
                     depth=args.depth, nhead=args.nhead).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # Patch targets: (B, T, 12*patch), matching the decoder's output layout.
    print(f"tokens={model.enc.n_tokens} params={n_params} mode={args.mode}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = len(dl)
    total = args.epochs * steps
    warmup = min(args.warmup_steps, total // 10)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    hist = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_m, tot_c, nb, frac = 0.0, 0.0, 0, 0.0
        for x, _ys, _yr, _ms, _mr in dl:
            x = x.to(device, non_blocking=True)
            b, _nl, length = x.shape
            n_tok = model.enc.n_tokens
            sm = span_mask(b, length, N_SPANS, SPAN, device)       # B,1000 bool
            opt.zero_grad(set_to_none=True)

            target = x.unfold(2, args.patch, args.stride)          # B,12,T,patch
            target = target.permute(0, 2, 1, 3)                    # B,T,12,patch
            x_in = x * (~sm).unsqueeze(1).float()
            pred = model.dec(model(x_in)).reshape(b, n_tok, 12, args.patch)
            idx = (12 * torch.arange(n_tok, device=device)[:, None]
                   + torch.arange(args.patch, device=device)[None, :])
            w = sm[:, idx].unsqueeze(2)                            # B,T,1,patch
            loss_m = ((pred - target) ** 2 * w).sum() / (w.sum() * 12).clamp(min=1.0)
            frac += sm.float().mean().item()

            if args.mode == "joint":
                v1, v2 = augment(x), augment(x)
                z1 = model.proj(model.enc.tokens(v1).mean(1))
                z2 = model.proj(model.enc.tokens(v2).mean(1))
                loss_c = nt_xent(z1, z2, TAU)
                loss = loss_m + CL_WEIGHT * loss_c
            else:
                loss_c = torch.zeros((), device=device)
                loss = loss_m

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot_m += loss_m.detach().item()
            tot_c += loss_c.detach().item()
            nb += 1
        rec = {"epoch": epoch, "msm_mse": tot_m / max(nb, 1),
               "cl_loss": tot_c / max(nb, 1) if args.mode == "joint" else None,
               "actual_mask_ratio": frac / max(nb, 1),
               "total": (tot_m + CL_WEIGHT * tot_c) / max(nb, 1) if args.mode == "joint"
                        else tot_m / max(nb, 1)}
        hist.append(rec)
        print(f"epoch {epoch:02d} msm {rec['msm_mse']:.4f} "
              f"cl {rec['cl_loss'] if rec['cl_loss'] is None else round(rec['cl_loss'], 4)} "
              f"mask {rec['actual_mask_ratio']:.3f} lr {sched.get_last_lr()[0]:.2e}",
              flush=True)

    torch.save(model.enc.state_dict(), CKPT / f"{tag}.pt")
    out = {
        "tag": tag, "mode": args.mode,
        "config": {"patch": args.patch, "stride": args.stride,
                   "d_model": args.d_model, "depth": args.depth, "nhead": args.nhead,
                   "n_tokens": int(model.enc.n_tokens), "params_ssl_model": n_params,
                   "params_encoder": sum(p.numel() for p in model.enc.parameters()),
                   "mask_ratio_nominal": MASK_RATIO, "n_spans": N_SPANS,
                   "span_samples": SPAN,
                   "masking": "raw-sample span mask before tokenization; "
                              "MSE on masked samples only (avoids overlap leakage)",
                   "tau": TAU, "cl_weight": CL_WEIGHT,
                   "augment": ["amplitude U(0.8,1.2) per record",
                               "circular time shift +-10 samples (100 ms)",
                               "Gaussian noise sigma U(0.01,0.05)",
                               "per-lead dropout p=0.05"],
                   "batch": args.batch, "lr": args.lr, "wd": args.wd,
                   "epochs": args.epochs, "warmup_steps": warmup,
                   "schedule": "linear warmup + cosine", "seed": args.seed,
                   "workers": args.workers},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device),
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "data": {"split": "train only", "n_records": int(len(train_rows)),
                 "n_patients": int(meta.iloc[train_rows]["patient_id"].nunique()),
                 "val_test_overlap": 0},
        "elapsed_sec": round(time.time() - t0, 1),
        "history": hist,
        "checkpoint": str(CKPT / f"{tag}.pt"),
    }
    (RESULTS / f"{tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", CKPT / f"{tag}.pt", "elapsed_s", out["elapsed_sec"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
