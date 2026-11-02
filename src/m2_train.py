"""M2: 1D-conv patch tokenizer + lightweight Transformer encoder, dual head.

Same split, seed, loss and evaluation as M1; architecture and LR schedule differ.
Patch rationale: at 100 Hz, 25 samples = 250 ms, roughly one P-QRS-T complex at
normal rates; stride 12 (50% overlap) yields 82 tokens, enough for inter-beat
attention without making attention over 1,000 raw samples.

Usage: python src/m2_train.py [--epochs 25] [--tag m2_transformer]
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
from m1_train import evaluate, masked_bce, predict, set_seed  # noqa: E402

FIGS = ROOT / "figures"
CKPT = ROOT / "models"


class PatchTransformer(nn.Module):
    def __init__(self, n_sup=5, n_rhy=12, patch=25, stride=12, d_model=128,
                 depth=4, nhead=4, ff=512, drop=0.1, length=1000):
        super().__init__()
        self.tok = nn.Sequential(
            nn.Conv1d(12, d_model, patch, stride, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU())
        n_tok = (length - patch) // stride + 1
        self.n_tokens = n_tok
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, n_tok + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout=drop, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop)
        self.head_sup = nn.Linear(d_model, n_sup)
        self.head_rhy = nn.Linear(d_model, n_rhy)

    def forward(self, x):
        t = self.tok(x).transpose(1, 2)              # B, T, D
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], dim=1) + self.pos
        t = self.enc(t)
        z = self.drop(self.norm(t[:, 0]))            # CLS token
        return self.head_sup(z), self.head_rhy(z)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patch", type=int, default=25)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--tag", default="m2_transformer")
    args = ap.parse_args()

    FIGS.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = load_meta()
    sup, rhy = label_columns(meta)
    ys, yr, ms, mr = build_labels(meta, sup, rhy)
    rows = split_rows(meta, load_split())
    cache = PROC / "ptbxl_100hz_filt.npy"
    ds = {k: ECGDataset(cache, rows[k], ys, yr, ms, mr) for k in rows}
    dl = {
        "train": DataLoader(ds["train"], batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0, drop_last=True),
        "val": DataLoader(ds["val"], batch_size=256, shuffle=False,
                          num_workers=args.workers, pin_memory=True,
                          persistent_workers=args.workers > 0),
        "test": DataLoader(ds["test"], batch_size=256, shuffle=False,
                           num_workers=args.workers, pin_memory=True,
                           persistent_workers=args.workers > 0),
    }

    model = PatchTransformer(len(sup), len(rhy), args.patch, args.stride,
                             args.d_model, args.depth, args.nhead).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"train/val/test = {len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])} | "
          f"tokens={model.n_tokens} params={n_params}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(dl["train"])
    total_steps = args.epochs * steps_per_epoch
    warmup = min(args.warmup_steps, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best = {"score": -1.0, "epoch": -1}
    hist = []
    t0 = time.time()
    gstep = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, ysu, yrh, msu, mrh in dl["train"]:
            x = x.to(device, non_blocking=True)
            ysu, yrh = ysu.to(device, non_blocking=True), yrh.to(device, non_blocking=True)
            msu, mrh = msu.to(device, non_blocking=True), mrh.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ls, lr = model(x)
            loss = masked_bce(ls, ysu, msu) + masked_bce(lr, yrh, mrh)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            gstep += 1
            tot += loss.detach().item()
            nb += 1
        val = evaluate(model, dl["val"], device, sup, rhy)
        rec = {"epoch": epoch, "train_loss": tot / max(nb, 1),
               "val_super_auc": val["super"]["macro_auc"],
               "val_rhythm_auc": val["rhythm"]["macro_auc"],
               "val_score": val["score"], "lr_end": sched.get_last_lr()[0]}
        hist.append(rec)
        print(f"epoch {epoch:02d} loss {rec['train_loss']:.4f} "
              f"val sup {rec['val_super_auc']:.4f} rhy {rec['val_rhythm_auc']:.4f} "
              f"score {rec['val_score']:.4f} lr {rec['lr_end']:.2e}", flush=True)
        if val["score"] > best["score"]:
            best = {"score": val["score"], "epoch": epoch}
            torch.save(model.state_dict(), CKPT / f"{args.tag}.pt")
        elif epoch - best["epoch"] >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    model.load_state_dict(torch.load(CKPT / f"{args.tag}.pt", map_location=device))
    val = evaluate(model, dl["val"], device, sup, rhy)
    test = evaluate(model, dl["test"], device, sup, rhy)
    ys_te, ps_te, yr_te, pr_te = predict(model, dl["test"], device)
    ys_va, ps_va, yr_va, pr_va = predict(model, dl["val"], device)

    out = {
        "tag": args.tag,
        "config": vars(args) | {
            "model": "PatchTransformer", "n_tokens": int(model.n_tokens),
            "params": n_params, "pos_encoding": "learned, CLS token prepended",
            "loss": "masked BCE per head, equal weight",
            "schedule": f"{warmup}-step linear warmup + cosine",
            "selection": "mean(super macro-AUC, rhythm macro-AUC) on val"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device),
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "classes": {"super": sup, "rhythm": rhy},
        "best_epoch": best["epoch"],
        "elapsed_sec": round(time.time() - t0, 1),
        "history": hist, "val": val, "test": test,
    }
    (RESULTS / f"{args.tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(RESULTS / f"{args.tag}_pred.npz",
             test_ecg_ids=meta["ecg_id"].to_numpy()[rows["test"]],
             test_super=ps_te, test_rhythm=pr_te,
             test_y_super=ys_te, test_y_rhythm=yr_te,
             val_ecg_ids=meta["ecg_id"].to_numpy()[rows["val"]],
             val_super=ps_va, val_rhythm=pr_va,
             val_y_super=ys_va, val_y_rhythm=yr_va)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ep = [h["epoch"] for h in hist]
    ax[0].plot(ep, [h["train_loss"] for h in hist], marker="o")
    ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [h["val_super_auc"] for h in hist], marker="o", label="super macro-AUC")
    ax[1].plot(ep, [h["val_rhythm_auc"] for h in hist], marker="o", label="rhythm macro-AUC")
    ax[1].axhline(best["score"], ls="--", lw=1, color="gray", label="selected score")
    ax[1].set_title("validation macro-AUC"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIGS / f"{args.tag}_curves.png", dpi=150)
    plt.close(fig)

    print("TEST super  macro-AUC %.4f  macro-F1@0.5 %.4f" %
          (test["super"]["macro_auc"], test["super"]["macro_f1_at_0.5"]))
    print("TEST rhythm macro-AUC %.4f  macro-F1@0.5 %.4f" %
          (test["rhythm"]["macro_auc"], test["rhythm"]["macro_f1_at_0.5"]))
    print("params", n_params, "best_epoch", best["epoch"], "elapsed_s", out["elapsed_sec"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
