"""M3 downstream fine-tuning at a fixed label fraction.

Three modes:
  full    - all parameters trainable (same schedule as M1/M2)
  adapter - encoder frozen; a Houlsby-style bottleneck adapter after every
            Transformer block, plus trainable CLS and heads
  linear  - handled by m3_linear_probe.py (frozen features, linear head only)

Initialisation is either `scratch` or a pretrained encoder from
models/m3_ssl_{msm,joint}.pt. Label fractions are drawn at the PATIENT level
from the train split and cached in splits/, so the pretrained and scratch arms
see exactly the same records at a given fraction.

Usage:
  python src/m3_finetune.py --init joint --mode full --fraction 0.1
  python src/m3_finetune.py --init scratch --mode adapter --fraction 1.0
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
from ecg_data import (PROC, RESULTS, ROOT, SPLITS, ECGDataset, build_labels,  # noqa: E402
                      label_columns, load_meta, load_split, split_rows)
from m1_train import evaluate, masked_bce, predict, set_seed  # noqa: E402
from m2_train import PatchTransformer  # noqa: E402

FIGS = ROOT / "figures"
CKPT = ROOT / "models"


class Adapter(nn.Module):
    def __init__(self, d, r=16, drop=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, r)
        self.up = nn.Linear(r, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.up(F.gelu(self.down(self.ln(x)))))


def load_ssl(model: PatchTransformer, path: Path) -> dict:
    src = torch.load(path, map_location="cpu")
    dst = model.state_dict()
    n_pos = src["pos"].shape[1]
    if n_pos + 1 != dst["pos"].shape[1]:
        raise ValueError(f"token mismatch: ckpt {n_pos} vs model {dst['pos'].shape[1] - 1}")
    with torch.no_grad():
        dst["pos"][:, 1:] = src["pos"]
    transfer = []
    for k, v in src.items():
        if k.startswith(("tok.", "enc.", "norm.")):
            if dst[k].shape != v.shape:
                raise ValueError(f"shape mismatch {k}: {tuple(v.shape)} vs {tuple(dst[k].shape)}")
            dst[k] = v
            transfer.append(k)
    model.load_state_dict(dst)
    return {"checkpoint": str(path), "n_tensors_transferred": len(transfer),
            "pos_tokens_transferred": int(n_pos),
            "fresh": ["cls", "head_sup", "head_rhy", "pos[:,0]"]}


def subset_path(fraction: float, seed: int) -> Path:
    return SPLITS / f"m3_frac_{int(round(fraction * 100)):03d}_seed{seed}.json"


def make_subset(meta, train_rows, fraction, seed):
    if fraction >= 1.0:
        return train_rows, {"fraction": 1.0, "n_patients": int(
            meta.iloc[train_rows]["patient_id"].nunique()), "n_records": int(len(train_rows))}
    p = subset_path(fraction, seed)
    if p.exists():
        info = json.loads(p.read_text(encoding="utf-8"))
        keep = np.array(info["patient_ids"], dtype=np.int64)
    else:
        pats = meta.iloc[train_rows]["patient_id"].to_numpy()
        uniq = np.array(sorted(set(pats.tolist())), dtype=np.int64)
        rng = np.random.default_rng(seed)
        n = max(1, int(round(fraction * len(uniq))))
        keep = np.sort(rng.choice(uniq, n, replace=False))
        sel = train_rows[np.isin(pats, keep)]
        info = {
            "fraction": fraction, "seed": seed, "n_patients_total": int(len(uniq)),
            "n_patients": int(len(keep)), "n_records": int(len(sel)),
            "patient_ids": keep.tolist(),
            "ecg_ids": meta.iloc[sel]["ecg_id"].tolist(),
        }
        p.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    pats = meta.iloc[train_rows]["patient_id"].to_numpy()
    sel = train_rows[np.isin(pats, keep)]
    return sel, {"fraction": fraction, "n_patients": int(len(set(keep.tolist()))),
                 "n_records": int(len(sel)), "subset_file": str(p)}


def prevalence(meta, rows, sup, rhy):
    out = {"super": {}, "rhythm": {}}
    for head, names in (("super", sup), ("rhythm", rhy)):
        cols = [f"{'sup' if head == 'super' else 'rhy'}_{c}" for c in names]
        pos = meta.iloc[rows][cols].to_numpy().sum(axis=0)
        out[head] = {n: int(v) for n, v in zip(names, pos)}
    return out


def freeze_for_adapter(model, d_model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.cls.requires_grad_(True)          # pretraining used mean pooling, so CLS is fresh
    model.head_sup.requires_grad_(True)
    model.head_rhy.requires_grad_(True)
    dev = next(model.parameters()).device
    model.adapters = nn.ModuleList([Adapter(d_model) for _ in model.enc.layers]).to(dev)
    for layer, ad in zip(model.enc.layers, model.adapters):
        layer.register_forward_hook(lambda m, i, o, ad=ad: ad(o))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["scratch", "msm", "joint"], default="joint")
    ap.add_argument("--mode", choices=["full", "adapter"], default="full")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patch", type=int, default=25)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    pct = int(round(args.fraction * 100))
    tag = args.tag or f"m3_ft_{args.init}_{args.mode}_{pct:03d}"
    FIGS.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = load_meta()
    sup, rhy = label_columns(meta)
    ys, yr, ms, mr = build_labels(meta, sup, rhy)
    rows = split_rows(meta, load_split())
    tr_rows, subset = make_subset(meta, rows["train"], args.fraction, args.seed)
    print(f"fraction={args.fraction} train records={subset['n_records']} "
          f"patients={subset['n_patients']} init={args.init} mode={args.mode}", flush=True)

    cache = PROC / "ptbxl_100hz_filt.npy"
    ds = {k: ECGDataset(cache, r, ys, yr, ms, mr)
          for k, r in (("train", tr_rows), ("val", rows["val"]), ("test", rows["test"]))}
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
    init_info = {"init": args.init}
    if args.init != "scratch":
        init_info["transfer"] = load_ssl(model, CKPT / f"m3_ssl_{args.init}.pt")
    if args.mode == "adapter":
        freeze_for_adapter(model, args.d_model)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total={n_total} trainable={n_train}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    steps = len(dl["train"])
    total_steps = args.epochs * steps
    warmup = min(args.warmup_steps, max(total_steps // 10, 1))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best = {"score": -1.0, "epoch": -1}
    hist = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.mode == "adapter":
            model.tok.eval()
            model.enc.eval()
            model.norm.eval()
            model.drop.eval()
        tot, nb = 0.0, 0
        for x, ysu, yrh, msu, mrh in dl["train"]:
            x = x.to(device, non_blocking=True)
            ysu, yrh = ysu.to(device, non_blocking=True), yrh.to(device, non_blocking=True)
            msu, mrh = msu.to(device, non_blocking=True), mrh.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ls, lr = model(x)
            loss = masked_bce(ls, ysu, msu) + masked_bce(lr, yrh, mrh)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            sched.step()
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
              f"score {rec['val_score']:.4f}", flush=True)
        if val["score"] > best["score"]:
            best = {"score": val["score"], "epoch": epoch}
            torch.save(model.state_dict(), CKPT / f"{tag}.pt")
        elif epoch - best["epoch"] >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    state = torch.load(CKPT / f"{tag}.pt", map_location=device)
    model.load_state_dict(state)
    val = evaluate(model, dl["val"], device, sup, rhy)
    test = evaluate(model, dl["test"], device, sup, rhy)
    ys_te, ps_te, yr_te, pr_te = predict(model, dl["test"], device)
    ys_va, ps_va, yr_va, pr_va = predict(model, dl["val"], device)

    out = {
        "tag": tag,
        "config": vars(args) | {
            "model": "PatchTransformer", "n_tokens": int(model.n_tokens),
            "params_total": n_total, "params_trainable": n_train,
            "loss": "masked BCE per head, equal weight",
            "schedule": f"{warmup}-step linear warmup + cosine",
            "selection": "mean(super macro-AUC, rhythm macro-AUC) on val",
            "init_info": init_info,
            "adapter": "LN -> Linear(d,16) -> GELU -> Linear(16,d), residual, "
                       "after each Transformer block" if args.mode == "adapter" else None,
            "label_fraction": args.fraction,
            "subset": subset,
            "steps_per_epoch": steps,
            "warmup_steps": warmup,
            "total_optimizer_steps": total_steps,
            "epochs_budget": args.epochs,
            "train_prevalence": prevalence(meta, tr_rows, sup, rhy),
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device),
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "classes": {"super": sup, "rhythm": rhy},
        "best_epoch": best["epoch"],
        "elapsed_sec": round(time.time() - t0, 1),
        "history": hist, "val": val, "test": test,
    }
    (RESULTS / f"{tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(RESULTS / f"{tag}_pred.npz",
             test_ecg_ids=meta["ecg_id"].to_numpy()[rows["test"]],
             test_super=ps_te, test_rhythm=pr_te,
             test_y_super=ys_te, test_y_rhythm=yr_te,
             val_ecg_ids=meta["ecg_id"].to_numpy()[rows["val"]],
             val_super=ps_va, val_rhythm=pr_va,
             val_y_super=ys_va, val_y_rhythm=yr_va)

    print("TEST super  macro-AUC %.4f  macro-F1@0.5 %.4f" %
          (test["super"]["macro_auc"], test["super"]["macro_f1_at_0.5"]))
    print("TEST rhythm macro-AUC %.4f  macro-F1@0.5 %.4f" %
          (test["rhythm"]["macro_auc"], test["rhythm"]["macro_f1_at_0.5"]))
    print("trainable", n_train, "best_epoch", best["epoch"],
          "elapsed_s", out["elapsed_sec"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
