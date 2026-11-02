"""Evaluate an already-trained M2 checkpoint without retraining.

Why this exists: the wide Transformer run (`--d-model 192 --depth 6`) trained to
early stop and saved its best checkpoint, then the process died during the final
val/test evaluation with OpenBLAS "Memory allocation still failed after 10
retries" (system RAM was exhausted). Nothing about training failed. This script
re-runs only the evaluation in a memory-light process:
num_workers=0, batch 128, OPENBLAS/OMP single-threaded.

It also rebuilds the epoch history from the captured training log when --log is
given, so results/<tag>.json keeps the same schema as a normal run. Elapsed time
is recorded as null because the original run's wall clock is not recoverable.

Usage:
  python src/m2_eval_ckpt.py --tag m2_transformer_wide \
      --d-model 192 --depth 6 --nhead 6 --log <training log path>
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from ecg_data import (PROC, RESULTS, ROOT, ECGDataset, build_labels,  # noqa: E402
                      label_columns, load_meta, load_split, split_rows)
from m1_train import evaluate, predict, set_seed  # noqa: E402
from m2_train import PatchTransformer  # noqa: E402

FIGS = ROOT / "figures"
CKPT = ROOT / "models"
EPOCH_RE = re.compile(
    r"^epoch\s+(\d+)\s+loss\s+([\d.]+)\s+val sup\s+([\d.]+)\s+rhy\s+([\d.]+)\s+"
    r"score\s+([\d.]+)\s+lr\s+([\deE.+-]+)")


def parse_history(log_path: Path) -> list:
    hist = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = EPOCH_RE.match(line.strip())
        if m:
            ep, loss, sup, rhy, score, lr = m.groups()
            hist.append({"epoch": int(ep), "train_loss": float(loss),
                         "val_super_auc": float(sup), "val_rhythm_auc": float(rhy),
                         "val_score": float(score), "lr_end": float(lr)})
    return hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--patch", type=int, default=25)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", default=None)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    FIGS.mkdir(exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = load_meta()
    sup, rhy = label_columns(meta)
    ys, yr, ms, mr = build_labels(meta, sup, rhy)
    rows = split_rows(meta, load_split())
    cache = PROC / "ptbxl_100hz_filt.npy"
    # num_workers=0: the point of this script is to stay inside a small RAM budget.
    ds = {k: ECGDataset(cache, rows[k], ys, yr, ms, mr) for k in ("val", "test")}
    dl = {k: DataLoader(ds[k], batch_size=args.batch, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
          for k in ds}

    model = PatchTransformer(len(sup), len(rhy), args.patch, args.stride,
                             args.d_model, args.depth, args.nhead).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ckpt = CKPT / f"{args.tag}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"loaded {ckpt} | tokens={model.n_tokens} params={n_params}", flush=True)

    hist = parse_history(Path(args.log)) if args.log else []
    best_epoch = (max(hist, key=lambda h: h["val_score"])["epoch"]
                  if hist else None)

    val = evaluate(model, dl["val"], device, sup, rhy)
    test = evaluate(model, dl["test"], device, sup, rhy)
    ys_te, ps_te, yr_te, pr_te = predict(model, dl["test"], device)
    ys_va, ps_va, yr_va, pr_va = predict(model, dl["val"], device)

    out = {
        "tag": args.tag,
        "recovered_from_checkpoint": True,
        "recovery_note": ("training completed and saved the best checkpoint; the "
                          "original process died in final evaluation from system "
                          "RAM exhaustion (OpenBLAS allocation failure). This run "
                          "re-evaluates the checkpoint only; elapsed_sec unknown."),
        "config": {"epochs": args.epochs, "batch": args.batch, "lr": 1e-3,
                   "wd": 1e-4, "warmup_steps": 200, "patience": 8,
                   "workers": 4, "seed": args.seed, "patch": args.patch,
                   "stride": args.stride, "d_model": args.d_model,
                   "depth": args.depth, "nhead": args.nhead,
                   "model": "PatchTransformer", "n_tokens": int(model.n_tokens),
                   "params": n_params, "pos_encoding": "learned, CLS token prepended",
                   "loss": "masked BCE per head, equal weight",
                   "schedule": "200-step linear warmup + cosine",
                   "selection": "mean(super macro-AUC, rhythm macro-AUC) on val"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device),
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "classes": {"super": sup, "rhythm": rhy},
        "best_epoch": best_epoch,
        "elapsed_sec": None,
        "history": hist,
        "val": val, "test": test,
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

    if hist:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [h["epoch"] for h in hist]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].plot(ep, [h["train_loss"] for h in hist], marker="o")
        ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
        ax[1].plot(ep, [h["val_super_auc"] for h in hist], marker="o", label="super macro-AUC")
        ax[1].plot(ep, [h["val_rhythm_auc"] for h in hist], marker="o", label="rhythm macro-AUC")
        ax[1].axhline(max(h["val_score"] for h in hist), ls="--", lw=1,
                      color="gray", label="selected score")
        ax[1].set_title("validation macro-AUC"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(FIGS / f"{args.tag}_curves.png", dpi=150)
        plt.close(fig)

    print("TEST super  macro-AUC %.4f  macro-F1@0.5 %.4f" %
          (test["super"]["macro_auc"], test["super"]["macro_f1_at_0.5"]))
    print("TEST rhythm macro-AUC %.4f  macro-F1@0.5 %.4f" %
          (test["rhythm"]["macro_auc"], test["rhythm"]["macro_f1_at_0.5"]))
    print("params", n_params, "best_epoch", best_epoch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
