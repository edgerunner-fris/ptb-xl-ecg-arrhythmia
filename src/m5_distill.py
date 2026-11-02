"""M5 (optional): distil the M1 ResNet teacher into the patch Transformer.

Teacher: m1_resnet (3.86M params, test super/rhythm macro-AUC 0.9239/0.9320 on
statement-masked records). Student: the same PatchTransformer as M2 (0.84M).
Loss = masked BCE against hard labels + 0.5 * masked BCE against teacher
probabilities. Teacher is frozen and in eval mode.

The point is a size/accuracy trade-off curve, not a claim that the student wins.

Usage: python src/m5_distill.py [--teacher m1_resnet] [--epochs 25]
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
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from ecg_data import PROC, RESULTS, ROOT, ECGDataset, build_labels, label_columns, load_meta, load_split, split_rows  # noqa: E402
from m1_train import ResNet1d, evaluate, masked_bce, predict, set_seed  # noqa: E402
from m2_train import PatchTransformer  # noqa: E402

CKPT = ROOT / "models"
KD_WEIGHT = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="m1_resnet")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="m5_kd_student")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_meta()
    sup, rhy = label_columns(meta)
    ys, yr, ms, mr = build_labels(meta, sup, rhy)
    rows = split_rows(meta, load_split())
    cache = PROC / "ptbxl_100hz_filt.npy"
    ds = {k: ECGDataset(cache, rows[k], ys, yr, ms, mr) for k in rows}
    dl = {k: DataLoader(ds[k], batch_size=args.batch if k == "train" else 256,
                        shuffle=(k == "train"), num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        drop_last=(k == "train")) for k in ds}

    teacher = ResNet1d(len(sup), len(rhy)).to(device)
    teacher.load_state_dict(torch.load(CKPT / f"{args.teacher}.pt", map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = PatchTransformer(len(sup), len(rhy)).to(device)
    n_student = sum(p.numel() for p in student.parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"teacher {n_teacher} -> student {n_student} params", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = len(dl["train"])
    total = args.epochs * steps
    warmup = min(200, max(total // 10, 1))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best = {"score": -1.0, "epoch": -1}
    hist = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        student.train()
        tot, nb = 0.0, 0
        for x, ysu, yrh, msu, mrh in dl["train"]:
            x = x.to(device, non_blocking=True)
            ysu, yrh = ysu.to(device, non_blocking=True), yrh.to(device, non_blocking=True)
            msu, mrh = msu.to(device, non_blocking=True), mrh.to(device, non_blocking=True)
            with torch.no_grad():
                ts, tr = teacher(x)
                ts, tr = torch.sigmoid(ts), torch.sigmoid(tr)
            ls, lr = student(x)
            hard = masked_bce(ls, ysu, msu) + masked_bce(lr, yrh, mrh)
            soft = masked_bce(ls, ts, msu) + masked_bce(lr, tr, mrh)
            loss = hard + KD_WEIGHT * soft
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.detach().item()
            nb += 1
        val = evaluate(student, dl["val"], device, sup, rhy)
        rec = {"epoch": epoch, "train_loss": tot / max(nb, 1),
               "val_super_auc": val["super"]["macro_auc"],
               "val_rhythm_auc": val["rhythm"]["macro_auc"],
               "val_score": val["score"]}
        hist.append(rec)
        print(f"epoch {epoch:02d} loss {rec['train_loss']:.4f} "
              f"val sup {rec['val_super_auc']:.4f} rhy {rec['val_rhythm_auc']:.4f} "
              f"score {rec['val_score']:.4f}", flush=True)
        if val["score"] > best["score"]:
            best = {"score": val["score"], "epoch": epoch}
            torch.save(student.state_dict(), CKPT / f"{args.tag}.pt")
        elif epoch - best["epoch"] >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    student.load_state_dict(torch.load(CKPT / f"{args.tag}.pt", map_location=device))
    val = evaluate(student, dl["val"], device, sup, rhy)
    test = evaluate(student, dl["test"], device, sup, rhy)
    _ys, ps, _yr, pr = predict(student, dl["test"], device)
    _ysv, psv, _yrv, prv = predict(student, dl["val"], device)

    out = {
        "tag": args.tag, "teacher": args.teacher,
        "config": {"model": "PatchTransformer", "params_student": n_student,
                   "params_teacher": n_teacher, "kd_weight": KD_WEIGHT,
                   "loss": "masked BCE(hard) + 0.5 * masked BCE(soft teacher probs)",
                   "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                   "seed": args.seed, "selection": "mean val macro-AUC"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device),
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "classes": {"super": sup, "rhythm": rhy},
        "best_epoch": best["epoch"], "elapsed_sec": round(time.time() - t0, 1),
        "history": hist, "val": val, "test": test,
    }
    (RESULTS / f"{args.tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(RESULTS / f"{args.tag}_pred.npz",
             test_ecg_ids=meta["ecg_id"].to_numpy()[rows["test"]],
             test_super=ps, test_rhythm=pr,
             test_y_super=ys[rows["test"]], test_y_rhythm=yr[rows["test"]],
             val_ecg_ids=meta["ecg_id"].to_numpy()[rows["val"]],
             val_super=psv, val_rhythm=prv,
             val_y_super=ys[rows["val"]], val_y_rhythm=yr[rows["val"]])
    print("TEST super  macro-AUC %.4f  macro-F1@0.5 %.4f" %
          (test["super"]["macro_auc"], test["super"]["macro_f1_at_0.5"]))
    print("TEST rhythm macro-AUC %.4f  macro-F1@0.5 %.4f" %
          (test["rhythm"]["macro_auc"], test["rhythm"]["macro_f1_at_0.5"]))
    print("best_epoch", best["epoch"], "elapsed_s", out["elapsed_sec"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
