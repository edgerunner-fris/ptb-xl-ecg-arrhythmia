"""M1 baseline: 1D ResNet, dual head (5 diagnostic superclasses + 12 rhythm).

Usage: python src/m1_train.py [--epochs 25] [--tag m1_resnet]
"""
from __future__ import annotations

import argparse
import json
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
from ecg_metrics import multilabel_metrics  # noqa: E402

FIGS = ROOT / "figures"
CKPT = ROOT / "models"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.short = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.short = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm1d(out_c))

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)), inplace=True)
        o = self.bn2(self.conv2(o))
        return F.relu(o + self.short(x), inplace=True)


class ResNet1d(nn.Module):
    def __init__(self, n_sup=5, n_rhy=12, width=64, blocks=(2, 2, 2, 2), drop=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, width, 7, 2, 3, bias=False), nn.BatchNorm1d(width),
            nn.ReLU(inplace=True), nn.MaxPool1d(3, 2, 1))
        self.l1 = self._stage(width, width, blocks[0], 1)
        self.l2 = self._stage(width, width * 2, blocks[1], 2)
        self.l3 = self._stage(width * 2, width * 4, blocks[2], 2)
        self.l4 = self._stage(width * 4, width * 8, blocks[3], 2)
        self.drop = nn.Dropout(drop)
        self.head_sup = nn.Linear(width * 8, n_sup)
        self.head_rhy = nn.Linear(width * 8, n_rhy)

    @staticmethod
    def _stage(in_c, out_c, n, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        layers += [BasicBlock(out_c, out_c, 1) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.l4(self.l3(self.l2(self.l1(x))))
        z = F.adaptive_avg_pool1d(x, 1).flatten(1)
        z = self.drop(z)
        return self.head_sup(z), self.head_rhy(z)


def masked_bce(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    m = mask.unsqueeze(1).to(loss.dtype)
    return (loss * m).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, pr, ys, yr = [], [], [], []
    for x, ysu, yrh, _ms, _mr in loader:
        x = x.to(device, non_blocking=True)
        ls, lr = model(x)
        ps.append(torch.sigmoid(ls).cpu().numpy())
        pr.append(torch.sigmoid(lr).cpu().numpy())
        ys.append(ysu.numpy())
        yr.append(yrh.numpy())
    return (np.concatenate(ys), np.concatenate(ps),
            np.concatenate(yr), np.concatenate(pr))


def evaluate(model, loader, device, sup, rhy):
    ys, ps, yr, pr = predict(model, loader, device)
    ms = multilabel_metrics(ys, ps, sup)
    mr = multilabel_metrics(yr, pr, rhy)
    return {"super": ms, "rhythm": mr,
            "score": float(np.nanmean([ms["macro_auc"], mr["macro_auc"]]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="m1_resnet")
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
    print(f"train/val/test = {len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])} | "
          f"sup={sup} | rhy={rhy}", flush=True)

    model = ResNet1d(len(sup), len(rhy)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = {"score": -1.0, "epoch": -1}
    hist = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, ysu, yrh, msu, mrh in dl["train"]:
            x = x.to(device, non_blocking=True)
            ysu = ysu.to(device, non_blocking=True)
            yrh = yrh.to(device, non_blocking=True)
            msu = msu.to(device, non_blocking=True)
            mrh = mrh.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ls, lr = model(x)
            loss = masked_bce(ls, ysu, msu) + masked_bce(lr, yrh, mrh)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.detach().item()
            nb += 1
        sched.step()
        val = evaluate(model, dl["val"], device, sup, rhy)
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
        "config": vars(args) | {"model": "ResNet1d", "width": 64,
                                "blocks": [2, 2, 2, 2], "params": n_params,
                                "loss": "masked BCE per head, equal weight",
                                "selection": "mean(super macro-AUC, rhythm macro-AUC) on val"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device),
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "classes": {"super": sup, "rhythm": rhy},
        "best_epoch": best["epoch"],
        "elapsed_sec": round(time.time() - t0, 1),
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
