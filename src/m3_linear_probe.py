"""M3 linear probe on frozen encoder features.

The SSL encoder has no CLS token, so the probe uses the mean of the 82 token
states (after the final LayerNorm). The encoder stays in eval mode (BatchNorm
statistics frozen); only a linear head is trained on the cached features.

Features are extracted once per initialisation and cached in results/. Cached
features are cached *inputs* to a frozen model; no labels are involved, so this
does not leak val/test labels into training.

Usage:
  python src/m3_linear_probe.py --init joint --fraction 0.1
  python src/m3_linear_probe.py --init scratch --fraction 0.1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from ecg_data import (PROC, RESULTS, ROOT, build_labels, label_columns,  # noqa: E402
                      load_meta, load_split, split_rows)
from ecg_metrics import multilabel_metrics  # noqa: E402
from m1_train import masked_bce, set_seed  # noqa: E402
from m3_finetune import make_subset  # noqa: E402
from m3_pretrain import SSLEncoder  # noqa: E402

CKPT = ROOT / "models"


class Signals(Dataset):
    def __init__(self, cache: Path, n: int):
        self.cache, self.n = str(cache), n
        self.mm = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = np.load(self.cache, mmap_mode="r")
        return torch.from_numpy(np.array(self.mm[i], dtype=np.float32, copy=True))


@torch.no_grad()
def extract(enc, cache, n, device, d_model, workers):
    enc.eval()
    dl = DataLoader(Signals(cache, n), batch_size=512, shuffle=False,
                    num_workers=workers, pin_memory=True)
    out = np.zeros((n, d_model), dtype=np.float32)
    start = 0
    for x in dl:
        h = enc.tokens(x.to(device, non_blocking=True)).mean(dim=1)
        out[start:start + len(h)] = h.cpu().numpy()
        start += len(h)
    return out


def features_for(init, seed, device, workers, d_model=128):
    if init == "scratch":
        set_seed(seed)
        enc = SSLEncoder(d_model=d_model).to(device)
        key = f"scratch_seed{seed}"
    else:
        ck = CKPT / f"m3_ssl_{init}.pt"
        enc = SSLEncoder(d_model=d_model).to(device)
        enc.load_state_dict(torch.load(ck, map_location=device))
        key = f"{init}_{int(ck.stat().st_mtime)}"
    meta = load_meta()
    n = len(meta)
    feat_path = RESULTS / f"m3_feat_{init}.npy"
    meta_path = RESULTS / f"m3_feat_{init}.json"
    if feat_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("key") == key and old.get("n") == n and old.get("d") == d_model:
            print(f"reuse features {feat_path} (key={key})", flush=True)
            return np.load(feat_path), old
    t0 = time.time()
    feats = extract(enc, PROC / "ptbxl_100hz_filt.npy", n, device, d_model, workers)
    np.save(feat_path, feats)
    info = {"key": key, "n": n, "d": d_model, "source": init,
            "pooling": "mean over 82 token states after final LayerNorm",
            "extract_sec": round(time.time() - t0, 1), "cache": str(feat_path)}
    meta_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"extracted features in {info['extract_sec']}s -> {feat_path}", flush=True)
    return feats, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["scratch", "msm", "joint"], default="joint")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    pct = int(round(args.fraction * 100))
    tag = args.tag or f"m3_ft_{args.init}_linear_{pct:03d}"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = load_meta()
    sup, rhy = label_columns(meta)
    ys, yr, ms, mr = build_labels(meta, sup, rhy)
    rows = split_rows(meta, load_split())
    tr_rows, subset = make_subset(meta, rows["train"], args.fraction, args.seed)
    feats, finfo = features_for(args.init, args.seed, device, args.workers)

    X = torch.from_numpy(feats).to(device)
    Ys = torch.from_numpy(ys).to(device)
    Yr = torch.from_numpy(yr).to(device)
    Ms = torch.from_numpy(ms.astype(np.float32)).to(device)
    Mr = torch.from_numpy(mr.astype(np.float32)).to(device)
    tr = torch.from_numpy(tr_rows).to(device)
    va = torch.from_numpy(rows["val"]).to(device)
    te = torch.from_numpy(rows["test"]).to(device)

    d = feats.shape[1]
    head_s = nn.Linear(d, len(sup)).to(device)
    head_r = nn.Linear(d, len(rhy)).to(device)
    opt = torch.optim.AdamW(list(head_s.parameters()) + list(head_r.parameters()),
                            lr=args.lr, weight_decay=args.wd)
    steps = math.ceil(len(tr_rows) / args.batch)
    total = args.epochs * steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1))
    best = {"score": -1.0, "epoch": -1}
    hist = []
    t0 = time.time()

    @torch.no_grad()
    def infer(idx):
        return torch.sigmoid(head_s(X[idx])), torch.sigmoid(head_r(X[idx]))

    @torch.no_grad()
    def score(idx):
        ps, pr = infer(idx)
        ms_ = multilabel_metrics(ys[idx.cpu().numpy()], ps.cpu().numpy(), sup)
        mr_ = multilabel_metrics(yr[idx.cpu().numpy()], pr.cpu().numpy(), rhy)
        return {"super": ms_, "rhythm": mr_,
                "score": float(np.nanmean([ms_["macro_auc"], mr_["macro_auc"]]))}

    for epoch in range(1, args.epochs + 1):
        head_s.train(); head_r.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            ls, lr_ = head_s(X[b]), head_r(X[b])
            loss = masked_bce(ls, Ys[b], Ms[b]) + masked_bce(lr_, Yr[b], Mr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.detach().item()
            nb += 1
        val = score(va)
        rec = {"epoch": epoch, "train_loss": tot / max(nb, 1),
               "val_super_auc": val["super"]["macro_auc"],
               "val_rhythm_auc": val["rhythm"]["macro_auc"],
               "val_score": val["score"]}
        hist.append(rec)
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} loss {rec['train_loss']:.4f} "
                  f"val sup {rec['val_super_auc']:.4f} rhy {rec['val_rhythm_auc']:.4f} "
                  f"score {rec['val_score']:.4f}", flush=True)
        if val["score"] > best["score"]:
            best = {"score": val["score"], "epoch": epoch,
                    "sd": {"head_s": {k: v.detach().clone()
                                      for k, v in head_s.state_dict().items()},
                           "head_r": {k: v.detach().clone()
                                      for k, v in head_r.state_dict().items()}}}
        elif epoch - best["epoch"] >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    head_s.load_state_dict(best["sd"]["head_s"])
    head_r.load_state_dict(best["sd"]["head_r"])
    val = score(va)
    test = score(te)
    with torch.no_grad():
        ps_te, pr_te = infer(te)
        ps_va, pr_va = infer(va)

    out = {
        "tag": tag,
        "config": {"init": args.init, "mode": "linear_probe", "fraction": args.fraction,
                   "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                   "wd": args.wd, "seed": args.seed, "patience": args.patience,
                   "head": "single Linear(d, n_classes) per head, no hidden layer",
                   "features": finfo, "subset": subset,
                   "encoder_frozen": True, "batchnorm_eval": True,
                   "params_trainable": int(sum(p.numel() for p in
                                                list(head_s.parameters()) +
                                                list(head_r.parameters())))},
        "classes": {"super": sup, "rhythm": rhy},
        "best_epoch": best["epoch"],
        "elapsed_sec": round(time.time() - t0, 1),
        "history": hist, "val": val, "test": test,
    }
    (RESULTS / f"{tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(RESULTS / f"{tag}_pred.npz",
             test_ecg_ids=meta["ecg_id"].to_numpy()[rows["test"]],
             test_super=ps_te.cpu().numpy(), test_rhythm=pr_te.cpu().numpy(),
             test_y_super=ys[rows["test"]], test_y_rhythm=yr[rows["test"]],
             val_ecg_ids=meta["ecg_id"].to_numpy()[rows["val"]],
             val_super=ps_va.cpu().numpy(), val_rhythm=pr_va.cpu().numpy(),
             val_y_super=ys[rows["val"]], val_y_rhythm=yr[rows["val"]])

    print("TEST super  macro-AUC %.4f  macro-F1@0.5 %.4f" %
          (test["super"]["macro_auc"], test["super"]["macro_f1_at_0.5"]))
    print("TEST rhythm macro-AUC %.4f  macro-F1@0.5 %.4f" %
          (test["rhythm"]["macro_auc"], test["rhythm"]["macro_f1_at_0.5"]))
    print("best_epoch", best["epoch"], "elapsed_s", out["elapsed_sec"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
