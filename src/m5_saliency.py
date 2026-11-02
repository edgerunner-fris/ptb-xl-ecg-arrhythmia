"""M5: gradient x input saliency for the deployed 1D ResNet.

Unlike attention rollout on the Transformer, this attribution is lead-specific:
the gradient of the highest-probability superclass logit is computed w.r.t. the
12x1000 input, and |grad * input| is shown per lead.

Samples: same selection rule as m5_attention.py (3 exact-set matches, 2 mistakes,
highest confidence, diagnostic statement present), but taken from the M1 ResNet.

Concentration metric: the share of attribution mass in the top 10% of
time-samples; 0.10 is uniform, higher means more concentrated.

Usage: python src/m5_saliency.py [--tag m1_resnet]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ecg_data import PROC, RESULTS, ROOT, build_labels, label_columns, load_meta, load_split, split_rows  # noqa: E402
from m1_train import ResNet1d  # noqa: E402

FIGS = ROOT / "figures"
CKPT = ROOT / "models"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def format_set(names, vec):
    on = [n for n, v in zip(names, vec) if v]
    return "{" + ", ".join(on) + "}" if on else "{}"


def top_share(flat, frac=0.1):
    v = np.sort(np.asarray(flat, dtype=np.float64).ravel())[::-1]
    k = max(1, int(round(len(v) * frac)))
    return float(v[:k].sum() / max(v.sum(), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m1_resnet")
    ap.add_argument("--n-correct", type=int, default=3)
    ap.add_argument("--n-wrong", type=int, default=2)
    args = ap.parse_args()
    FIGS.mkdir(exist_ok=True)

    j = json.loads((RESULTS / f"{args.tag}.json").read_text(encoding="utf-8"))
    names = j["classes"]
    meta = load_meta()
    sup, rhy = label_columns(meta)
    ys, yr, ms, mr = build_labels(meta, sup, rhy)
    rows = split_rows(meta, load_split())
    z = np.load(RESULTS / f"{args.tag}_pred.npz")
    stats = json.loads((RESULTS / "m1_norm_stats.json").read_text(encoding="utf-8"))
    mean = np.array(stats["mean_mv"], dtype=np.float64)[:, None]
    std = np.array(stats["std_mv"], dtype=np.float64)[:, None]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1d(len(sup), len(rhy)).to(device)
    model.load_state_dict(torch.load(CKPT / f"{args.tag}.pt", map_location=device))
    model.eval()

    te = rows["test"]
    scored = np.where(ms[te])[0]
    y = z["test_y_super"][scored]
    p = z["test_super"][scored]
    pred = (p >= 0.5).astype(int)
    exact = np.all(pred == y.astype(int), axis=1)
    conf = p.max(axis=1)

    def pick(mask, n):
        idx = np.where(mask)[0]
        return idx[np.argsort(-conf[idx])][:n]

    sel_correct = pick(exact, args.n_correct)
    sel_wrong = pick(~exact, args.n_wrong)
    sel = np.concatenate([sel_correct, sel_wrong])

    mm = np.load(PROC / "ptbxl_100hz_filt.npy", mmap_mode="r")
    recs = []
    for pos in sel:
        g = te[scored[pos]]
        x = np.array(mm[g], dtype=np.float32)
        xt = torch.from_numpy(x[None]).to(device)
        xt.requires_grad_(True)
        ls, _ = model(xt)
        cls = int(torch.argmax(ls[0]))
        model.zero_grad(set_to_none=True)
        ls[0, cls].backward()
        grad = xt.grad.detach()[0].cpu().numpy()
        sal = np.abs(grad * x)
        sal = sal / max(sal.max(), 1e-12)
        recs.append({
            "ecg_id": int(z["test_ecg_ids"][scored[pos]]),
            "correct": bool(exact[pos]),
            "true_set": format_set(names["super"], y[pos]),
            "pred_set": format_set(names["super"], pred[pos]),
            "explained_class": names["super"][cls],
            "explained_prob": float(p[pos, cls]),
            "max_prob": float(conf[pos]),
            "top10pct_mass": top_share(sal),
            "per_lead_top10pct": {n: top_share(sal[k]) for k, n in enumerate(LEADS)},
            "signal_mv": x * std + mean,
            "saliency": sal,
        })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def panel(recs, path, title):
        n = len(recs)
        fig, axes = plt.subplots(n, 1, figsize=(13, 3.0 * n), squeeze=False)
        t = np.arange(1000) / 100.0
        for ax, r in zip(axes[:, 0], recs):
            sig, sal = r["signal_mv"], r["saliency"]
            for k in range(12):
                off = (11 - k) * 3.5
                ax.imshow(sal[k][None, :], aspect="auto", origin="lower",
                          extent=[0, 10, off - 1.6, off + 1.6], cmap="Reds",
                          vmin=0, vmax=1, alpha=0.35)
                ax.plot(t, np.clip(sig[k], -2.0, 2.5) + off, color="black", lw=0.55)
            ax.set_yticks([(11 - k) * 3.5 for k in range(12)])
            ax.set_yticklabels(LEADS, fontsize=7)
            ax.set_xlim(0, 10); ax.set_ylim(-2.0, 12 * 3.5 + 1.0)
            mark = "correct" if r["correct"] else "WRONG"
            ax.set_title(f"ecg {r['ecg_id']} [{mark}] true {r['true_set']} | "
                         f"pred {r['pred_set']} | explained {r['explained_class']} "
                         f"p={r['explained_prob']:.2f} | top10% mass "
                         f"{r['top10pct_mass']:.2f}", fontsize=8)
            if r is recs[-1]:
                ax.set_xlabel("time (s)")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    panel(recs[:len(sel_correct)], FIGS / "m5_saliency_correct.png",
          f"M5 gradient x input, {args.tag}: 3 highest-confidence exact matches")
    panel(recs[len(sel_correct):], FIGS / "m5_saliency_wrong.png",
          f"M5 gradient x input, {args.tag}: 2 highest-confidence mistakes")

    out = {"tag": args.tag, "method": "gradient x input of the top-probability "
                                      "superclass logit, |grad*x| normalised per record",
           "selection": "highest-confidence exact-set matches/mistakes",
           "n_correct": int(len(sel_correct)), "n_wrong": int(len(sel_wrong)),
           "uniform_top10pct_mass": 0.10,
           "samples": [{k: v for k, v in r.items()
                        if k not in ("signal_mv", "saliency")} for r in recs]}
    (RESULTS / "m5_saliency.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in recs:
        print(f"ecg {r['ecg_id']} {'OK ' if r['correct'] else 'ERR'} "
              f"true {r['true_set']} pred {r['pred_set']} "
              f"explained {r['explained_class']} p={r['explained_prob']:.3f} "
              f"top10%={r['top10pct_mass']:.3f}")
    print("wrote", FIGS / "m5_saliency_correct.png", FIGS / "m5_saliency_wrong.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
