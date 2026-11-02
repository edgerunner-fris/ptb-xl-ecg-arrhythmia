"""M4: temperature scaling and calibration diagnostics.

Works from the saved val/test probabilities of any run (`<tag>_pred.npz`), so no
checkpoint reload is needed. Logits are recovered as log(p/(1-p)) after clipping
to [1e-6, 1-1e-6]; temperature T is fitted per head on the val split by
minimising masked BCE, then applied to test.

The super head only scores records with a diagnostic statement; the rhythm head
only scores records with at least one rhythm statement. Masks are rebuilt from
meta.parquet and the split rows, and the label arrays in the npz are checked
against them so an ordering bug cannot pass silently.

Reported: T, ECE before/after (pooled over classes, and mean per-class ECE),
NLL, Brier, and test AUC before/after (temperature is monotonic, so AUC must not
move; this is a sanity check, not a result).

Usage: python src/m4_calibration.py [--tag m1_resnet] [--bins 15]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from ecg_data import RESULTS, ROOT, build_labels, label_columns, load_meta, load_split, split_rows  # noqa: E402

FIGS = ROOT / "figures"
EPS = 1e-6


def to_logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def masked_bce(p, y, mask):
    p = np.clip(p[mask], EPS, 1 - EPS)
    return float(-np.mean(y[mask] * np.log(p) + (1 - y[mask]) * np.log(1 - p)))


def brier(p, y, mask):
    return float(np.mean((p[mask] - y[mask]) ** 2))


def ece(p, y, mask, n_bins):
    p = p[mask].ravel()
    y = y[mask].ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = len(p)
    e, rows = 0.0, []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                         "n": 0, "conf": None, "acc": None})
            continue
        conf, acc = float(p[sel].mean()), float(y[sel].mean())
        w = float(sel.sum()) / total
        e += w * abs(acc - conf)
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": int(sel.sum()), "conf": conf, "acc": acc})
    return float(e), rows


def per_class_ece(p, y, mask, n_bins):
    vals = []
    for j in range(p.shape[1]):
        e, _ = ece(p[:, [j]], y[:, [j]], mask, n_bins)
        vals.append(e)
    return float(np.mean(vals)), vals


def fit_temperature(z, y, mask):
    def obj(logt):
        t = np.exp(logt)
        return masked_bce(sigmoid(z / t), y, mask)
    r = minimize_scalar(obj, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x)), float(r.fun)


def head_report(p_val, y_val, m_val, p_test, y_test, m_test, names, n_bins):
    z_val, z_test = to_logit(p_val), to_logit(p_test)
    T, _ = fit_temperature(z_val, y_val, m_val)
    pv = sigmoid(z_val / T)
    pt = sigmoid(z_test / T)

    e_b, bins_b = ece(p_test, y_test, m_test, n_bins)
    e_a, bins_a = ece(pt, y_test, m_test, n_bins)
    pc_b, pc_b_list = per_class_ece(p_test, y_test, m_test, n_bins)
    pc_a, pc_a_list = per_class_ece(pt, y_test, m_test, n_bins)

    def auc(p):
        out = {}
        for j, n in enumerate(names):
            yj = y_test[m_test, j]
            out[n] = (float(roc_auc_score(yj, p[m_test, j]))
                      if len(np.unique(yj)) > 1 else None)
        vals = [v for v in out.values() if v is not None]
        return float(np.mean(vals)), out

    auc_b, per_b = auc(p_test)
    auc_a, per_a = auc(pt)
    return {
        "temperature": T,
        "n_val_scored": int(m_val.sum()), "n_test_scored": int(m_test.sum()),
        "ece_pooled_before": e_b, "ece_pooled_after": e_a,
        "ece_mean_per_class_before": pc_b, "ece_mean_per_class_after": pc_a,
        "ece_per_class_before": dict(zip(names, pc_b_list)),
        "ece_per_class_after": dict(zip(names, pc_a_list)),
        "nll_before": masked_bce(p_test, y_test, m_test),
        "nll_after": masked_bce(pt, y_test, m_test),
        "brier_before": brier(p_test, y_test, m_test),
        "brier_after": brier(pt, y_test, m_test),
        "macro_auc_before": auc_b, "macro_auc_after": auc_a,
        "auc_per_class_before": per_b, "auc_per_class_after": per_a,
        "max_abs_auc_change": float(max(
            abs(per_b[k] - per_a[k]) for k in per_b
            if per_b[k] is not None and per_a[k] is not None)),
        "reliability_before": bins_b, "reliability_after": bins_a,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m1_resnet")
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()

    FIGS.mkdir(exist_ok=True)
    j = json.loads((RESULTS / f"{args.tag}.json").read_text(encoding="utf-8"))
    names = j["classes"]
    z = np.load(RESULTS / f"{args.tag}_pred.npz")

    meta = load_meta()
    sup, rhy = label_columns(meta)
    ys, yr, ms, mr = build_labels(meta, sup, rhy)
    rows = split_rows(meta, load_split())
    assert sup == names["super"] and rhy == names["rhythm"], "class order mismatch"
    masks = {"val": {"super": ms[rows["val"]], "rhythm": mr[rows["val"]]},
             "test": {"super": ms[rows["test"]], "rhythm": mr[rows["test"]]}}
    for split in ("val", "test"):
        assert np.array_equal(z[f"{split}_y_super"], ys[rows[split]]), f"{split} super order"
        assert np.array_equal(z[f"{split}_y_rhythm"], yr[rows[split]]), f"{split} rhythm order"

    out = {"tag": args.tag, "model": j["config"].get("model", j["config"].get("init", "?")),
           "source": f"{args.tag}_pred.npz", "n_bins": args.bins,
           "fit_split": "val", "eval_split": "test",
           "note": "temperature is monotonic; macro-AUC must be unchanged",
           "heads": {}}

    fig_data = {}
    for head, label in (("super", "diagnostic superclass"), ("rhythm", "rhythm")):
        r = head_report(z[f"val_{head}"], z[f"val_y_{head}"], masks["val"][head],
                        z[f"test_{head}"], z[f"test_y_{head}"], masks["test"][head],
                        names[head], args.bins)
        out["heads"][head] = r
        fig_data[head] = r
        print(f"{head}: T={r['temperature']:.3f} ECE {r['ece_pooled_before']:.4f} -> "
              f"{r['ece_pooled_after']:.4f} | per-class {r['ece_mean_per_class_before']:.4f} -> "
              f"{r['ece_mean_per_class_after']:.4f} | NLL {r['nll_before']:.4f} -> "
              f"{r['nll_after']:.4f} | AUC {r['macro_auc_before']:.4f} -> "
              f"{r['macro_auc_after']:.4f}", flush=True)

    (RESULTS / "m4_calibration.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, head in zip(axes, ("super", "rhythm")):
        r = fig_data[head]
        for key, lab, c, mk in (("reliability_before", "before", "#d62728", "o"),
                                ("reliability_after", "after", "#1f77b4", "s")):
            bins = [b for b in r[key] if b["n"] > 0]
            ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins],
                    marker=mk, ms=3, lw=1, color=c, label=lab)
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="gray")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
        ax.set_title(f"{head}: ECE {r['ece_pooled_before']:.4f} -> "
                     f"{r['ece_pooled_after']:.4f} (T={r['temperature']:.2f})")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle(f"M4 reliability, {args.tag}, test split, {args.bins} bins")
    fig.tight_layout()
    fig.savefig(FIGS / "m4_reliability.png", dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / "m4_calibration.json", FIGS / "m4_reliability.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
