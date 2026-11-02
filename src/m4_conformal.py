"""M4: split conformal prediction for multi-label ECG, per class.

Protocol
--------
Calibration = val split (2,183 records, patient-disjoint from train/test).
Test        = test split (2,198 records).
For each class j independently:
  nonconformity s = 1 - p(y) for the true label y in {0,1}
  q_j = the ceil((n+1)(1-alpha))-th smallest calibration score
  prediction set = {1 if p >= 1-q_j} union {0 if p <= q_j}
The guarantee is marginal PER CLASS. Joint coverage over all classes is reported
separately, plus a Bonferroni variant (alpha/C per class) that does target joint
coverage.

In a long-tailed multi-label problem, the marginal per-class guarantee is mostly
carried by negative inclusion: the negative scores set the threshold, so the set
contains 0 for almost every record and rarely contains 1. A class-conditional
(Mondrian) variant is therefore also reported: separate quantiles for calibration
positives (score 1-p) and negatives (score p), which targets 1-alpha coverage for
each label value separately. Positive recall is the number that matters
clinically, so it is printed and plotted explicitly for both variants.

Temperature scaling is monotonic in p, so it cannot change these sets; raw
probabilities are used. The model was early-stopped on val, so val and test are
not strictly exchangeable; that caveat is recorded in the JSON.

Usage: python src/m4_conformal.py [--tag m1_resnet] [--alpha 0.1]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ecg_data import RESULTS, ROOT, build_labels, label_columns, load_meta, load_split, split_rows  # noqa: E402

FIGS = ROOT / "figures"
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def _quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        return 1.0
    k = math.ceil((n + 1) * (1.0 - alpha))
    return 1.0 if k > n else float(np.sort(scores)[k - 1])


def _sets(pc, yc, pt, alpha, names):
    n = len(pc)
    inc1 = np.zeros_like(pt, dtype=bool)
    inc0 = np.zeros_like(pt, dtype=bool)
    per = {}
    for j, nm in enumerate(names):
        s = np.where(yc[:, j] == 1, 1.0 - pc[:, j], pc[:, j])
        q = _quantile(s, alpha)
        inc1[:, j] = pt[:, j] >= 1.0 - q
        inc0[:, j] = pt[:, j] <= q
        pos = yc[:, j] == 1
        per[nm] = {"q": q, "positive_threshold": 1.0 - q,
                   "n_pos_cal": int(pos.sum()), "n_cal": n,
                   "empty_rate": float(((~inc0[:, j]) & (~inc1[:, j])).mean()),
                   "both_rate": float((inc0[:, j] & inc1[:, j]).mean())}
    return inc0, inc1, per


def evaluate(pc, yc, pt, yt, names, alpha):
    C = len(names)
    inc0, inc1, per = _sets(pc, yc, pt, alpha, names)
    cover = np.where(yt == 1, inc1, inc0)
    for j, nm in enumerate(names):
        pos = yt[:, j] == 1
        per[nm].update({
            "coverage": float(cover[:, j].mean()),
            "coverage_pos": float(inc1[pos, j].mean()) if pos.any() else None,
            "coverage_neg": float(inc0[~pos, j].mean()) if (~pos).any() else None,
            "mean_positive_rate": float(inc1[:, j].mean()),
            "prevalence_test": float(pos.mean()),
            "n_pos_test": int(pos.sum()),
        })
    covs = [per[nm]["coverage"] for nm in names]
    empty = sum(per[nm]["empty_rate"] for nm in names)
    both = sum(per[nm]["both_rate"] for nm in names)
    out = {
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "marginal_coverage_mean_per_class": float(np.mean(covs)),
        "marginal_coverage_min_class": float(np.min(covs)),
        "pooled_coverage": float(cover.mean()),
        "joint_coverage_all_classes": float(cover.all(axis=1).mean()),
        "mean_positive_set_size": float(inc1.sum(axis=1).mean()),
        "mean_total_labels": float((inc1.sum(axis=1) + inc0.sum(axis=1)).mean()),
        "mean_empty_sets_per_record": float(empty),
        "mean_both_labels_per_record": float(both),
        "mean_singleton_sets_per_record": float(len(names) - empty - both),
        "record_has_empty_set": float(((~inc0) & (~inc1)).any(axis=1).mean()),
        "n_test": int(len(pt)),
        "per_class": per,
    }
    # Bonferroni: per-class alpha/C targets joint >= 1-alpha
    inc0b, inc1b, _perb = _sets(pc, yc, pt, alpha / C, names)
    coverb = np.where(yt == 1, inc1b, inc0b)
    out["bonferroni"] = {
        "per_class_alpha": alpha / C,
        "joint_coverage_all_classes": float(coverb.all(axis=1).mean()),
        "marginal_coverage_mean_per_class": float(np.mean([coverb[:, j].mean()
                                                           for j in range(C)])),
        "mean_positive_set_size": float(inc1b.sum(axis=1).mean()),
        "mean_total_labels": float((inc1b.sum(axis=1) + inc0b.sum(axis=1)).mean()),
    }
    return out


def _sets_mondrian(pc, yc, pt, alpha, names):
    """Class-conditional (Mondrian) thresholds: separate quantiles for y=1 and y=0.

    Standard per-class split conformal pools positives and negatives, and in a
    long-tailed label the negative scores set the threshold. That gives ~90%
    marginal coverage while rarely including the positive label. Mondrian
    thresholds target 1-alpha coverage for positives and negatives separately.
    """
    inc1 = np.zeros_like(pt, dtype=bool)
    inc0 = np.zeros_like(pt, dtype=bool)
    per = {}
    for j, nm in enumerate(names):
        pos = yc[:, j] == 1
        qp = _quantile(1.0 - pc[pos, j], alpha)
        qn = _quantile(pc[~pos, j], alpha)
        inc1[:, j] = pt[:, j] >= 1.0 - qp
        inc0[:, j] = pt[:, j] <= qn
        per[nm] = {"q_pos": qp, "q_neg": qn, "positive_threshold": 1.0 - qp,
                   "negative_threshold": qn,
                   "n_pos_cal": int(pos.sum()), "n_neg_cal": int((~pos).sum()),
                   "empty_rate": float(((~inc0[:, j]) & (~inc1[:, j])).mean()),
                   "both_rate": float((inc0[:, j] & inc1[:, j]).mean()),
                   "degenerate_positive": bool(qp >= 1.0)}
    return inc0, inc1, per


def evaluate_mondrian(pc, yc, pt, yt, names, alpha):
    inc0, inc1, per = _sets_mondrian(pc, yc, pt, alpha, names)
    cover = np.where(yt == 1, inc1, inc0)
    for j, nm in enumerate(names):
        pos = yt[:, j] == 1
        per[nm].update({
            "coverage": float(cover[:, j].mean()),
            "coverage_pos": float(inc1[pos, j].mean()) if pos.any() else None,
            "coverage_neg": float(inc0[~pos, j].mean()) if (~pos).any() else None,
            "mean_positive_rate": float(inc1[:, j].mean()),
            "prevalence_test": float(pos.mean()),
            "n_pos_test": int(pos.sum()),
        })
    pos_cov = [per[nm]["coverage_pos"] for nm in names
               if per[nm]["coverage_pos"] is not None]
    neg_cov = [per[nm]["coverage_neg"] for nm in names
               if per[nm]["coverage_neg"] is not None]
    empty = sum(per[nm]["empty_rate"] for nm in names)
    both = sum(per[nm]["both_rate"] for nm in names)
    return {
        "alpha": alpha, "nominal_coverage": 1.0 - alpha,
        "positive_coverage_mean_per_class": float(np.mean(pos_cov)),
        "positive_coverage_min_class": float(np.min(pos_cov)),
        "negative_coverage_mean_per_class": float(np.mean(neg_cov)),
        "negative_coverage_min_class": float(np.min(neg_cov)),
        "pooled_coverage": float(cover.mean()),
        "joint_coverage_all_classes": float(cover.all(axis=1).mean()),
        "mean_positive_set_size": float(inc1.sum(axis=1).mean()),
        "mean_total_labels": float((inc1.sum(axis=1) + inc0.sum(axis=1)).mean()),
        "mean_empty_sets_per_record": float(empty),
        "mean_both_labels_per_record": float(both),
        "mean_singleton_sets_per_record": float(len(names) - empty - both),
        "min_pos_cal_for_non_degenerate": math.ceil((1.0 - alpha) / alpha),
        "degenerate_classes": [nm for nm in names
                               if per[nm]["degenerate_positive"]],
        "n_test": int(len(pt)), "per_class": per,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m1_resnet")
    ap.add_argument("--alpha", type=float, default=0.10)
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

    out = {"tag": args.tag, "protocol": "split conformal, calibration=val, test=test",
           "primary_alpha": args.alpha, "alphas_curve": ALPHAS,
           "note_temperature": "temperature scaling is monotonic and does not change "
                               "conformal sets; raw probabilities used",
           "caveat": "the model was early-stopped on val, so val calibration scores are "
                     "not strictly exchangeable with test scores; coverage is empirical",
           "heads": {}}

    for head in ("super", "rhythm"):
        mask = ms if head == "super" else mr
        m_cal, m_te = mask[rows["val"]], mask[rows["test"]]
        pc, yc = z[f"val_{head}"][m_cal], z[f"val_y_{head}"][m_cal].astype(int)
        pt, yt = z[f"test_{head}"][m_te], z[f"test_y_{head}"][m_te].astype(int)
        primary = evaluate(pc, yc, pt, yt, names[head], args.alpha)
        mon = evaluate_mondrian(pc, yc, pt, yt, names[head], args.alpha)
        curve = [evaluate(pc, yc, pt, yt, names[head], a) for a in ALPHAS]
        mon_curve = [evaluate_mondrian(pc, yc, pt, yt, names[head], a) for a in ALPHAS]
        shift = {}
        for j, nm in enumerate(names[head]):
            a, b = yc[:, j] == 1, yt[:, j] == 1
            shift[nm] = {
                "n_pos_cal": int(a.sum()), "n_pos_test": int(b.sum()),
                "mean_p_cal_pos": float(pc[a, j].mean()) if a.any() else None,
                "mean_p_test_pos": float(pt[b, j].mean()) if b.any() else None,
            }
            if a.any() and b.any():
                shift[nm]["delta"] = shift[nm]["mean_p_test_pos"] - shift[nm]["mean_p_cal_pos"]
        out["heads"][head] = {"primary": primary,
                              "mondrian_primary": mon,
                              "positive_confidence_shift": shift,
                              "curve": [{k: v for k, v in c.items()
                                         if k not in ("per_class", "bonferroni")}
                                        for c in curve],
                              "curve_bonferroni": [c["bonferroni"] for c in curve],
                              "mondrian_curve": [{k: v for k, v in c.items()
                                                  if k != "per_class"}
                                                 for c in mon_curve]}
        p, m = primary, mon
        print(f"{head}: alpha={args.alpha} marginal(mean class)={p['marginal_coverage_mean_per_class']:.4f} "
              f"(min class {p['marginal_coverage_min_class']:.4f}) joint={p['joint_coverage_all_classes']:.4f} "
              f"| pos-set {p['mean_positive_set_size']:.2f} labels {p['mean_total_labels']:.2f} "
              f"| Bonferroni joint={p['bonferroni']['joint_coverage_all_classes']:.4f} "
              f"pos-set {p['bonferroni']['mean_positive_set_size']:.2f}", flush=True)
        print(f"  standard: mean positive recall (coverage of y=1) "
              f"{np.mean([v['coverage_pos'] for v in p['per_class'].values() if v['coverage_pos'] is not None]):.4f} "
              f"| per record: empty {p['mean_empty_sets_per_record']:.2f} "
              f"both {p['mean_both_labels_per_record']:.2f} "
              f"singleton {p['mean_singleton_sets_per_record']:.2f} sets", flush=True)
        print(f"  Mondrian: pos-cov mean={m['positive_coverage_mean_per_class']:.4f} "
              f"(min class {m['positive_coverage_min_class']:.4f}) "
              f"neg-cov mean={m['negative_coverage_mean_per_class']:.4f} "
              f"pooled={m['pooled_coverage']:.4f} joint={m['joint_coverage_all_classes']:.4f} "
              f"| pos-set {m['mean_positive_set_size']:.2f} labels {m['mean_total_labels']:.2f} "
              f"| per record: empty {m['mean_empty_sets_per_record']:.2f} "
              f"both {m['mean_both_labels_per_record']:.2f} "
              f"singleton {m['mean_singleton_sets_per_record']:.2f}", flush=True)
        if m["degenerate_classes"]:
            print(f"  Mondrian degenerate (n_pos_cal < {m['min_pos_cal_for_non_degenerate']}, "
                  f"always predicts 1): {', '.join(m['degenerate_classes'])}", flush=True)

    (RESULTS / "m4_conformal.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ax, head in zip(axes, ("super", "rhythm")):
        cur = out["heads"][head]["curve"]
        mcur = out["heads"][head]["mondrian_curve"]
        ax.plot([c["mean_positive_set_size"] for c in cur],
                [c["marginal_coverage_mean_per_class"] for c in cur],
                marker="o", color="#1f77b4", label="marginal, per-class mean")
        ax.plot([c["mean_positive_set_size"] for c in cur],
                [c["joint_coverage_all_classes"] for c in cur],
                marker="s", color="#d62728", label="joint, all classes")
        ax.plot([c["mean_positive_set_size"] for c in mcur],
                [c["positive_coverage_mean_per_class"] for c in mcur],
                marker="^", color="#2ca02c", label="Mondrian, positive recall")
        ax.plot([c["mean_positive_set_size"] for c in mcur],
                [c["negative_coverage_mean_per_class"] for c in mcur],
                marker="v", color="#9467bd", label="Mondrian, negative coverage")
        for c in cur:
            ax.annotate(f"{1-c['alpha']:.2f}", (c["mean_positive_set_size"],
                                                c["marginal_coverage_mean_per_class"]),
                        textcoords="offset points", xytext=(3, 4), fontsize=7)
        ax.axhline(1 - args.alpha, ls="--", lw=0.8, color="gray")
        ax.set_xlabel("mean positive-set size (classes predicted 1)")
        ax.set_ylabel("empirical coverage")
        ax.set_title(f"{head} head")
        ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"M4 conformal coverage vs set size, {args.tag}, calibration=val, test=test")
    fig.tight_layout()
    fig.savefig(FIGS / "m4_coverage_vs_size.png", dpi=150)
    plt.close(fig)

    # per-class coverage at the primary alpha, standard vs Mondrian
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, head in zip(axes, ("super", "rhythm")):
        per = out["heads"][head]["primary"]["per_class"]
        mper = out["heads"][head]["mondrian_primary"]["per_class"]
        nm = list(per)
        idx = np.arange(len(nm))
        std_all = [per[k]["coverage"] for k in nm]
        std_pos = [per[k]["coverage_pos"] if per[k]["coverage_pos"] is not None
                   else np.nan for k in nm]
        mon_pos = [mper[k]["coverage_pos"] if mper[k]["coverage_pos"] is not None
                   else np.nan for k in nm]
        w = 0.27
        ax.bar(idx - w, std_all, w, color="#4c78a8", label="standard, overall")
        ax.bar(idx, std_pos, w, color="#f58518", label="standard, positive recall")
        ax.bar(idx + w, mon_pos, w, color="#2ca02c", label="Mondrian, positive recall")
        for i, k in enumerate(nm):
            ax.text(i, 1.015, f"{per[k]['n_pos_test']}", ha="center", fontsize=5.5)
        ax.axhline(1 - args.alpha, ls="--", lw=0.9, color="#d62728")
        ax.set_xticks(idx)
        ax.set_xticklabels(nm, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.08); ax.set_ylabel("coverage / recall")
        ax.set_title(f"{head}: per-class coverage (top numbers = test positives)")
        ax.grid(axis="y", alpha=0.25)
    h, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, lab, fontsize=8, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.savefig(FIGS / "m4_per_class_coverage.png", dpi=150)
    plt.close(fig)
    print("wrote", RESULTS / "m4_conformal.json", FIGS / "m4_coverage_vs_size.png",
          FIGS / "m4_per_class_coverage.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
