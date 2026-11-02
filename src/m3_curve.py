"""Aggregate M3 runs into the label-fraction curve and mode comparison.

Reads results/m3_ft_*.json (+ _stable.json) and writes
results/m3_summary.json and figures/m3_label_fraction.png.
Nothing is estimated: every point is a test-split metric from a saved run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.0]


def pct(f):
    return int(round(f * 100))


def load(tag):
    p = RESULTS / f"{tag}.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    cfg = j.get("config", {})
    subset = cfg.get("subset", {})
    row = {
        "tag": tag,
        "fraction": cfg.get("fraction", cfg.get("label_fraction")),
        "n_train_records": subset.get("n_records"),
        "n_train_patients": subset.get("n_patients"),
        "best_epoch": j.get("best_epoch"),
        "val_score": j["val"]["score"],
        "test_super_auc_unmasked": j["test"]["super"]["macro_auc"],
        "test_rhythm_auc_unmasked": j["test"]["rhythm"]["macro_auc"],
    }
    masked = RESULTS / f"{tag}_masked.json"
    if masked.exists():
        md = json.loads(masked.read_text(encoding="utf-8"))["splits"]["test"]
        row["test_super_auc"] = md["super"]["metrics"]["macro_auc"]
        row["test_super_f1"] = md["super"]["metrics"]["macro_f1_at_0.5"]
        row["test_rhythm_auc"] = md["rhythm"]["metrics"]["macro_auc"]
        row["test_rhythm_f1"] = md["rhythm"]["metrics"]["macro_f1_at_0.5"]
        row["test_super_auc_min10"] = md["super"]["stable_min10"]["macro_auc"]
        row["test_rhythm_auc_min20"] = md["rhythm"]["stable_min20"]["macro_auc"]
        row["metric_scope"] = "masked to records with the relevant statement"
    else:
        row["test_super_auc"] = row["test_super_auc_unmasked"]
        row["test_rhythm_auc"] = row["test_rhythm_auc_unmasked"]
        row["test_super_f1"] = j["test"]["super"]["macro_f1_at_0.5"]
        row["test_rhythm_f1"] = j["test"]["rhythm"]["macro_f1_at_0.5"]
        row["metric_scope"] = "all test records; unlabelled scored as negative"
    return row


def curve(init, mode):
    out = []
    for f in FRACTIONS:
        r = load(f"m3_ft_{init}_{mode}_{pct(f):03d}")
        out.append(r)
    return out


def main() -> int:
    FIGS.mkdir(exist_ok=True)
    summary = {"fractions": FRACTIONS, "curve": {}, "mode_comparison": {},
               "missing": []}

    for init in ("joint", "msm", "scratch"):
        rows = curve(init, "full")
        summary["curve"][f"{init}_full"] = rows
        if init != "msm":                     # msm full is an optional ablation
            summary["missing"] += [f"{init}_full_{pct(f)}" for f, r in zip(FRACTIONS, rows)
                                   if r is None]

    for mode, fracs in (("linear", (0.10, 1.0)), ("adapter", (0.10, 1.0))):
        for init in ("joint", "msm", "scratch"):
            for f in fracs:
                r = load(f"m3_ft_{init}_{mode}_{pct(f):03d}")
                if r is not None:
                    summary["mode_comparison"][f"{init}_{mode}_{pct(f)}"] = r

    # paired delta at each fraction, same subset
    deltas = []
    for f in FRACTIONS:
        a = load(f"m3_ft_joint_full_{pct(f):03d}")
        b = load(f"m3_ft_scratch_full_{pct(f):03d}")
        if a and b:
            deltas.append({
                "fraction": f,
                "super_auc_delta": a["test_super_auc"] - b["test_super_auc"],
                "rhythm_auc_delta": a["test_rhythm_auc"] - b["test_rhythm_auc"],
                "super_f1_delta": a["test_super_f1"] - b["test_super_f1"],
                "rhythm_f1_delta": a["test_rhythm_f1"] - b["test_rhythm_f1"],
            })
    summary["paired_delta_joint_minus_scratch"] = deltas
    summary["notes"] = [
        "All points are test-split metrics; early stopping used the val split only.",
        "Fractions are drawn at patient level; both arms share the same subset file.",
        "Subsets at different fractions are INDEPENDENT patient draws, not nested: the "
        "5% subset is not contained in the 10% subset, so the curve is not monotone by "
        "construction and the 5%->10% dip on the rhythm head is subset sampling, not a "
        "training effect.",
        "Single seed 42; no multi-seed variance is reported.",
        "macro-AUC includes classes with <2 test positives and is therefore noisy; "
        "_min10/_min20 columns restrict to classes with >=10/>=20 positives.",
    ]
    if (RESULTS / "m3_seed_check.json").exists():
        summary["notes"].append(
            "Seed-43 re-runs at the listed fractions are in results/m3_seed_check.json "
            "(paired joint-minus-scratch deltas per seed).")

    (RESULTS / "m3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    xs = [f * 100 for f in FRACTIONS]
    color = {"joint": "#1f77b4", "msm": "#2ca02c", "scratch": "#d62728"}
    label = {"joint": "joint SSL (MSM+CL)", "msm": "MSM only",
             "scratch": "from scratch"}
    for ax, head, key in ((axes[0], "super", "test_super_auc"),
                          (axes[1], "rhythm", "test_rhythm_auc")):
        for init in ("joint", "msm", "scratch"):
            rows = summary["curve"][f"{init}_full"]
            pts = [(f * 100, r[key]) for f, r in zip(FRACTIONS, rows) if r]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                        linestyle="-" if len(pts) >= 3 else "none",
                        color=color[init], label=f"{label[init]} full")
        for init in ("joint", "msm", "scratch"):
            for mode, mk in (("linear", "s"), ("adapter", "^")):
                rows = summary["mode_comparison"]
                pts = [(f * 100, rows[f"{init}_{mode}_{pct(f)}"][key])
                       for f in (0.10, 1.0)
                       if f"{init}_{mode}_{pct(f)}" in rows]
                if pts:
                    ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=mk,
                               s=46, facecolors="none", edgecolors=color[init],
                               label=f"{init} {mode}")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in xs]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("labelled train patients (% of train split)")
        ax.set_ylabel("test macro-AUC")
        ax.set_title(f"{head} head")
        ax.grid(alpha=0.25)
    h, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.16, 1, 0.92))
    fig.legend(h, lab, fontsize=7, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle("M3: label-fraction curve (test split, statement-masked, seed 42)",
                 y=0.99)
    fig.savefig(FIGS / "m3_label_fraction.png", dpi=150)
    plt.close(fig)

    for r in deltas:
        print(f"frac {int(r['fraction'] * 100):3d}%  super Δ {r['super_auc_delta']:+.4f}  "
              f"rhythm Δ {r['rhythm_auc_delta']:+.4f}")
    if summary["missing"]:
        print("missing:", ", ".join(summary["missing"]))
    print("wrote", RESULTS / "m3_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
