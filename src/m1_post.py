"""Post-hoc robustness metric: macro-AUC restricted to classes with enough positives.

With 2 test positives, a class AUC is nearly a coin flip on ranking two samples,
and macro-AUC weights it the same as SR (1,674 positives). Report both.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def head_metrics(y, p, names, min_pos):
    per = {}
    for j, n in enumerate(names):
        yt = y[:, j]
        auc = float(roc_auc_score(yt, p[:, j])) if len(np.unique(yt)) > 1 else float("nan")
        per[n] = {"auc": auc, "n_pos": int(yt.sum()),
                  "f1": float(f1_score(yt, (p[:, j] >= 0.5).astype(int), zero_division=0))}
    sel = [v["auc"] for v in per.values() if v["n_pos"] >= min_pos and not np.isnan(v["auc"])]
    return {"min_pos": min_pos, "n_classes_included": len(sel),
            "macro_auc": float(np.mean(sel)) if sel else float("nan"), "per_class": per}


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "m1_resnet"
    main_json = json.loads((RESULTS / f"{tag}.json").read_text(encoding="utf-8"))
    names = main_json["classes"]
    z = np.load(RESULTS / f"{tag}_pred.npz")
    out = {"tag": tag, "source": f"{tag}_pred.npz (test split)",
           "thresholds_of_min_positives": [10, 20], "heads": {}}
    for head in ("super", "rhythm"):
        y = z[f"test_y_{head}"]
        p = z[f"test_{head}"]
        out["heads"][head] = {f"min_pos_{m}": head_metrics(y, p, names[head], m)
                              for m in (10, 20)}
        out["heads"][head]["all_classes"] = head_metrics(
            y, p, names[head], 0)["macro_auc"]
    (RESULTS / f"{tag}_stable.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for h in ("super", "rhythm"):
        print(h, "all", round(out["heads"][h]["all_classes"], 4),
              ">=10", round(out["heads"][h]["min_pos_10"]["macro_auc"], 4),
              f"({out['heads'][h]['min_pos_10']['n_classes_included']} classes)",
              ">=20", round(out["heads"][h]["min_pos_20"]["macro_auc"], 4),
              f"({out['heads'][h]['min_pos_20']['n_classes_included']} classes)")


if __name__ == "__main__":
    main()
