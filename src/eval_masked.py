"""Evaluate saved predictions on records that actually carry the label.

The training loss already masks records without a diagnostic statement (super
head) or without any rhythm statement (rhythm head). The original run JSONs did
not apply that mask at evaluation time, so a record with no statement was scored
as an all-negative label vector. This script recomputes test metrics on the
correctly masked subset and writes results/<tag>_masked.json.

It also restricts macro-AUC to classes with >=10 / >=20 test positives, which is
the more stable view under the long tail.

Usage: python src/eval_masked.py <tag>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ecg_data import RESULTS, build_labels, label_columns, load_meta, load_split, split_rows  # noqa: E402
from ecg_metrics import multilabel_metrics  # noqa: E402


def stable(y, p, names, min_pos):
    sel = []
    per = {}
    from sklearn.metrics import roc_auc_score
    for j, n in enumerate(names):
        yj = y[:, j]
        a = float(roc_auc_score(yj, p[:, j])) if len(np.unique(yj)) > 1 else float("nan")
        per[n] = {"auc": a, "n_pos": int(yj.sum())}
        if yj.sum() >= min_pos and not np.isnan(a):
            sel.append(a)
    return {"min_pos": min_pos, "n_classes_included": len(sel),
            "macro_auc": float(np.mean(sel)) if sel else float("nan"),
            "per_class": per}


def main() -> int:
    tag = sys.argv[1]
    j = json.loads((RESULTS / f"{tag}.json").read_text(encoding="utf-8"))
    names = j["classes"]
    z = np.load(RESULTS / f"{tag}_pred.npz")
    meta = load_meta()
    sup, rhy = label_columns(meta)
    ys, yr, ms, mr = build_labels(meta, sup, rhy)
    rows = split_rows(meta, load_split())

    out = {"tag": tag, "source": f"{tag}_pred.npz",
           "note": "metrics restricted to records with the relevant statement; "
                   "the run JSON's headline metrics scored unlabelled records as all-negative",
           "splits": {}}
    for split in ("val", "test"):
        r = rows[split]
        out["splits"][split] = {}
        for head, y_all, mask in (("super", ys, ms), ("rhythm", yr, mr)):
            m = mask[r]
            y, p = y_all[r][m], z[f"{split}_{head}"][m]
            mt = multilabel_metrics(y, p, names[head])
            out["splits"][split][head] = {
                "n_scored": int(m.sum()), "n_excluded_no_statement": int((~m).sum()),
                "metrics": mt,
                "stable_min10": stable(y, p, names[head], 10),
                "stable_min20": stable(y, p, names[head], 20),
            }
    (RESULTS / f"{tag}_masked.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    for split in ("val", "test"):
        parts = []
        for head in ("super", "rhythm"):
            d = out["splits"][split][head]
            raw = j[split][head]["macro_auc"]
            parts.append(f"{head} {d['metrics']['macro_auc']:.4f} (unmasked {raw:.4f}, "
                         f"n={d['n_scored']})")
        print(split, " | ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
