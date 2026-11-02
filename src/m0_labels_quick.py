"""Metadata-only pre-check for M0: label counts and patient-disjoint split.

Runs before the waveforms finish downloading. The authoritative numbers still
come from m0_inspect.py; this is an early failure detector, printed only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from m0_inspect import RAW, parse_scp  # noqa: E402


def main() -> None:
    db = pd.read_csv(RAW / "ptbxl_database.csv")
    scp = pd.read_csv(RAW / "scp_statements.csv", index_col=0)
    scp["diagnostic"] = pd.to_numeric(scp["diagnostic"], errors="coerce").fillna(0).astype(int)
    scp["rhythm"] = pd.to_numeric(scp["rhythm"], errors="coerce").fillna(0).astype(int)

    sup = sorted(scp.loc[scp.diagnostic == 1, "diagnostic_class"].dropna().unique())
    rhy = sorted(scp.index[scp.rhythm == 1])

    rows, stmts, likes, no_diag = [], Counter(), Counter(), 0
    for _, r in db.iterrows():
        codes = parse_scp(r["scp_codes"])
        s, h = set(), set()
        has = False
        for k, v in codes.items():
            stmts[k] += 1
            likes[float(v)] += 1
            if k not in scp.index:
                continue
            if scp.at[k, "diagnostic"]:
                has = True
                c = scp.at[k, "diagnostic_class"]
                if isinstance(c, str):
                    s.add(c)
            if scp.at[k, "rhythm"]:
                h.add(k)
        no_diag += int(not has)
        rows.append({"ecg_id": int(r["ecg_id"]), "patient_id": int(r["patient_id"]),
                     "fold": int(r["strat_fold"]), "human": bool(r["validated_by_human"]),
                     "sup": s, "rhy": h, "ndiag": has})
    lab = pd.DataFrame(rows)

    tr = lab[lab.fold <= 8]
    va = lab[lab.fold == 9]
    te = lab[lab.fold == 10]
    P = {k: set(v.patient_id) for k, v in
         {"train": tr, "val": va, "test": te}.items()}
    multi = lab.groupby("patient_id").fold.nunique()
    out = {
        "records": len(lab),
        "patients": int(lab.patient_id.nunique()),
        "records_per_patient": {
            "min": int(lab.groupby("patient_id").size().min()),
            "p50": float(lab.groupby("patient_id").size().median()),
            "max": int(lab.groupby("patient_id").size().max()),
        },
        "no_diagnostic_statement": no_diag,
        "patient_intersections": {
            "train_val": len(P["train"] & P["val"]),
            "train_test": len(P["train"] & P["test"]),
            "val_test": len(P["val"] & P["test"]),
        },
        "patients_in_multiple_folds": int((multi > 1).sum()),
        "super_counts_all": {c: int(lab.sup.apply(lambda x, c=c: c in x).sum()) for c in sup},
        "rhythm_counts_all": {c: int(lab.rhy.apply(lambda x, c=c: c in x).sum()) for c in rhy},
        "splits": {},
        "likelihood_top": [[str(k), v] for k, v in sorted(likes.items())[:12]],
        "likelihood_values": len(likes),
        "top_statements": stmts.most_common(10),
    }
    for name, part in (("train", tr), ("val", va), ("test", te)):
        out["splits"][name] = {
            "records": len(part),
            "patients": int(part.patient_id.nunique()),
            "human_false": int((~part.human).sum()),
            "super_positive": {c: int(part.sup.apply(lambda x, c=c: c in x).sum()) for c in sup},
            "rhythm_positive": {c: int(part.rhy.apply(lambda x, c=c: c in x).sum()) for c in rhy},
            "mean_codes": round(
                float(db.set_index("ecg_id").loc[part.ecg_id, "scp_codes"]
                      .apply(lambda s: len(parse_scp(s))).mean()), 3),
        }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
