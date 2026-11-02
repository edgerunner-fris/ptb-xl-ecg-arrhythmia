"""M0: PTB-XL v1.0.3 data integrity, label taxonomy and split verification.

Reads the official 100 Hz records, builds a float32 memmap cache, and writes
every measured number to results/m0_data_report.json. Nothing here is an
estimate: if a field is missing from the data it is reported as null/0.

Usage:  python src/m0_inspect.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ptb-xl" / "1.0.3"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
SPLITS = ROOT / "splits"
SEED = 42
EXPECTED_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
# PTB-XL headers spell the augmented leads in upper case (AVR/AVL/AVF).
LEAD_INDEX = {nm.upper(): i for i, nm in enumerate(EXPECTED_LEADS)}
FS_EXPECTED, LEN_EXPECTED, NLEAD_EXPECTED = 100, 1000, 12


def jnum(x):
    """JSON-safe number."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    return x


def parse_scp(s):
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        d = ast.literal_eval(s)
        return d if isinstance(d, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def main() -> int:
    t_start = time.time()
    for d in (PROC, RESULTS, FIGS, SPLITS):
        d.mkdir(parents=True, exist_ok=True)
    report: dict = {"seed": SEED, "dataset": "PTB-XL v1.0.3 (records100)"}

    # ---------- metadata ----------
    db = pd.read_csv(RAW / "ptbxl_database.csv")
    scp_raw = pd.read_csv(RAW / "scp_statements.csv")
    first = scp_raw.columns[0]
    scp = (scp_raw.rename(columns={first: "statement"}).set_index("statement")
           if str(first).startswith("Unnamed") else scp_raw.set_index(first))
    print(f"ptbxl_database.csv: {db.shape}, columns={list(db.columns)}")
    print(f"scp_statements.csv: {scp.shape}, columns={list(scp.columns)}")
    report["metadata"] = {
        "db_shape": list(db.shape),
        "db_columns": list(db.columns),
        "scp_shape": list(scp.shape),
        "scp_columns": list(scp.columns),
        "scp_index_unique": bool(scp.index.is_unique),
    }

    # ---------- basic integrity ----------
    files_hea = sorted(RAW.glob("records100/*/*.hea"))
    files_dat = sorted(RAW.glob("records100/*/*.dat"))
    db["hea_path"] = db["filename_lr"].astype(str) + ".hea"
    db["dat_path"] = db["filename_lr"].astype(str) + ".dat"
    miss_hea = [p for p in db["hea_path"] if not (RAW / p).exists()]
    miss_dat = [p for p in db["dat_path"] if not (RAW / p).exists()]
    integ = {
        "csv_rows": int(len(db)),
        "unique_ecg_id": int(db["ecg_id"].nunique()),
        "unique_patient_id": int(db["patient_id"].nunique()),
        "hea_files_on_disk": len(files_hea),
        "dat_files_on_disk": len(files_dat),
        "missing_hea": len(miss_hea),
        "missing_dat": len(miss_dat),
        "records_per_patient": {
            k: jnum(v) for k, v in db.groupby("patient_id").size()
            .describe(percentiles=[.5, .9, .99]).items()
        },
    }
    report["integrity"] = integ
    print(json.dumps(integ, indent=2, ensure_ascii=False, default=jnum))

    # ---------- label taxonomy ----------
    scp = scp.copy()
    for c in ("diagnostic", "form", "rhythm"):
        if c in scp.columns:
            scp[c] = pd.to_numeric(scp[c], errors="coerce").fillna(0).astype(int)
    diag_mask = scp["diagnostic"] == 1
    super_classes = sorted(scp.loc[diag_mask, "diagnostic_class"].dropna().unique())
    sub_classes = sorted(scp.loc[diag_mask, "diagnostic_subclass"].dropna().unique())
    rhythm_classes = sorted(scp.index[scp["rhythm"] == 1])
    form_classes = sorted(scp.index[scp["form"] == 1])

    unknown = Counter()
    stmt_counter = Counter()
    like_counter = Counter()
    rows = []
    for _, r in db.iterrows():
        codes = parse_scp(r["scp_codes"])
        sup, sub, rhy, frm = set(), set(), set(), set()
        has_diag = False
        for s, lik in codes.items():
            stmt_counter[s] += 1
            like_counter[float(lik)] += 1
            if s not in scp.index:
                unknown[s] += 1
                continue
            row = scp.loc[s]
            if row["diagnostic"]:
                has_diag = True
                if isinstance(row.get("diagnostic_class"), str):
                    sup.add(row["diagnostic_class"])
                if isinstance(row.get("diagnostic_subclass"), str):
                    sub.add(row["diagnostic_subclass"])
            if row["rhythm"]:
                rhy.add(s)
            if row["form"]:
                frm.add(s)
        rows.append({
            "ecg_id": int(r["ecg_id"]),
            "patient_id": int(r["patient_id"]),
            "strat_fold": int(r["strat_fold"]),
            "validated_by_human": bool(r["validated_by_human"]),
            "age": jnum(r.get("age", np.nan)),
            "sex": r.get("sex", None),
            "has_diag_statement": has_diag,
            "n_codes": len(codes),
            **{f"sup_{c}": int(c in sup) for c in super_classes},
            **{f"sub_{c}": int(c in sub) for c in sub_classes},
            **{f"rhy_{c}": int(c in rhy) for c in rhythm_classes},
            **{f"frm_{c}": int(c in frm) for c in form_classes},
        })
    lab = pd.DataFrame(rows).set_index("ecg_id")

    like_items = sorted(like_counter.items())
    label_report = {
        "n_super_classes": len(super_classes),
        "super_classes": super_classes,
        "n_sub_classes": len(sub_classes),
        "sub_classes": sub_classes,
        "n_rhythm_classes": len(rhythm_classes),
        "rhythm_classes": rhythm_classes,
        "n_form_classes": len(form_classes),
        "form_classes": form_classes,
        "n_distinct_statements_used": len(stmt_counter),
        "n_unknown_statements": len(unknown),
        "unknown_statements": dict(unknown.most_common(20)),
        "records_with_no_diagnostic_statement": int((~lab["has_diag_statement"]).sum()),
        "records_with_no_rhythm_statement": int(
            (lab[[f"rhy_{c}" for c in rhythm_classes]].sum(axis=1) == 0).sum()),
        "records_with_empty_scp_codes": int((lab["n_codes"] == 0).sum()),
        "likelihood_value_counts": {str(k): v for k, v in like_items[:30]},
        "n_distinct_likelihood_values": len(like_items),
        "mean_labels_per_record": {
            "diagnostic_super": float(lab[[f"sup_{c}" for c in super_classes]].sum(axis=1).mean()),
            "rhythm": float(lab[[f"rhy_{c}" for c in rhythm_classes]].sum(axis=1).mean()),
        },
        "super_class_counts": {c: int(lab[f"sup_{c}"].sum()) for c in super_classes},
        "rhythm_class_counts": {c: int(lab[f"rhy_{c}"].sum()) for c in rhythm_classes},
        "sub_class_counts": {c: int(lab[f"sub_{c}"].sum()) for c in sub_classes},
        "top20_statements": stmt_counter.most_common(20),
    }
    report["labels"] = label_report
    print(f"super={len(super_classes)} sub={len(sub_classes)} "
          f"rhythm={len(rhythm_classes)} form={len(form_classes)}")
    print("super counts:", label_report["super_class_counts"])
    print("rhythm counts:", label_report["rhythm_class_counts"])
    print("records without diagnostic statement:",
          label_report["records_with_no_diagnostic_statement"])

    # ---------- read signals ----------
    n = len(db)
    mm = np.lib.format.open_memmap(
        PROC / "ptbxl_100hz.npy", mode="w+", dtype=np.float32,
        shape=(n, NLEAD_EXPECTED, LEN_EXPECTED))
    mm[:] = np.nan
    lead_orders = Counter()
    fs_bad = 0
    len_bad = 0
    nlead_bad = 0
    read_errors = []
    nan_cnt = np.zeros(NLEAD_EXPECTED, dtype=np.int64)
    zero_cnt = np.zeros(NLEAD_EXPECTED, dtype=np.int64)
    flat_cnt = np.zeros(NLEAD_EXPECTED, dtype=np.int64)
    total_vals = np.zeros(NLEAD_EXPECTED, dtype=np.int64)

    def read_one(idx_path):
        idx, base = idx_path
        rec = wfdb.rdrecord(str(RAW / base), physical=True, return_res=64)
        return idx, rec

    tasks = list(enumerate(db["filename_lr"].astype(str)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(read_one, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="read 100Hz"):
            try:
                idx, rec = fut.result()
            except Exception as e:  # noqa: BLE001
                read_errors.append(str(e))
                continue
            sig = np.asarray(rec.p_signal, dtype=np.float32)
            names = list(rec.sig_name)
            lead_orders[tuple(names)] += 1
            if rec.fs != FS_EXPECTED:
                fs_bad += 1
            if rec.sig_len != LEN_EXPECTED:
                len_bad += 1
            if len(names) != NLEAD_EXPECTED:
                nlead_bad += 1
            out = np.full((NLEAD_EXPECTED, LEN_EXPECTED), np.nan, dtype=np.float32)
            for j, nm in enumerate(names):
                k = LEAD_INDEX.get(str(nm).upper())
                if k is not None and j < sig.shape[1]:
                    v = sig[:, j]
                    out[k, :min(len(v), LEN_EXPECTED)] = v[:LEN_EXPECTED]
            mm[idx] = out

    # incremental per-lead stats
    for s in range(0, n, 500):
        chunk = np.asarray(mm[s:s + 500])
        for k in range(NLEAD_EXPECTED):
            v = chunk[:, k, :]
            isnan = np.isnan(v)
            nan_cnt[k] += int(isnan.sum())
            total_vals[k] += v.size
            finite = v[~isnan]
            if finite.size:
                zero_cnt[k] += int((finite == 0).sum())
                per_rec_std = np.nanstd(v, axis=1)
                flat_cnt[k] += int(np.nansum(per_rec_std < 1e-6))
    mm.flush()

    lead_stats = {
        "lead_order_variants": [{"order": list(o), "count": c}
                                for o, c in lead_orders.most_common(5)],
        "n_lead_order_variants": len(lead_orders),
        "fs_not_100": fs_bad,
        "length_not_1000": len_bad,
        "n_signals_not_12": nlead_bad,
        "read_errors": len(read_errors),
        "read_error_examples": read_errors[:5],
        "per_lead": {},
    }
    for k, nm in enumerate(EXPECTED_LEADS):
        v = np.asarray(mm[:, k, :])
        finite = v[np.isfinite(v)]
        lead_stats["per_lead"][nm] = {
            "nan_rate": float(nan_cnt[k] / max(total_vals[k], 1)),
            "zero_rate_of_finite": float(zero_cnt[k] / max(len(finite), 1)),
            "flat_record_rate": float(flat_cnt[k] / n),
            "p1": float(np.percentile(finite, 1)) if finite.size else None,
            "p50": float(np.percentile(finite, 50)) if finite.size else None,
            "p99": float(np.percentile(finite, 99)) if finite.size else None,
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "unit": "mV",
        }
    report["signals"] = lead_stats
    print("lead order variants:", lead_stats["n_lead_order_variants"],
          lead_stats["lead_order_variants"][:1])
    print("read errors:", len(read_errors), "fs_bad:", fs_bad, "len_bad:", len_bad)

    # ---------- split (official strat_fold 1-8 / 9 / 10) ----------
    lab["strat_fold"] = db.set_index("ecg_id").loc[lab.index, "strat_fold"].astype(int)
    split = {
        "train": lab.index[lab["strat_fold"].between(1, 8)],
        "val": lab.index[lab["strat_fold"] == 9],
        "test": lab.index[lab["strat_fold"] == 10],
    }
    pats = {k: set(lab.loc[v, "patient_id"].astype(int)) for k, v in split.items()}
    fold_pat = lab.groupby("patient_id")["strat_fold"].nunique()
    split_report = {
        "protocol": "official strat_fold: train=1-8, val=9, test=10",
        "patient_intersection": {
            "train_val": len(pats["train"] & pats["val"]),
            "train_test": len(pats["train"] & pats["test"]),
            "val_test": len(pats["val"] & pats["test"]),
        },
        "patients_with_multiple_folds": int((fold_pat > 1).sum()),
        "splits": {},
    }
    for k, ids in split.items():
        sub = lab.loc[ids]
        split_report["splits"][k] = {
            "records": int(len(ids)),
            "patients": int(len(pats[k])),
            "record_fraction": float(len(ids) / n),
            "mean_codes": float(sub["n_codes"].mean()),
            "mean_super_labels": float(
                sub[[f"sup_{c}" for c in super_classes]].sum(axis=1).mean()),
            "validated_by_human_false": int((~sub["validated_by_human"]).sum()),
            "validated_by_human_false_rate": float(
                (~sub["validated_by_human"]).mean()),
            "super_positive": {c: int(sub[f"sup_{c}"].sum()) for c in super_classes},
            "super_prevalence": {c: float(sub[f"sup_{c}"].mean()) for c in super_classes},
            "rhythm_positive": {c: int(sub[f"rhy_{c}"].sum()) for c in rhythm_classes},
        }
    report["split"] = split_report
    print(json.dumps(split_report["patient_intersection"]))
    print("patients with >1 fold:", split_report["patients_with_multiple_folds"])
    for k in split_report["splits"]:
        s = split_report["splits"][k]
        print(f"  {k}: records={s['records']} patients={s['patients']} "
              f"validatedFalse={s['validated_by_human_false']}")

    # ---------- save split artifact ----------
    split_obj = {
        "protocol": "official strat_fold",
        "train_ecg_ids": sorted(int(x) for x in split["train"]),
        "val_ecg_ids": sorted(int(x) for x in split["val"]),
        "test_ecg_ids": sorted(int(x) for x in split["test"]),
        "train_patient_ids": sorted(int(x) for x in pats["train"]),
        "val_patient_ids": sorted(int(x) for x in pats["val"]),
        "test_patient_ids": sorted(int(x) for x in pats["test"]),
    }
    blob = json.dumps(split_obj, sort_keys=True).encode()
    split_obj["md5"] = hashlib.md5(blob).hexdigest()
    (SPLITS / "main_v1.json").write_text(json.dumps(split_obj), encoding="utf-8")

    # ---------- save artifacts ----------
    lab["split"] = np.where(lab["strat_fold"].between(1, 8), "train",
                            np.where(lab["strat_fold"] == 9, "val", "test"))
    lab["age"] = db.set_index("ecg_id").loc[lab.index, "age"]
    lab["sex"] = db.set_index("ecg_id").loc[lab.index, "sex"]
    lab.reset_index().to_parquet(PROC / "meta.parquet", index=False)

    counts = pd.DataFrame({
        "level": (["super"] * len(super_classes) + ["rhythm"] * len(rhythm_classes)
                  + ["sub"] * len(sub_classes)),
        "class": super_classes + rhythm_classes + sub_classes,
        "positive": ([int(lab[f"sup_{c}"].sum()) for c in super_classes]
                     + [int(lab[f"rhy_{c}"].sum()) for c in rhythm_classes]
                     + [int(lab[f"sub_{c}"].sum()) for c in sub_classes]),
    })
    counts["prevalence"] = counts["positive"] / n
    counts.to_csv(RESULTS / "m0_label_counts.csv", index=False)

    report["timing_sec"] = round(time.time() - t_start, 1)
    report["cache"] = str(PROC / "ptbxl_100hz.npy")
    (RESULTS / "m0_data_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=jnum),
        encoding="utf-8")

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    vals = [label_report["super_class_counts"][c] for c in super_classes]
    ax.bar(super_classes, vals)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Diagnostic superclasses (n={n})")
    ax.set_ylabel("records (positive)")
    ax = axes[1]
    vals = [label_report["rhythm_class_counts"][c] for c in rhythm_classes]
    order = np.argsort(vals)[::-1]
    ax.barh([rhythm_classes[i] for i in order], [vals[i] for i in order])
    ax.set_title("Rhythm statements")
    ax.set_xlabel("records")
    ax.invert_yaxis()
    ax = axes[2]
    top = stmt_counter.most_common(20)[::-1]
    ax.barh([t[0] for t in top], [t[1] for t in top])
    ax.set_title("Top-20 SCP statements (all categories)")
    ax.set_xlabel("records")
    fig.tight_layout()
    fig.savefig(FIGS / "m0_label_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    nanr = [lead_stats["per_lead"][c]["nan_rate"] * 100 for c in EXPECTED_LEADS]
    flatr = [lead_stats["per_lead"][c]["flat_record_rate"] * 100 for c in EXPECTED_LEADS]
    x = np.arange(len(EXPECTED_LEADS))
    axes[0].bar(x - 0.2, nanr, 0.4, label="NaN %")
    axes[0].bar(x + 0.2, flatr, 0.4, label="flat-record %")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(EXPECTED_LEADS)
    axes[0].set_title("Per-lead missing / flat rate")
    axes[0].legend()
    p1 = [lead_stats["per_lead"][c]["p1"] for c in EXPECTED_LEADS]
    p50 = [lead_stats["per_lead"][c]["p50"] for c in EXPECTED_LEADS]
    p99 = [lead_stats["per_lead"][c]["p99"] for c in EXPECTED_LEADS]
    axes[1].plot(EXPECTED_LEADS, p50, marker="o", label="p50")
    axes[1].fill_between(EXPECTED_LEADS, p1, p99, alpha=0.25, label="p1-p99")
    axes[1].set_title("Per-lead amplitude (mV)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "m0_lead_quality.png", dpi=150)
    plt.close(fig)

    print(f"done in {report['timing_sec']}s")
    print("SUMMARY " + json.dumps({
        "records": n, "patients": int(db["patient_id"].nunique()),
        "super": len(super_classes), "rhythm": len(rhythm_classes),
        "sub": len(sub_classes),
        "no_diag_stmt": label_report["records_with_no_diagnostic_statement"],
        "read_errors": len(read_errors),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
