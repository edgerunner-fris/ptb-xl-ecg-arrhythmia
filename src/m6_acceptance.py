"""M6 acceptance check: verify the project without retraining anything.

This is the "验收" entry point. It answers one question: are the artifacts on
disk consistent with each other and with the protocol, and can the headline
numbers be reproduced from the saved predictions?

What it checks, in order:
  1. environment versions and presence of data caches
  2. raw file counts (21,799 .hea / 21,799 .dat)
  3. cached signal shape (21799, 12, 1000)
  4. split file: sizes, disjoint ecg_ids, no duplicates
  5. patient-level disjointness, and agreement with the M0 report
  6. label counts recomputed from meta.parquet vs results/m0_data_report.json
  7. artifact inventory: expected JSONs, checkpoints, figures
  8. masked test metrics recomputed from each *_pred.npz vs *_masked.json
  9. calibration: AUC invariance and pooled ECE improvement
 10. conformal: per-class coverage recomputed from the stored q values
 11. master table consistency with the per-run masked JSONs
 12. every decimal cited in the docs matches a value in results/

Nothing is trained and nothing is downloaded. Exit code 0 = all checks pass.

Usage: python src/m6_acceptance.py [--skip-raw-count]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from ecg_data import RESULTS, label_columns, load_meta, load_split  # noqa: E402
from ecg_metrics import multilabel_metrics  # noqa: E402

FIGS = ROOT / "figures"
MODELS = ROOT / "models"
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "ptb-xl" / "1.0.3" / "records100"
CKPT_TAGS = ("m1_resnet", "m2_transformer", "m2_transformer_wide",
             "m3_ssl_msm", "m3_ssl_joint", "m5_kd_student")
PRED_TAGS = ("m1_resnet", "m2_transformer", "m2_transformer_wide", "m5_kd_student")
FIGURES = ("m0_label_distribution.png", "m0_lead_quality.png", "m1_resnet_curves.png",
           "m2_transformer_curves.png", "m2_transformer_wide_curves.png",
           "m3_label_fraction.png", "m4_coverage_vs_size.png",
           "m4_per_class_coverage.png", "m4_reliability.png",
           "m5_attention_correct.png", "m5_attention_wrong.png",
           "m5_quantize.png", "m5_saliency_correct.png", "m5_saliency_wrong.png")


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, fn):
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 - a failing check must not abort the rest
            ok, detail = False, f"ERROR {type(e).__name__}: {e}"
        self.rows.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        return ok

    def summary(self):
        bad = [r for r in self.rows if not r[1]]
        print("-" * 70)
        if bad:
            print(f"ACCEPTANCE FAIL: {len(bad)}/{len(self.rows)} checks failed")
            for n, _, d in bad:
                print(f"  - {n}: {d}")
            return 1
        print(f"ACCEPTANCE PASS: {len(self.rows)}/{len(self.rows)} checks passed")
        return 0


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-raw-count", action="store_true",
                    help="skip counting 43k raw files (slower on network drives)")
    args = ap.parse_args()
    R = Report()

    # 1. environment / caches -------------------------------------------------
    def env():
        import matplotlib
        import pandas
        import scipy
        import sklearn
        import torch
        detail = (f"python {sys.version.split()[0]}, torch {torch.__version__}, "
                  f"numpy {np.__version__}, scipy {scipy.__version__}, "
                  f"sklearn {sklearn.__version__}, pandas {pandas.__version__}, "
                  f"matplotlib {matplotlib.__version__}")
        need = ((PROC / "ptbxl_100hz_filt.npy").exists(),
                (PROC / "meta.parquet").exists(),
                (ROOT / "splits" / "main_v1.json").exists())
        return all(need), detail
    R.check("environment and processed caches", env)

    # 2. raw files -------------------------------------------------------------
    def raw():
        if args.skip_raw_count:
            return True, "skipped"
        hea = sum(1 for _ in RAW.rglob("*.hea"))
        dat = sum(1 for _ in RAW.rglob("*.dat"))
        ok = hea == 21799 and dat == 21799
        return ok, f"{hea} .hea / {dat} .dat (expected 21799 each)"
    R.check("raw PTB-XL file count", raw)

    # 3. cached signals --------------------------------------------------------
    def cache():
        mm = np.load(PROC / "ptbxl_100hz_filt.npy", mmap_mode="r")
        ok = mm.shape == (21799, 12, 1000) and mm.dtype == np.float32
        return ok, f"shape {mm.shape}, dtype {mm.dtype}"
    R.check("filtered signal cache", cache)

    # 4. split integrity -------------------------------------------------------
    meta = load_meta()
    split = load_split()
    names = label_columns(meta)

    def split_check():
        ids = {k: [int(e) for e in split[f"{k}_ecg_ids"]]
               for k in ("train", "val", "test")}
        sizes = {k: len(v) for k, v in ids.items()}
        size_ok = sizes == {"train": 17418, "val": 2183, "test": 2198}
        dup_ok = all(len(set(v)) == len(v) for v in ids.values())
        s = {k: set(v) for k, v in ids.items()}
        disj = (not (s["train"] & s["val"]) and not (s["train"] & s["test"])
                and not (s["val"] & s["test"]))
        return size_ok and dup_ok and disj, (
            f"sizes {sizes}, duplicates {not dup_ok}, "
            f"pairwise ecg_id intersections "
            f"{len(s['train'] & s['val'])}/{len(s['train'] & s['test'])}/"
            f"{len(s['val'] & s['test'])}")
    R.check("split sizes and ecg_id disjointness", split_check)

    def patient_check():
        pid = {int(e): int(p) for e, p in zip(meta["ecg_id"], meta["patient_id"])}
        s = {k: {pid[int(e)] for e in split[f"{k}_ecg_ids"]}
             for k in ("train", "val", "test")}
        inter = {"train_val": len(s["train"] & s["val"]),
                 "train_test": len(s["train"] & s["test"]),
                 "val_test": len(s["val"] & s["test"])}
        m0 = json.loads((RESULTS / "m0_data_report.json").read_text(encoding="utf-8"))
        counts = {k: len(v) for k, v in s.items()}
        same = (counts == {k: v["patients"] for k, v in m0["split"]["splits"].items()}
                and inter == m0["split"]["patient_intersection"])
        return (all(v == 0 for v in inter.values()) and same,
                f"patients {counts}, intersections {inter}, matches M0 report: {same}")
    R.check("patient-level disjointness", patient_check)

    # 5. labels vs M0 report ---------------------------------------------------
    def label_check():
        m0 = json.loads((RESULTS / "m0_data_report.json").read_text(encoding="utf-8"))
        idx = {int(e): i for i, e in enumerate(meta["ecg_id"])}
        tr = np.array([idx[int(e)] for e in split["train_ecg_ids"]], dtype=np.int64)
        out = {}
        for head, cols in (("super", names[0]), ("rhythm", names[1])):
            vals = meta.iloc[tr][[f"{'sup' if head == 'super' else 'rhy'}_{c}"
                                   for c in cols]].to_numpy().sum(axis=0)
            got = {c: int(v) for c, v in zip(cols, vals)}
            want = m0["split"]["splits"]["train"][f"{head}_positive"]
            out[head] = (got == want)
            if not out[head]:
                diff = {k: (got[k], want.get(k)) for k in got if got[k] != want.get(k)}
                return False, f"{head} mismatch {diff}"
        return all(out.values()), "train positives recomputed from meta.parquet match M0"
    R.check("label counts vs M0 report", label_check)

    # 6. artifact inventory ----------------------------------------------------
    def inventory():
        need = ["m0_data_report.json", "m1_norm_stats.json", "m1_resnet.json",
                "m1_resnet_masked.json", "m2_transformer.json",
                "m2_transformer_masked.json", "m2_transformer_wide.json",
                "m2_transformer_wide_masked.json", "m3_ssl_msm.json",
                "m3_ssl_joint.json", "m3_seed_check.json", "m3_summary.json",
                "m4_calibration.json", "m4_conformal.json", "m5_attention.json",
                "m5_saliency.json", "m5_kd_student.json", "m5_kd_student_masked.json",
                "m5_quantize.json", "m6_master_table.json", "predict_selfcheck.json"]
        missing = [n for n in need if not (RESULTS / n).exists()]
        ft = sorted(p.name for p in RESULTS.glob("m3_ft_*.json")
                    if not p.name.endswith("_masked.json"))
        ft_masked = sorted(p.name for p in RESULTS.glob("m3_ft_*_masked.json"))
        ft_missing = [t for t in ft if not (RESULTS / f"{t[:-5]}_masked.json").exists()]
        ck_missing = [t for t in CKPT_TAGS if not (MODELS / f"{t}.pt").exists()]
        ft_ck = [t[:-5] for t in ft if "_linear_" not in t]
        ck_missing += [t for t in ft_ck if not (MODELS / f"{t}.pt").exists()]
        fig_missing = [f for f in FIGURES if not (FIGS / f).exists()]
        ok = not (missing or ft_missing or ck_missing or fig_missing) and len(ft) == 29
        return ok, (f"{len(ft)} M3 runs / {len(ft_masked)} masked, "
                    f"missing json {missing or 'none'}, missing ckpt {ck_missing or 'none'}, "
                    f"missing figures {fig_missing or 'none'}")
    R.check("artifact inventory", inventory)

    # 7. recompute masked metrics from predictions -----------------------------
    def rows_for():
        idx = {int(e): i for i, e in enumerate(meta["ecg_id"])}
        return {k: np.array(sorted(idx[int(e)] for e in split[f"{k}_ecg_ids"]),
                            dtype=np.int64)
                for k in ("train", "val", "test")}

    try:
        rows = rows_for()
    except Exception as e:  # noqa: BLE001
        rows = None
        print(f"[FAIL] split_rows: {e}")
    ys, yr = None, None
    if rows is not None:
        ys = meta[[f"sup_{c}" for c in names[0]]].to_numpy(np.float32)
        yr = meta[[f"rhy_{c}" for c in names[1]]].to_numpy(np.float32)
        ms = meta["has_diag_statement"].to_numpy(bool)
        mr = yr.sum(axis=1) > 0

    def predictions():
        r = rows["test"]
        diffs = []
        for tag in PRED_TAGS:
            z = np.load(RESULTS / f"{tag}_pred.npz")
            md = json.loads((RESULTS / f"{tag}_masked.json").read_text(encoding="utf-8"))
            for head, y_all, mask in (("super", ys, ms), ("rhythm", yr, mr)):
                m = mask[r]
                y, p = y_all[r][m], z[f"test_{head}"][m]
                mt = multilabel_metrics(y, p, names[0 if head == "super" else 1])
                st = md["splits"]["test"][head]
                d_auc = abs(mt["macro_auc"] - st["metrics"]["macro_auc"])
                d_f1 = abs(mt["macro_f1_at_0.5"] - st["metrics"]["macro_f1_at_0.5"])
                n_ok = int(m.sum()) == st["n_scored"]
                diffs.append((tag, head, d_auc, d_f1, n_ok))
        bad = [d for d in diffs if d[2] > 1e-9 or d[3] > 1e-9 or not d[4]]
        worst = max(d[2] for d in diffs)
        return not bad, (f"{len(diffs)} head-checks, max |dAUC| {worst:.2e}, "
                         f"mismatches {len(bad)}")
    R.check("masked metrics recomputed from *_pred.npz", predictions)

    # 8. calibration -----------------------------------------------------------
    def calibration():
        cal = json.loads((RESULTS / "m4_calibration.json").read_text(encoding="utf-8"))
        msgs = []
        ok = True
        for head, v in cal["heads"].items():
            inv = v["max_abs_auc_change"] <= 1e-12   # monotone transform: float noise only
            ece_ok = v["ece_pooled_after"] <= v["ece_pooled_before"]
            temp_ok = v["temperature"] > 0
            ok = ok and inv and ece_ok and temp_ok
            msgs.append(f"{head} T={v['temperature']:.4f}, ECE "
                        f"{v['ece_pooled_before']:.4f}->{v['ece_pooled_after']:.4f}, "
                        f"dAUC={v['max_abs_auc_change']:.1e}")
        return ok, "; ".join(msgs)
    R.check("calibration: AUC invariant, pooled ECE improves", calibration)

    # 9. conformal coverage from stored q --------------------------------------
    def conformal():
        conf = json.loads((RESULTS / "m4_conformal.json").read_text(encoding="utf-8"))
        z = np.load(RESULTS / "m1_resnet_pred.npz")
        r = rows["test"]
        worst = 0.0
        ok = True
        msgs = []
        for head, y_all, mask in (("super", ys, ms), ("rhythm", yr, mr)):
            m = mask[r]
            y = y_all[r][m]
            p = z[f"test_{head}"][m]
            v = conf["heads"][head]
            # marginal
            cov = []
            for j, cname in enumerate(names[0 if head == "super" else 1]):
                q = v["primary"]["per_class"][cname]["q"]
                inc1 = p[:, j] >= 1.0 - q
                inc0 = p[:, j] <= q
                cov.append(float(np.where(y[:, j] == 1, inc1, inc0).mean()))
            stored = [v["primary"]["per_class"][c]["coverage"]
                      for c in names[0 if head == "super" else 1]]
            worst = max(worst, max(abs(a - b) for a, b in zip(cov, stored)))
            mean = float(np.mean(cov))
            if (not close(mean, v["primary"]["marginal_coverage_mean_per_class"], 1e-9)
                    or max(abs(a - b) for a, b in zip(cov, stored)) > 1e-9):
                ok = False
            # mondrian
            covm = []
            for j, cname in enumerate(names[0 if head == "super" else 1]):
                d = v["mondrian_primary"]["per_class"][cname]
                inc1 = p[:, j] >= 1.0 - d["q_pos"]
                inc0 = p[:, j] <= d["q_neg"]
                covm.append(float(np.where(y[:, j] == 1, inc1, inc0).mean()))
            storedm = [v["mondrian_primary"]["per_class"][c]["coverage"]
                       for c in names[0 if head == "super" else 1]]
            worst = max(worst, max(abs(a - b) for a, b in zip(covm, storedm)))
            pooled = float(np.mean(covm))
            if not close(pooled, v["mondrian_primary"]["pooled_coverage"], 1e-9):
                ok = False
            msgs.append(f"{head} marginal {mean:.4f}, mondrian {pooled:.4f}")
        return ok and worst < 1e-9, (
            f"{'; '.join(msgs)}, max |recompute-stored| {worst:.2e}")
    R.check("conformal coverage recomputed from stored q", conformal)

    # 10. master table consistency --------------------------------------------
    def master():
        mt = json.loads((RESULTS / "m6_master_table.json").read_text(encoding="utf-8"))
        bad = []
        for tag, row in mt["models"].items():
            md = json.loads((RESULTS / f"{tag}_masked.json").read_text(encoding="utf-8"))
            for head in ("super", "rhythm"):
                a = row["masked"]["test"][head]["macro_auc"]
                b = md["splits"]["test"][head]["metrics"]["macro_auc"]
                if not close(a, b):
                    bad.append(f"{tag}/{head}")
        return not bad, f"{len(mt['models'])} models, mismatches {bad or 'none'}"
    R.check("master table matches per-run masked JSON", master)

    # 11. docs number audit ----------------------------------------------------
    def docs():
        import m6_check_numbers
        rc = m6_check_numbers.main()
        return rc == 0, "all decimal citations matched" if rc == 0 else "unmatched citations"
    R.check("docs numbers traceable to results/", docs)

    return R.summary()


if __name__ == "__main__":
    sys.exit(main())
