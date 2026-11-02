"""Self-check for src/predict.py: every supported input format must agree.

It loads one held-out PTB-XL record through each accepted path (WFDB in mV,
WFDB whose header declares uV, NPY, CSV with index/time columns, headerless
CSV), runs the same preprocessing and model, and compares the probabilities.
It also compares the preprocessing with the training cache and the prediction
with the saved test prediction.

Writes results/predict_selfcheck.json. Nothing is trained.

Usage: python src/m6_predict_check.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import predict  # noqa: E402
from ecg_data import RESULTS  # noqa: E402

RECORD = ROOT / "data/raw/ptb-xl/1.0.3/records100/00000/00009_lr"
ECG_ID = 9
TAG = "m1_resnet"


def infer(sig, names, fs, model):
    sig = predict.reorder(sig, names)
    sig, note = predict.to_length_1000(sig, fs)
    x = predict.preprocess(sig)
    with torch.no_grad():
        logit_s, logit_r = model(torch.from_numpy(x).unsqueeze(0))
    return (torch.sigmoid(logit_s)[0].numpy(), torch.sigmoid(logit_r)[0].numpy(),
            x, note)


def main() -> int:
    model, sup, rhy = predict.load_model(TAG, torch.device("cpu"))
    sig0, names0, fs0 = predict.load_wfdb(RECORD)
    ps0, pr0, x0, _ = infer(sig0, names0, fs0, model)

    def record(name, ps, pr):
        return {
            "max_abs_diff_super": float(np.abs(ps - ps0).max()),
            "max_abs_diff_rhythm": float(np.abs(pr - pr0).max()),
            "NORM": float(ps[sup.index("NORM")]),
            "CD": float(ps[sup.index("CD")]),
        }

    formats = {"wfdb_mv": record("wfdb_mv", ps0, pr0)}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        np.save(tmp / "ecg.npy", sig0)
        s, n, f = predict.load_npy(tmp / "ecg.npy")
        ps, pr, _, _ = infer(s, n, fs0, model)
        formats["npy"] = record("npy", ps, pr)

        df = pd.DataFrame(sig0.T, columns=names0)
        df.insert(0, "Time (s)", np.arange(sig0.shape[1]) / fs0)
        df.to_csv(tmp / "ecg.csv", index_label="Unnamed: 0")
        s, n, f = predict.load_csv(tmp / "ecg.csv", no_header=False)
        ps, pr, _, _ = infer(s, n, fs0, model)
        formats["csv_time_index"] = record("csv_time_index", ps, pr)

        df[names0].to_csv(tmp / "ecg_noheader.csv", header=False, index=False)
        s, n, f = predict.load_csv(tmp / "ecg_noheader.csv", no_header=True)
        ps, pr, _, _ = infer(s, n, fs0, model)
        formats["csv_no_header"] = record("csv_no_header", ps, pr)

        import wfdb
        wfdb.wrsamp("ecg_uv", fs=fs0, units=["uV"] * 12, sig_name=names0,
                    p_signal=(sig0.T * 1000).astype(np.float64),
                    fmt=["16"] * 12, write_dir=str(tmp))
        s, n, f = predict.load_wfdb(tmp / "ecg_uv")
        ps, pr, _, _ = infer(s, n, fs0, model)
        formats["wfdb_uv"] = record("wfdb_uv", ps, pr)

    meta = pd.read_parquet(ROOT / "data/processed/meta.parquet")
    row = int(meta.index[meta["ecg_id"] == ECG_ID][0])
    cache = np.load(ROOT / "data/processed/ptbxl_100hz_filt.npy", mmap_mode="r")[row]
    cache_diff = float(np.abs(x0 - cache).max())

    z = np.load(RESULTS / f"{TAG}_pred.npz")
    ids = [int(e) for e in z["test_ecg_ids"]]
    i = ids.index(ECG_ID)
    saved_super = float(np.abs(ps0 - z["test_super"][i]).max())
    saved_rhythm = float(np.abs(pr0 - z["test_rhythm"][i]).max())

    out = {
        "record": {"ecg_id": ECG_ID, "source": str(RECORD), "fs": fs0,
                   "n_samples": int(sig0.shape[1])},
        "model": TAG,
        "raw_sigmoid": True,
        "formats": formats,
        "preprocess_vs_training_cache_max_abs": cache_diff,
        "prediction_vs_saved_test_max_abs": {
            "super": saved_super, "rhythm": saved_rhythm,
        },
        "note": "saved test predictions were produced on GPU with batch 256; "
                "this check runs on CPU with batch 1, so probability differences "
                "are floating point, not a different input",
    }
    (RESULTS / "predict_selfcheck.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"preprocess vs training cache max |d|: {cache_diff:.2e}")
    print(f"prediction vs saved test max |dp|: super {saved_super:.2e}, "
          f"rhythm {saved_rhythm:.2e}")
    for k, v in formats.items():
        print(f"{k:>16}: max |dp| {v['max_abs_diff_super']:.2e} / "
              f"{v['max_abs_diff_rhythm']:.2e}, NORM {v['NORM']:.4f}, CD {v['CD']:.4f}")
    print(f"wrote {RESULTS / 'predict_selfcheck.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
