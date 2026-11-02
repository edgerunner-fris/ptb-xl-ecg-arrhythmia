"""M1 preprocessing: clip, 0.5-40 Hz zero-phase bandpass, train-only per-lead z-score.

Writes data/processed/ptbxl_100hz_filt.npy and results/m1_norm_stats.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from ecg_data import PROC, RESULTS, SPLITS, load_meta, load_split, split_rows  # noqa: E402

FS = 100
BAND = (0.5, 40.0)
ORDER = 4
CLIP_MV = 5.0
CHUNK = 256


def main() -> int:
    t0 = time.time()
    raw = PROC / "ptbxl_100hz.npy"
    out_path = PROC / "ptbxl_100hz_filt.npy"
    src = np.load(raw, mmap_mode="r")
    n, nlead, length = src.shape
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(n, nlead, length))
    sos = signal.butter(ORDER, BAND, btype="bandpass", fs=FS, output="sos")

    meta = load_meta()
    rows = split_rows(meta, load_split())
    train_mask = np.zeros(n, dtype=bool)
    train_mask[rows["train"]] = True

    s = np.zeros(nlead, dtype=np.float64)
    ss = np.zeros(nlead, dtype=np.float64)
    cnt = 0
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        x = np.asarray(src[start:end], dtype=np.float32)
        x = np.clip(x, -CLIP_MV, CLIP_MV)
        y = signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)
        out[start:end] = y
        sel = train_mask[start:end]
        if sel.any():
            z = y[sel].astype(np.float64)
            s += z.sum(axis=(0, 2))
            ss += (z ** 2).sum(axis=(0, 2))
            cnt += z.shape[0] * z.shape[2]
        if (start // CHUNK) % 20 == 0:
            print(f"filter {end}/{n}", flush=True)

    mean = (s / cnt).astype(np.float32)
    std = np.sqrt(ss / cnt - (s / cnt) ** 2).astype(np.float32)
    std = np.maximum(std, 1e-3)

    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        out[start:end] = (out[start:end] - mean[None, :, None]) / std[None, :, None]
    out.flush()

    stats = {
        "fs": FS, "bandpass_hz": list(BAND), "order": ORDER, "clip_mv": CLIP_MV,
        "filter": "scipy.signal.butter + sosfiltfilt (zero phase)",
        "fit_on": "train split only",
        "train_records": int(train_mask.sum()),
        "mean_mv": mean.tolist(), "std_mv": std.tolist(),
        "elapsed_sec": round(time.time() - t0, 1),
        "cache": str(out_path),
    }
    (RESULTS / "m1_norm_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
