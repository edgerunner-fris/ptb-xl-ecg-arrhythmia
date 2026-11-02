"""Run the trained ResNet1d on one 12-lead ECG record and print a verdict.

Input is a *digital* 12-lead signal, not an image. Supported formats:
  --wfdb  path to a WFDB record (header .hea + signal .dat); units in the header
          are converted to mV automatically (mV / uV / V)
  --npy   array shaped (12, N) or (N, 12), in mV
  --csv   table with 12 lead columns, in mV; time/index columns are ignored

For --npy/--csv, --fs gives the sampling rate (default 100). A record that is
not 100 Hz is resampled; a record that is not 10 s is center-cropped or
zero-padded to 1000 samples. For non-mV input use --scale (e.g. 0.001 for uV).

The pipeline is exactly the one used for training and evaluation:
  reorder leads to I,II,III,aVR,aVL,aVF,V1-V6
  resample to 100 Hz if needed -> center-crop / zero-pad to 1000 samples
  clip to +/-5 mV -> 0.5-40 Hz 4th-order zero-phase bandpass
  per-lead z-score with the train-only statistics in results/m1_norm_stats.json

Probabilities are temperature-scaled per head by default (M4 calibration);
--raw disables that and reproduces the saved test predictions exactly.

This is a research model, not a diagnostic device. "Normal" means the NORM
superclass is the only one above threshold on this dataset's label scheme; it
does not rule out disease.

Usage:
  python src/predict.py --wfdb data/raw/ptb-xl/1.0.3/records100/00000/00009_lr
  python src/predict.py --npy my_ecg.npy --fs 500 --json out.json
  python src/predict.py --csv my_ecg.csv --fs 100 --scale 0.001   # uV -> mV
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from ecg_data import RESULTS  # noqa: E402
from m1_preprocess import BAND, CLIP_MV, FS, ORDER  # noqa: E402
from m1_train import ResNet1d  # noqa: E402

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_INDEX = {nm.upper(): i for i, nm in enumerate(LEADS)}
ABNORMAL = ("CD", "HYP", "MI", "STTC")


def load_wfdb(path: Path):
    import wfdb
    p = Path(path)
    if p.suffix.lower() in {".hea", ".dat"}:   # accept a path with the extension too
        p = p.with_suffix("")
    rec = wfdb.rdrecord(str(p), physical=True, return_res=64)
    if rec.p_signal is None:
        raise ValueError(f"{path}: no physical signal (check the .hea/.dat pair)")
    names = [str(n) for n in rec.sig_name]
    raw_units = list(rec.units) if rec.units is not None else ["mV"] * len(names)
    factors = []
    for u in raw_units:
        u = str(u).strip().lower().replace("µ", "u")
        if u in {"mv", "millivolt", "millivolts"}:
            factors.append(1.0)
        elif u in {"uv", "microvolt", "microvolts"}:
            factors.append(1e-3)
        elif u in {"v", "volt", "volts"}:
            factors.append(1e3)
        else:
            raise ValueError(f"{path}: unsupported WFDB units {u!r}; convert to mV first")
    sig = np.asarray(rec.p_signal, dtype=np.float64)
    if sig.shape[1] != len(factors):
        raise ValueError(f"{path}: {sig.shape[1]} signals but {len(factors)} units")
    sig = (sig * np.asarray(factors)[None, :]).astype(np.float32).T
    if any(abs(f - 1.0) > 1e-12 for f in factors):
        print(f"[info] converted WFDB units to mV: {raw_units}")
    return sig, names, float(rec.fs)


def _time_columns(cols):
    out = []
    for c in cols:
        s = str(c).strip().lower()
        if s.startswith("unnamed:") or s.startswith("time") or s.startswith("timestamp"):
            out.append(c)
        elif s in {"t", "sec", "second", "seconds", "s", "sample", "sample#", "index"}:
            out.append(c)
    return out


def load_csv(path: Path, no_header: bool):
    df = pd.read_csv(path, header=None if no_header else 0)
    drop = _time_columns(df.columns)
    if drop:
        df = df.drop(columns=drop)
    upper = {str(c).upper(): c for c in df.columns}
    if all(nm.upper() in upper for nm in LEADS):
        names = LEADS
        sig = df[[upper[nm.upper()] for nm in LEADS]].to_numpy(np.float32).T
    elif df.shape[1] == 12:
        names = LEADS
        sig = df.to_numpy(np.float32).T
        print("[warn] CSV has no lead names; assuming order " + ",".join(LEADS))
    else:
        raise ValueError(f"{path}: expected 12 lead columns, found {df.shape[1]}")
    return sig, names, None


def load_npy(path: Path):
    a = np.load(path)
    if a.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D array, got shape {a.shape}")
    if a.shape[0] == 12:
        sig, names = a, LEADS
    elif a.shape[1] == 12:
        sig, names = a.T, LEADS
    else:
        raise ValueError(f"{path}: one axis must be 12 leads, got {a.shape}")
    print("[warn] NPY has no lead names; assuming order " + ",".join(LEADS))
    return sig.astype(np.float32), names, None


def reorder(sig: np.ndarray, names) -> np.ndarray:
    """Put leads in the training order; reject missing or duplicated leads."""
    seen = {}
    for j, nm in enumerate(names):
        key = str(nm).strip().upper()
        if key in LEAD_INDEX:
            if LEAD_INDEX[key] in seen:
                raise ValueError(f"lead {nm} appears twice")
            seen[LEAD_INDEX[key]] = j
    missing = [LEADS[i] for i in range(12) if i not in seen]
    if missing:
        raise ValueError(f"missing leads: {missing}; got {list(names)}")
    return sig[[seen[i] for i in range(12)], :]


def to_length_1000(x: np.ndarray, fs: float) -> tuple[np.ndarray, str]:
    """Resample to 100 Hz, then center-crop or zero-pad to exactly 10 s."""
    note = ""
    if abs(fs - FS) > 1e-9:
        frac = Fraction(FS / fs).limit_denominator(1000)
        x = signal.resample_poly(x, frac.numerator, frac.denominator, axis=-1)
        note += f"resampled {fs:g} -> {FS} Hz; "
    x = x.astype(np.float32)
    n = x.shape[1]
    if n > 1000:
        start = (n - 1000) // 2
        x = x[:, start:start + 1000]
        note += f"center-cropped {n} -> 1000 samples; "
    elif n < 1000:
        x = np.pad(x, ((0, 0), (0, 1000 - n)))
        note += f"zero-padded {n} -> 1000 samples; "
    return x, note.strip("; ")


def preprocess(x: np.ndarray) -> np.ndarray:
    stats = json.loads((RESULTS / "m1_norm_stats.json").read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean_mv"], dtype=np.float32)
    std = np.asarray(stats["std_mv"], dtype=np.float32)
    x = np.clip(x, -CLIP_MV, CLIP_MV).astype(np.float32)
    sos = signal.butter(ORDER, BAND, btype="bandpass", fs=FS, output="sos")
    y = signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)
    return (y - mean[:, None]) / std[:, None]


def temperatures():
    p = RESULTS / "m4_calibration.json"
    if not p.exists():
        return {"super": 1.0, "rhythm": 1.0}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {h: float(v["temperature"]) for h, v in d["heads"].items()}


def load_model(tag: str, device: torch.device):
    meta = json.loads((RESULTS / f"{tag}.json").read_text(encoding="utf-8"))
    if meta["config"].get("model") != "ResNet1d":
        raise ValueError(f"{tag} is {meta['config'].get('model')}, this script only runs ResNet1d")
    sup, rhy = meta["classes"]["super"], meta["classes"]["rhythm"]
    model = ResNet1d(len(sup), len(rhy)).to(device)
    state = torch.load(ROOT / "models" / f"{tag}.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, sup, rhy


def main() -> int:
    ap = argparse.ArgumentParser(description="12-lead ECG inference (digital signal only)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wfdb", type=Path, help="WFDB record path without extension")
    src.add_argument("--npy", type=Path, help=".npy array (12,N) or (N,12), mV")
    src.add_argument("--csv", type=Path, help=".csv with 12 lead columns, mV")
    ap.add_argument("--fs", type=float, default=100.0,
                    help="sampling rate of --npy/--csv input (default 100; WFDB uses its header)")
    ap.add_argument("--no-header", action="store_true", help="CSV has no header row")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply input by this to get mV (e.g. 0.001 for uV); "
                         "WFDB units are already converted, so keep 1 for WFDB")
    ap.add_argument("--model", default="m1_resnet", help="tag under models/ (ResNet1d only)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability threshold for the normal/abnormal verdict")
    ap.add_argument("--raw", action="store_true",
                    help="skip temperature scaling (reproduces *_pred.npz)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--json", type=Path, help="write the result as JSON")
    args = ap.parse_args()

    if args.wfdb:
        sig, names, fs = load_wfdb(args.wfdb)
        source = str(args.wfdb)
        if args.scale != 1.0:
            print("[warn] WFDB units were converted to mV; --scale multiplies again")
    elif args.npy:
        sig, names, fs = load_npy(args.npy)
        fs = args.fs
        source = str(args.npy)
    else:
        sig, names, fs = load_csv(args.csv, args.no_header)
        fs = args.fs
        source = str(args.csv)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"invalid sampling rate: {fs}")

    sig = reorder(sig, names)
    duration = sig.shape[1] / fs
    if not (5.0 <= duration <= 15.0):
        print(f"[warn] input duration {duration:.1f} s at fs={fs:g}; expected about 10 s. "
              "Check --fs.")
    peak = float(np.max(np.abs(sig))) if sig.size else 0.0
    if args.scale == 1.0 and peak > 20.0:
        print(f"[warn] max |value| is {peak:.1f}; input may be uV or ADC counts. "
              "Use --scale (1e-3 for uV).")
    sig = sig * args.scale
    if not np.isfinite(sig).all():
        bad = np.where(~np.isfinite(sig).any(axis=1))[0]
        raise ValueError(f"non-finite samples in leads {[LEADS[i] for i in bad]}")
    flat = [LEADS[i] for i in range(12) if float(np.std(sig[i])) < 1e-6]
    if flat:
        print(f"[warn] flat leads: {flat}")

    sig, note = to_length_1000(sig, fs)
    x = preprocess(sig)
    if note:
        print(f"[info] {note}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    model, sup, rhy = load_model(args.model, device)
    t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        logit_s, logit_r = model(t)
    temp = {"super": 1.0, "rhythm": 1.0} if args.raw else temperatures()
    p_sup = torch.sigmoid(logit_s / temp["super"]).cpu().numpy()[0]
    p_rhy = torch.sigmoid(logit_r / temp["rhythm"]).cpu().numpy()[0]

    print(f"\ninput: {source} | leads {LEADS} | 100 Hz x 1000 samples | "
          f"temperature super={temp['super']:.4f} rhythm={temp['rhythm']:.4f}")
    for head, names_h, p in (("diagnostic superclass", sup, p_sup),
                             ("rhythm", rhy, p_rhy)):
        print(f"\n{head}:")
        for i in np.argsort(p)[::-1]:
            print(f"  {names_h[i]:>6}  {p[i]:.4f}")

    abn = {c: float(p_sup[sup.index(c)]) for c in ABNORMAL if c in sup}
    norm = float(p_sup[sup.index("NORM")]) if "NORM" in sup else float("nan")
    top_abn = max(abn, key=abn.get)
    if abn[top_abn] >= args.threshold:
        verdict = f"abnormal: {top_abn} p={abn[top_abn]:.3f}"
    elif norm >= args.threshold:
        verdict = (f"normal: NORM p={norm:.3f}, highest abnormal {top_abn} "
                   f"p={abn[top_abn]:.3f}")
    else:
        verdict = (f"uncertain: NORM p={norm:.3f} and highest abnormal {top_abn} "
                   f"p={abn[top_abn]:.3f} are both < {args.threshold}")
    sr = float(p_rhy[rhy.index("SR")]) if "SR" in rhy else float("nan")
    print(f"\nverdict: {verdict}")
    print(f"rhythm: SR p={sr:.3f}; other rhythms >= {args.threshold}: "
          f"{[c for c in rhy if c != 'SR' and p_rhy[rhy.index(c)] >= args.threshold] or 'none'}")
    print("\nResearch model on PTB-XL. Not a medical device; do not use for diagnosis.")

    if args.json:
        out = {
            "input": source, "model": args.model, "leads": LEADS,
            "fs": FS, "n_samples": 1000, "scale": args.scale,
            "temperature": temp, "threshold": args.threshold, "verdict": verdict,
            "super": {c: float(p_sup[i]) for i, c in enumerate(sup)},
            "rhythm": {c: float(p_rhy[i]) for i, c in enumerate(rhy)},
        }
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
