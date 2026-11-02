"""Run every M3 downstream experiment sequentially, skipping finished tags.

Pretraining is NOT run here (see m3_pretrain.py); this driver expects
models/m3_ssl_{msm,joint}.pt to exist.

Each run is a separate process, deliberately: the machine has limited free RAM
and a long-lived process that accumulates CUDA/page-cache state already caused
one crash (M2 wide). Single-threaded BLAS also keeps the memory ceiling lower.

Usage: python src/m3_run_all.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SRC = ROOT / "src"
RESULTS = ROOT / "results"

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.0]
ENV = os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1"}


def tag_of(init, mode, frac):
    return f"m3_ft_{init}_{mode}_{int(round(frac * 100)):03d}"


def run(script, args, tag):
    out = RESULTS / f"{tag}.json"
    if out.exists():
        print(f"[skip] {tag}", flush=True)
    else:
        print(f"[run ] {tag}", flush=True)
        t0 = time.time()
        r = subprocess.run([str(PY), str(SRC / script), *args], cwd=str(ROOT), env=ENV)
        if r.returncode != 0:
            raise SystemExit(f"{tag} failed with {r.returncode}")
        print(f"[done] {tag} {time.time() - t0:.0f}s", flush=True)
    masked = RESULTS / f"{tag}_masked.json"
    if not masked.exists():
        r = subprocess.run([str(PY), str(SRC / "eval_masked.py"), tag],
                           cwd=str(ROOT), env=ENV)
        if r.returncode != 0:
            raise SystemExit(f"masked metrics failed for {tag}")


def main() -> int:
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.environ.update(env)

    for frac in FRACTIONS:
        run("m3_finetune.py",
            ["--init", "joint", "--mode", "full", "--fraction", str(frac)],
            tag_of("joint", "full", frac))
        run("m3_finetune.py",
            ["--init", "scratch", "--mode", "full", "--fraction", str(frac)],
            tag_of("scratch", "full", frac))

    for init, frac in (("joint", 0.10), ("joint", 1.0), ("msm", 0.10)):
        run("m3_finetune.py",
            ["--init", init, "--mode", "adapter", "--fraction", str(frac)],
            tag_of(init, "adapter", frac))

    for init, frac in (("joint", 0.10), ("joint", 1.0), ("msm", 0.10),
                       ("msm", 1.0), ("scratch", 0.10)):
        run("m3_linear_probe.py",
            ["--init", init, "--fraction", str(frac)],
            tag_of(init, "linear", frac))

    print("all M3 downstream runs complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
