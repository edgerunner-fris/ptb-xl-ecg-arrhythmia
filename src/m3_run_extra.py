"""Extra M3 ablation points, run after m3_run_all.py.

Adds three points that the main driver does not cover, so the two design
questions can be answered with measured numbers rather than argument:
  - MSM-only full fine-tune at 10% / 100%  -> does the contrastive term help
    when the whole encoder is trainable, or only for frozen-feature probes?
  - scratch adapter at 10%                 -> adapter capacity on a frozen
    *random* encoder, the control for the pretrained adapter result.

Usage: python src/m3_run_extra.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SRC = ROOT / "src"
RESULTS = ROOT / "results"
ENV = os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1"}


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
    if not (RESULTS / f"{tag}_masked.json").exists():
        r = subprocess.run([str(PY), str(SRC / "eval_masked.py"), tag],
                           cwd=str(ROOT), env=ENV)
        if r.returncode != 0:
            raise SystemExit(f"masked metrics failed for {tag}")


def main() -> int:
    for frac, pct in ((0.10, "010"), (1.0, "100")):
        run("m3_finetune.py",
            ["--init", "msm", "--mode", "full", "--fraction", str(frac)],
            f"m3_ft_msm_full_{pct}")
    run("m3_finetune.py",
        ["--init", "scratch", "--mode", "adapter", "--fraction", "0.10"],
        "m3_ft_scratch_adapter_010")
    r = subprocess.run([str(PY), str(SRC / "m3_curve.py")], cwd=str(ROOT), env=ENV)
    if r.returncode != 0:
        raise SystemExit("m3_curve failed")
    print("extra M3 runs complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
