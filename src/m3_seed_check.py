"""Seed check for the M3 headline deltas.

Every M3 number so far is a single seed (42). The joint-minus-scratch deltas at
5-25% are only +0.004..+0.005 on the super head, which is inside plausible
seed noise, so 1%, 10% and 100% are re-run with seed 43. 10% is included
because every linear-probe and adapter comparison lives at 10% / 100%, and the
seed-42 curve is non-monotone between 5% and 10% (an independent patient draw
at each fraction, not a nested one).

MSM-only is also re-run at 10% and 100%, because the seed-42 result (MSM-only
full fine-tune > joint SSL at 100%) is the one that contradicts the original
design assumption and therefore needs a second seed before it is reported.

Usage: python src/m3_seed_check.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SRC = ROOT / "src"
RESULTS = ROOT / "results"
ENV = os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1"}
SEED = 43
FRACS = [(0.01, "001"), (0.10, "010"), (1.0, "100")]
MSM_FRACS = {"010", "100"}


def run(script, args):
    r = subprocess.run([str(PY), str(SRC / script), *args], cwd=str(ROOT), env=ENV)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed with {r.returncode}")


def metrics(tag):
    p = RESULTS / f"{tag}_masked.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))["splits"]["test"]
    return {"super_auc": d["super"]["metrics"]["macro_auc"],
            "rhythm_auc": d["rhythm"]["metrics"]["macro_auc"],
            "super_f1": d["super"]["metrics"]["macro_f1_at_0.5"],
            "rhythm_f1": d["rhythm"]["metrics"]["macro_f1_at_0.5"]}


def delta(a, b):
    return {k: a[k] - b[k] for k in a}


def main() -> int:
    for frac, pct in FRACS:
        inits = ["joint", "scratch"] + (["msm"] if pct in MSM_FRACS else [])
        for init in inits:
            tag = f"m3_ft_{init}_full_{pct}_s{SEED}"
            if not (RESULTS / f"{tag}.json").exists():
                print(f"[run ] {tag}", flush=True)
                run("m3_finetune.py",
                    ["--init", init, "--mode", "full", "--fraction", str(frac),
                     "--seed", str(SEED), "--tag", tag])
            if not (RESULTS / f"{tag}_masked.json").exists():
                run("eval_masked.py", [tag])

    out = {"seed_rerun": SEED,
           "note": "fraction<1 draws a different patient subset for this seed; all "
                   "arms share it, so the paired deltas isolate initialisation",
           "points": {}}
    for frac, pct in FRACS:
        row = {}
        for seed in (42, SEED):
            t = (f"m3_ft_{{init}}_full_{pct}" if seed == 42
                 else f"m3_ft_{{init}}_full_{pct}_s{seed}")
            arms = {}
            for init in ("joint", "scratch", "msm"):
                m = metrics(t.format(init=init))
                if m:
                    arms[init] = m
            if "joint" not in arms or "scratch" not in arms:
                continue
            d = {"joint_minus_scratch": delta(arms["joint"], arms["scratch"])}
            if "msm" in arms:
                d["msm_minus_scratch"] = delta(arms["msm"], arms["scratch"])
                d["joint_minus_msm"] = delta(arms["joint"], arms["msm"])
            row[seed] = {"arms": arms, "delta": d}
        out["points"][f"{frac}"] = row

    (RESULTS / "m3_seed_check.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for f, row in out["points"].items():
        for seed, d in row.items():
            jm = d["delta"]["joint_minus_scratch"]
            line = (f"frac {float(f)*100:5.1f}% seed {seed}: "
                    f"joint-scratch sup {jm['super_auc']:+.4f} rhy {jm['rhythm_auc']:+.4f}")
            if "joint_minus_msm" in d["delta"]:
                mj = d["delta"]["joint_minus_msm"]
                line += (f" | joint-msm sup {mj['super_auc']:+.4f} "
                         f"rhy {mj['rhythm_auc']:+.4f}")
            print(line, flush=True)
    print("wrote", RESULTS / "m3_seed_check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
