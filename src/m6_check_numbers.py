"""M6 audit: every decimal cited in the docs must match a value in results/.

The README, experiment log, Q&A and resume quote measured numbers. Hand-copying
them is exactly where a typo turns a real result into a fabricated one, so this
script re-reads results/*.json, collects every float, and fails loudly if a
decimal in the docs does not coincide with one of them (absolute tolerance half
a unit in the last cited digit).

Scope and limits:
  - decimals with 2-4 places are checked; integers are not (counts like 2,158
    are verified by the generated RESULTS_TABLE.md instead)
  - software versions and the RAM size are skipped via SKIP
  - derived differences are allowed only because src/m6_master_table.py stores
    them under "derived" in results/m6_master_table.json

Usage: python src/m6_check_numbers.py   (exit 1 if a citation is unmatched)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
SKIP = {"1.0.3", "2.9.1", "3.13.12", "1.18.1", "1.9.1", "3.1.0", "3.11.2",
        "4.3.1", "2.5.4", "4.70.1", "2.9", "1.0", "15.64"}  # versions + RAM GB


def collect(obj, acc):
    if isinstance(obj, dict):
        for v in obj.values():
            collect(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            collect(v, acc)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        acc.append(float(obj))


def main() -> int:
    acc = []
    for p in (ROOT / "results").glob("*.json"):
        collect(json.loads(p.read_text(encoding="utf-8")), acc)
    pat = re.compile(r"(?<![\w.])v?(\d+\.\d{2,4})(?![\w.])")
    bad = {}
    n_checked = 0
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for m in pat.finditer(text):
            s = m.group(1)
            if s in SKIP:
                continue
            n_checked += 1
            val = float(s)
            dec = len(s.split(".")[1])
            tol = 0.5 * 10 ** (-dec) * 1.01
            if not any(abs(abs(v) - val) <= tol for v in acc):
                line = text[:m.start()].count("\n") + 1
                bad.setdefault(s, []).append(f"{doc.name}:{line}")
    if bad:
        for k, where in sorted(bad.items()):
            print(f"UNMATCHED {k:>8}  {', '.join(where)}")
        return 1
    print(f"ok: {n_checked} decimal citations matched a results/ value")
    return 0


if __name__ == "__main__":
    sys.exit(main())
