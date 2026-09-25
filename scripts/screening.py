#!/usr/bin/env python
"""Conflict screening of all 45 digit pairs (K = 2) or all 120 digit triples (K = 3), Appendix C.1.

    python scripts/screening.py --K 2
    python scripts/screening.py --K 3

CPU only (about 7 s per pair and 11 s per triple).  Writes results/screening_k<K>.json (one record per candidate,
ranked by C_bal).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm.screening import candidates, ranking, screen  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, choices=(2, 3), required=True)
    a = ap.parse_args()
    records = []
    for digits in candidates(a.K):
        rec = screen(digits)
        records.append(rec)
        print(f"{digits}: C_bal {rec['C_bal']:.4f}  C_mean {rec['C_mean']:.4f}  ({rec['seconds']}s)", flush=True)
    ranked = ranking(records)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"screening_k{a.K}.json").write_text(json.dumps(ranked, indent=1))
    print("most conflicting:", [tuple(r["digits"]) for r in ranked[:6]])


if __name__ == "__main__":
    main()
