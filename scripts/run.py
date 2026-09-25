#!/usr/bin/env python
"""Run legs of the MNIST experiments (Section 4.1): one leg = one method, one resolution, one seed, B gradient calls.

    python scripts/run.py --K 2 --legs adaptive,uniform:60,surf:38 --seeds 41,42,43 --device cuda
    python scripts/run.py --K 3 --legs uniform:24 --seeds 41

Each leg writes runs/k<K>/<leg>/summary.json (checkpoints, audited worst-case gradient norm, timings) and grams.npz
(Gram matrices, objective values, budget and lambda of every bundle point).  A leg whose summary.json exists is
skipped.  ``--legs all`` runs every leg of the paper (K = 2: 111 legs, K = 3: 54 legs); on one RTX A5000 a K = 2 leg
takes 1 hour (adaptive: 5 hours), a K = 3 leg 1 to 3 hours plus the audits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm import config as C  # noqa: E402
from abm.experiment import leg_name, run_leg  # noqa: E402


def paper_legs(K):
    legs = [("adaptive", None)] + [("uniform", r) for r in C.UNIFORM_R[K]]
    if K == 2:
        legs += [("surf", N) for N in C.SURF_N]
    return legs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, choices=(2, 3), required=True)
    ap.add_argument("--legs", required=True, help="'all' or a comma list of adaptive | uniform:<r> | surf:<N>")
    ap.add_argument("--seeds", default=",".join(str(s) for s in C.SEEDS))
    ap.add_argument("--budget", type=float, default=C.BUDGET)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    ap.add_argument("--out", default=None, help="default: runs/k<K>")
    a = ap.parse_args()
    if a.legs == "all":
        legs = paper_legs(a.K)
    else:
        legs = []
        for item in a.legs.split(","):
            method, _, p = item.strip().partition(":")
            legs.append((method, int(p) if p else None))
    out = Path(a.out) if a.out else ROOT / "runs" / f"k{a.K}"
    for seed in [int(s) for s in a.seeds.split(",")]:
        for method, param in legs:
            d = out / leg_name(method, param, seed)
            if (d / "summary.json").exists():
                print(f"[skip] {d} exists", flush=True)
                continue
            run_leg(a.K, method, param, seed, d, budget=a.budget, device=a.device, threads=a.threads)


if __name__ == "__main__":
    main()
