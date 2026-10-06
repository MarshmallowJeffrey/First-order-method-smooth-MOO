#!/usr/bin/env python
"""Certified K = 3 audits of finished legs: the worst-case gradient norm at every checkpoint, by simplicial branch and
bound (abm/certify.py; the global optimization fallback of Appendix A.1), to a relative gap of 1e-3 in GNS*.

    python scripts/certify_k3.py --runs runs/k3 --workers 8
    python scripts/certify_k3.py --runs runs/k3 --update-summary

For every leg folder (summary.json with ck_m, and grams.npz) without certified.json, the first form writes
certified.json: per checkpoint the certified lower and upper bound of GN* (square roots), whether the gap was reached,
the point attaining the lower bound, the splits and the seconds.  Legs are independent; each worker uses one thread.
The second form writes the bounds into summary.json: audit_gn = the larger of the certified lower bound and the earlier
lower bound (both are values of phi at feasible points), audit_gn_upper = the certified upper bound, audit =
"certified", audit_gap, audit_seconds = the seconds of the certified audit; the earlier values are kept as
audit_gn_previous and audit_seconds_previous.  New legs (scripts/run.py) are audited this way directly.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm.certify import certify_k3_prefixes  # noqa: E402
from abm.meter import CERT_GAP, CERT_TIME_LIMIT  # noqa: E402


def certify_leg(args):
    folder, gap, time_limit = args
    d = Path(folder)
    sm = json.loads((d / "summary.json").read_text())
    Q = np.load(d / "grams.npz")["gram_stack"]
    t0 = time.time()
    res = certify_k3_prefixes(Q, sm["ck_m"], gap=gap, time_limit=time_limit)
    out = {"gap": gap, "time_limit": time_limit, "ck_grads": sm["ck_grads"], "ck_m": sm["ck_m"],
           "lower_gn": [float(np.sqrt(max(r["lower"], 0.0))) for r in res],
           "upper_gn": [float(np.sqrt(max(r["upper"], 0.0))) for r in res],
           "certified": [bool(r["certified"]) for r in res], "lam": [r["lam"] for r in res],
           "splits": [int(r["splits"]) for r in res], "seconds": [float(r["seconds"]) for r in res],
           "total_seconds": time.time() - t0}
    tmp = d / "certified.json.tmp"
    tmp.write_text(json.dumps(out))
    tmp.replace(d / "certified.json")
    return d.name, out


def update_summary(d):
    sm = json.loads((d / "summary.json").read_text())
    if sm.get("audit") == "certified":
        return False
    c = json.loads((d / "certified.json").read_text())
    if c["ck_grads"] != sm["ck_grads"]:
        raise ValueError(f"{d}: checkpoints differ")
    prev = [float(v) for v in sm["audit_gn"]]
    sm["audit_gn_previous"] = prev
    sm["audit_gn"] = [max(a, b) for a, b in zip(c["lower_gn"], prev)]
    sm["audit_gn_upper"] = c["upper_gn"]
    sm["audit_seconds_previous"], sm["audit_seconds"] = sm["audit_seconds"], c["total_seconds"]
    sm.update(audit="certified", audit_gap=c["gap"], audit_uncertified=int(sum(not x for x in c["certified"])))
    (d / "summary.json").write_text(json.dumps(sm, indent=1))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="folder of K = 3 leg folders")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gap", type=float, default=CERT_GAP)
    ap.add_argument("--time-limit", type=float, default=CERT_TIME_LIMIT, help="seconds per checkpoint")
    ap.add_argument("--update-summary", action="store_true")
    a = ap.parse_args()
    legs = sorted(d for d in Path(a.runs).iterdir() if (d / "summary.json").exists() and (d / "grams.npz").exists())
    if a.update_summary:
        done = [d for d in legs if (d / "certified.json").exists()]
        n = sum(update_summary(d) for d in done)
        print(f"summary.json updated in {n} of {len(done)} certified legs ({len(legs) - len(done)} legs not certified yet)")
        return
    todo = [d for d in legs if not (d / "certified.json").exists()]
    print(f"{len(legs)} legs, {len(todo)} to certify, {a.workers} workers", flush=True)
    jobs = [(str(d), a.gap, a.time_limit) for d in todo]
    it = map(certify_leg, jobs) if a.workers <= 1 else get_context("spawn").Pool(a.workers).imap_unordered(certify_leg, jobs)
    for name, out in it:
        print(f"  {name}: {out['total_seconds'] / 60:.1f} min, last checkpoint GN* in [{out['lower_gn'][-1]:.5e}, "
              f"{out['upper_gn'][-1]:.5e}], uncertified checkpoints {sum(not x for x in out['certified'])}", flush=True)


if __name__ == "__main__":
    main()
