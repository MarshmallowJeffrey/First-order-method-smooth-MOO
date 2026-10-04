#!/usr/bin/env python
"""Put the K = 2 adaptive legs of a run folder into results/k2.json and results/k2_fronts.json, keeping every
baseline entry as it is.

    python scripts/update_k2_adaptive.py --runs runs/k2_envelope

Used on 2026-10-04 when the K = 2 adaptive method of the paper changed from the CCP lambda-search to the exact
envelope (config.SELECTOR): only the three adaptive legs were rerun (the baselines choose no lambda).  The adaptive
entries are computed exactly as scripts/analyze.py computes them (same functions); the earlier CCP version of both
files is kept as results/k2_ccp.json and results/k2_fronts_ccp.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from abm import config as C  # noqa: E402
from analyze import analyze, fronts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="folder with adaptive_seed<s>/summary.json and grams.npz")
    a = ap.parse_args()
    runs_dir = Path(a.runs)
    path_res, path_fr = ROOT / "results" / "k2.json", ROOT / "results" / "k2_fronts.json"
    res, fr = json.loads(path_res.read_text()), json.loads(path_fr.read_text())

    new = analyze(2, runs_dir)
    legs = sorted(n for n, r in new["runs"].items() if r["method"] == "adaptive")
    old_legs = sorted(n for n, r in res["runs"].items() if r["method"] == "adaptive")
    if legs != old_legs:
        sys.exit(f"the folder holds {legs}, results/k2.json {old_legs}: not the same legs")
    for n in legs:
        sm = json.loads((runs_dir / n / "summary.json").read_text())
        selector = sm.get("selector", "ccp")
        if selector != C.SELECTOR[2]:
            sys.exit(f"{n}: selector {selector}, but config.SELECTOR[2] = {C.SELECTOR[2]}")
        old = res["runs"][n]
        if (old["budget"], old["ck_grads"], old["segments"]) != (sm["budget"], sm["ck_grads"], sm["segments"]):
            sys.exit(f"{n}: budget, checkpoints or segments differ from the run it replaces")
        if set(new["runs"][n]) != set(old):
            sys.exit(f"{n}: the record fields differ: {sorted(set(new['runs'][n]) ^ set(old))}")
        new["runs"][n]["selector"] = selector

    for n in legs:
        print(f"  {n}: final {res['runs'][n]['final']:.4e} -> {new['runs'][n]['final']:.4e}, "
              f"wall {res['runs'][n]['wall_seconds']:.0f} s -> {new['runs'][n]['wall_seconds']:.0f} s")
        res["runs"][n] = new["runs"][n]
    print(f"  geometric mean {res['adaptive_final_geomean']:.4e} -> {new['adaptive_final_geomean']:.4e}")
    res["adaptive_final"], res["adaptive_final_geomean"] = new["adaptive_final"], new["adaptive_final_geomean"]
    res["adaptive_selector"] = C.SELECTOR[2]

    new_fr = fronts(2, runs_dir)
    for n in legs:
        fr[n] = new_fr[n]
    path_res.write_text(json.dumps(res, indent=1))
    path_fr.write_text(json.dumps(fr))
    print("saved", path_res, "and", path_fr)


if __name__ == "__main__":
    main()
