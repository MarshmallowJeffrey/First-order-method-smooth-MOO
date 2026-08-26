"""audit_v2_K6_without_256_checkpoints.py — method-symmetric
two-instrument audits for the K = 6 comparison campaign.

NEW FILE (Aug 9, 2026).  Design: Note/Aug_9_note.md §4 (user-approved
upgrade).  Post-hoc, off both cost axes; no existing file is modified.

For every leg under output/ccp_compare_without_256_checkpoints/K6_B80912:

    audit_v2(stack) = max( strict-64 IPOPT multistart value,
                           heavy CCP value (N0 = 8192, r = 20,
                           single round, fresh solver) )

Both instruments are lower bounds of the true GNS of the stack, so the
max is a tighter lower bound and the ruler is symmetric between the
two methods.  Adaptive legs are audited at every checkpoint prefix;
baseline legs at the final stack only (they are plotted as final
points).

Because checkpoints are recorded right after full decision blocks,
ck_m = 1 + s * d for a completed decision d, so the checkpoint prefixes
double as the gap-curve stacks: for adaptive legs the script also
writes gap data  gap_d = audit_v2(stack_{m_d}) − phi(lambda_d; stack_{m_d})
at the checkpoint-aligned decisions (K6 gap granularity per the Aug-9
Q&A: checkpoint-level, decision-exact stacks).

Output: <leg>/audit_v2.json.

Usage:
    python audit_v2_K6_without_256_checkpoints.py            # full (~15 min)
    python audit_v2_K6_without_256_checkpoints.py --quick    # first 3 stacks/leg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from algorithm_fast_without_256_checkpoints import _maximise_GN_fast
from baseline_svrg_certified_without_256_checkpoints import _GramSet
from ccp_lambda_solver import CCPConfig, CCPLambdaSolver, phi_batch

HERE = Path(__file__).resolve().parent
K6_HOME = (HERE.parent.parent / "output" / "CCP/ccp_compare_without_256_checkpoints"
           / "K6_B80912")


def _instruments(Ms, K, ccp_seed=1):
    gs = _GramSet(list(Ms), K)
    v_ipopt, lam_i = _maximise_GN_fast(gs, prev_lam=None, tier="strict",
                                       max_starts=64)
    solver = CCPLambdaSolver(K, CCPConfig(N0=8192, r=20, seed=ccp_seed,
                                          seed_sampler="exp",
                                          adaptive_seed_schedule=False))
    v_ccp, lam_c = solver.solve(Ms)
    if v_ccp >= v_ipopt:
        return float(v_ipopt), float(v_ccp), float(v_ccp), lam_c
    return float(v_ipopt), float(v_ccp), float(v_ipopt), lam_i


def audit_leg(leg_dir: Path, quick: bool) -> None:
    sm = json.loads((leg_dir / "summary.json").read_text())
    Ms = np.asarray(np.load(leg_dir / "grams.npz")["gram_stack"], float)
    K = Ms.shape[1]
    adaptive = sm["policy"] in ("adaptive", "adaptive_ccp")
    stacks = [m for m in sm["ck_m"] if m >= 1] if adaptive \
        else [Ms.shape[0]]
    if stacks[-1] != Ms.shape[0]:
        stacks.append(Ms.shape[0])
    stacks = sorted(set(stacks))
    if quick:
        stacks = stacks[:3]

    lam_hist = (np.asarray(np.load(leg_dir / "grams.npz")["lam_history"],
                           float) if adaptive else None)
    s_block = int(sm["s"])
    out = {"policy": sm["policy"], "stacks_m": stacks,
           "strict64": [], "ccp8192": [], "v2": [],
           "gap": {"m": [], "decision": [], "phi_lambda": [],
                   "ref_v2": [], "gap": []}}
    t0 = time.time()
    for m in stacks:
        v_i, v_c, v2, _lam = _instruments(Ms[:m], K)
        out["strict64"].append(v_i)
        out["ccp8192"].append(v_c)
        out["v2"].append(v2)
        if adaptive and (m - 1) % s_block == 0:
            # decision d was chosen ON the stack of size 1 + s*d == m
            d = (m - 1) // s_block
            if 0 <= d < len(lam_hist):
                lam_d = lam_hist[d]
                phi_d, _ = phi_batch(Ms[:m], lam_d[None, :])
                out["gap"]["m"].append(int(m))
                out["gap"]["decision"].append(int(d))
                out["gap"]["phi_lambda"].append(float(phi_d[0]))
                out["gap"]["ref_v2"].append(v2)
                out["gap"]["gap"].append(float(v2 - phi_d[0]))
    (leg_dir / "audit_v2.json").write_text(json.dumps(out, indent=2))
    n_ccp_wins = sum(1 for a, b in zip(out["ccp8192"], out["strict64"])
                     if a > b + 1e-12)
    print(f"[audit_v2] {leg_dir.name:20s} stacks={len(stacks)} "
          f"final v2={out['v2'][-1]:.6e} (strict64 {out['strict64'][-1]:.6e}"
          f" / ccp {out['ccp8192'][-1]:.6e}; ccp instrument tighter on "
          f"{n_ccp_wins}/{len(stacks)}) {time.time() - t0:.0f}s",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--home", type=str, default=str(K6_HOME))
    args = ap.parse_args()
    home = Path(args.home)
    legs = sorted(p.parent for p in home.glob("*/summary.json"))
    if not legs:
        raise SystemExit(f"no legs under {home}")
    for leg in legs:
        audit_leg(leg, args.quick)
    print("[audit_v2] done", flush=True)


if __name__ == "__main__":
    main()
