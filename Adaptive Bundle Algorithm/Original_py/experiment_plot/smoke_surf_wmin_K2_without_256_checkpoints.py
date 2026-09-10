"""smoke_surf_wmin_K2_without_256_checkpoints.py — SURF dial-trim smoke
(user request Sep 8 2026).  Question: with the trim tied to the slot
count, w_min = 1/(2N) (half a slot spacing = the boundary resolution of
a uniform grid with r = N), do SURF's slots stay spread over the dial,
or does the vertex arm pull them in (the mu = 0 collapse of Sep 2)?

Runs N in {20, 55}, adam core, mu = 1e-3, B = 2,500, seed 41 (the S3
ladder args), under V2_HOME/surf_wmin_smoke/<core>/, and prints a
readout next to the w_min = 0.05 ladder run of N = 20 (same seed and
budget).  Nothing existing is modified.

Usage:
    python smoke_surf_wmin_K2_without_256_checkpoints.py
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

import _layout  # noqa: F401
from run_surf_compare_K2_without_256_checkpoints import (  # noqa: E402
    CAMPAIGN_MU,
    CORES,
    LADDER_HOME,
    PAIR,
    V2_HOME,
    _ladder_args,
    _load_or_run,
)
from run_stepper_pre_experiment_K2_without_256_checkpoints import (  # noqa: E402
    S2_CFG,
)
from baseline_surf_without_256_checkpoints import run_surf_leg  # noqa: E402

CORE_TAG = "adam_1e-3_b0.9"
SMOKE_NS = (20, 55)
SEED = 41


def readout(run_dir, N, w_min):
    sm = json.loads((run_dir / "summary.json").read_text())
    g = np.load(run_dir / "grams.npz")
    w = np.asarray(g["seg_lams"], dtype=float)[:, 0]
    w = w[np.isfinite(w)]
    last = np.sort(w[-(N + 1):])          # latest slot positions
    gaps = np.diff(last)
    hist = sm["audited_gn_norm_history"]
    return {
        "dir": run_dir.name, "N": N, "w_min": w_min,
        "segments": int(len(w)),
        "complete_rounds": int(len(w) // (N + 1)),
        "final_worst_gn": float(hist[-1]),
        "w_star": float(sm.get("w_star", float("nan"))),
        "slots_min": float(last[0]), "slots_max": float(last[-1]),
        "slots_below_0.1": int((last < 0.1).sum()),
        "slots_above_0.9": int((last > 0.9).sum()),
        "slots_in_0.1_0.9": int(((last >= 0.1) & (last <= 0.9)).sum()),
        "gap_min": float(gaps.min()), "gap_max": float(gaps.max()),
        "gap_median": float(np.median(gaps)),
        "slots_sorted": [round(float(v), 4) for v in last],
    }


def main():
    core = next(c for c in CORES if c[0] == CORE_TAG)
    core_tag, sname, scfg = core
    args = _ladder_args()
    home = V2_HOME / "surf_wmin_smoke" / core_tag
    home.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    for N in SMOKE_NS:
        w_min = 1.0 / (2.0 * N)
        out = home / f"surf_N{N}_wmin{w_min:.4f}_seed{SEED}"
        _load_or_run(out, lambda od, NN=N, wm=w_min: run_surf_leg(
            PAIR, dict(S2_CFG), args, od,
            {"core": core_tag, "smoke": "w_min = 1/(2N)"},
            N=NN, stepper_name=sname, stepper_cfg=scfg,
            sampler_seed=SEED, mu=CAMPAIGN_MU, w_min=wm))
        rows.append(readout(out, N, w_min))
        print(f"[smoke] N={N} w_min={w_min:.4f} done, "
              f"{time.time() - t0:.0f}s elapsed", flush=True)
    ref = LADDER_HOME / core_tag / f"surf_N20_seed{SEED}"
    if (ref / "summary.json").exists():
        rows.append(readout(ref, 20, 0.05))
    (home / "smoke_readout.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"\n{r['dir']}: N={r['N']} w_min={r['w_min']:.4f} "
              f"segments={r['segments']} rounds={r['complete_rounds']}")
        print(f"  final worst GN={r['final_worst_gn']:.4f}  "
              f"w*={r['w_star']:.4f}")
        print(f"  slots: min={r['slots_min']:.4f} max={r['slots_max']:.4f} "
              f"<0.1: {r['slots_below_0.1']}  >0.9: {r['slots_above_0.9']}  "
              f"in [0.1,0.9]: {r['slots_in_0.1_0.9']}")
        print(f"  gaps: min={r['gap_min']:.4f} median={r['gap_median']:.4f} "
              f"max={r['gap_max']:.4f}")
        print(f"  positions: {r['slots_sorted']}")


if __name__ == "__main__":
    main()
