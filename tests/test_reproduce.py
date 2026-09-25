"""Short reproduction checks (CPU, about 6 minutes):

1. training + audit: four short runs (K = 2: adaptive, uniform r = 3, SURF N = 3; K = 3: adaptive) against reference
   values of the original code of the paper (tests/reference_short_runs.json);
2. screening of {4,9} against results/screening_k2.json;
3. results/: the configuration statistics and the numbers quoted in the paper follow from the per-run records.

    python tests/test_reproduce.py        (or: python -m pytest tests)

On the machine that produced the references the runs agree bit for bit; elsewhere floating-point differences of the
BLAS may appear, hence the relative tolerance.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm.analysis import geomean  # noqa: E402
from abm.experiment import run_leg  # noqa: E402
from abm.screening import screen  # noqa: E402

RTOL = 1e-6


def _close(a, b, rtol=RTOL):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= rtol * np.abs(b)))


def test_short_runs():
    ref = json.loads((ROOT / "tests" / "reference_short_runs.json").read_text())["runs"]
    for name, r in ref.items():
        schedule = [(float("inf"), r["cadence"])] if r["cadence"] else None
        with tempfile.TemporaryDirectory() as tmp:
            sm = run_leg(r["K"], r["method"], r["param"], r["seed"], Path(tmp) / name, budget=r["budget"],
                         schedule=schedule, threads=r["threads"])
        exact = sm["ck_grads"] == r["ck_grads"] and sm["audit_gn"] == r["audit_gn"]
        assert sm["segments"] == r["segments"] and sm["rejections"] == r["rejections"], name
        assert _close(sm["ck_grads"], r["ck_grads"]) and _close(sm["audit_gn"], r["audit_gn"]), name
        print(f"  {name}: worst-case GN {sm['audit_gn'][-1]:.6e} "
              f"({'bit-identical' if exact else 'within rtol'} to the reference)", flush=True)


def test_screening_pair_4_9():
    rec = screen((4, 9))
    stored = next(r for r in json.loads((ROOT / "results" / "screening_k2.json").read_text()) if r["digits"] == [4, 9])
    assert _close(rec["c_j"], stored["c_j"], 1e-9) and _close(rec["C_bal"], stored["C_bal"], 1e-9)
    print(f"  screening {{4,9}}: C_bal {rec['C_bal']:.4f} (stored {stored['C_bal']:.4f})", flush=True)


def test_results_consistency():
    for K, quoted in ((2, {"adaptive": 1.00e-3, ("uniform", 60): 6.86e-3, ("surf", 38): 6.27e-3}),
                      (3, {"adaptive": 1.61e-2, ("uniform", 24): 9.04e-2})):
        res = json.loads((ROOT / "results" / f"k{K}.json").read_text())
        for s in res["configs"]:
            runs = [r for r in res["runs"].values() if r["method"] == s["method"] and r["param"] == s["param"]]
            runs.sort(key=lambda r: r["seed"])
            assert np.isclose(geomean([r["marker"]["y"] for r in runs]), s["y_geomean"], rtol=1e-12)
            assert s["n_plateau"] == sum(r["B_run"] is not None for r in runs)
        stats = {(s["method"], s["param"]): s["y_geomean"] for s in res["configs"]}
        for key, value in quoted.items():
            y = res["adaptive_final_geomean"] if key == "adaptive" else stats[key]
            assert f"{y:.2e}" == f"{value:.2e}", (K, key, y)
        print(f"  results/k{K}.json: consistent; quoted values {', '.join(f'{v:.2e}' for v in quoted.values())}", flush=True)


if __name__ == "__main__":
    for t in (test_results_consistency, test_screening_pair_4_9, test_short_runs):
        print(t.__name__, flush=True)
        t()
    print("all checks passed")
