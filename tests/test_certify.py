"""Checks of abm/certify.py (certified K = 3 worst-case gradient norm by simplicial branch and bound), CPU, about 30 s:

1. random bundles: the lower bound is phi at the returned point (all rows), the upper bound is at least the maximum of
   phi on a fine simplex grid, and upper <= (1 + gap) lower;
2. a bundle of one point: lower = upper = its largest diagonal entry (a convex quadratic is maximal at a vertex);
3. the saved K = 3 runs, if present (runs/k3/adaptive_seed41): at the last checkpoint the certified interval contains
   the stored lower bound or lies above it.

    python tests/test_certify.py        (or: python -m pytest tests)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm.certify import certify_k3, phi_at  # noqa: E402
from abm.meter import grid_maxmin_k3  # noqa: E402


def random_bundle(m, d=20, seed=0):
    """Gram matrices of 3 x d Jacobians whose rows share a common direction more and more."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(d)
    out = []
    for i in range(m):
        J = (3.0 / (1.0 + 0.05 * i)) * (rng.uniform(0.5, 1.5, 3)[:, None] * base[None, :] + 0.3 * rng.standard_normal((3, d)))
        out.append(J @ J.T)
    return np.asarray(out)


def test_random_bundles():
    for seed, m in ((1, 40), (2, 300), (3, 1500)):
        Q = random_bundle(m, seed=seed)
        r = certify_k3(Q, gap=1e-3)
        g, _ = grid_maxmin_k3(Q, 600)
        assert r["certified"] and r["upper"] <= (1.0 + 1e-3) * r["lower"]
        assert abs(phi_at(Q, r["lam"]) - r["lower"]) <= 1e-12 * r["lower"]
        assert g <= r["upper"] * (1.0 + 1e-12), (g, r["upper"])
        print(f"  m = {m}: GN* in [{np.sqrt(r['lower']):.6e}, {np.sqrt(r['upper']):.6e}], grid 1/600 {np.sqrt(g):.6e}, "
              f"{r['splits']} splits")


def test_single_point():
    Q = random_bundle(1, seed=4)
    r = certify_k3(Q)
    top = float(np.max(np.diagonal(Q[0])))
    assert abs(r["lower"] - top) <= 1e-12 * top and abs(r["upper"] - top) <= 1e-12 * top
    print("  one point: lower = upper = largest diagonal entry")


def test_saved_run():
    d = ROOT / "runs" / "k3" / "adaptive_seed41"
    if not (d / "grams.npz").exists():
        print("  (no saved K = 3 run; skipped)")
        return
    sm = json.loads((d / "summary.json").read_text())
    m = sm["ck_m"][-1]
    Q = np.load(d / "grams.npz")["gram_stack"][:m]
    r = certify_k3(Q, gap=1e-3)
    stored = float(sm.get("audit_gn_previous", sm["audit_gn"])[-1]) ** 2
    assert r["upper"] >= stored * (1.0 - 1e-12)
    print(f"  adaptive_seed41, last checkpoint: GN* in [{np.sqrt(r['lower']):.6e}, {np.sqrt(r['upper']):.6e}], stored "
          f"lower bound {np.sqrt(stored):.6e}")


if __name__ == "__main__":
    for t in (test_random_bundles, test_single_point, test_saved_run):
        print(t.__name__, flush=True)
        t()
    print("all checks passed")
