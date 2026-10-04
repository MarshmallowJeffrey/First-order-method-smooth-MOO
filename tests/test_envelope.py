"""Checks of the exact K = 2 envelope selection (abm/envelope.py; Appendix A.4.1, Algorithms 3-5).

1. random bundles: the envelope equals the pointwise minimum of the parabolas, and the returned maximum is attained
   (the minimum over all parabolas at s* equals it) and is not below any value on a fine grid;
2. merging the new points at every decision gives the same envelope as rebuilding it with ENV;
3. saved K = 2 bundles (runs/warm_start_k2, if present): the exact maximum lies between the grid value and the
   certified upper bound of the audit meter (abm/meter.py), and EnvelopeSelector agrees with a rebuild.

    python tests/test_envelope.py        (or: python -m pytest tests/test_envelope.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm.envelope import EnvelopeSelector, coeffs, envelope, envelope_eval, envelope_max  # noqa: E402
from abm.meter import gn_k2  # noqa: E402


def _random_grams(rng, m, scale):
    J = rng.standard_normal((m, 2, 6)) * np.sqrt(scale)
    return J @ J.transpose(0, 2, 1)


def _brute_min(a, b, c, x):
    return ((a[:, None] * x + b[:, None]) * x + c[:, None]).min(axis=0)


def test_random_bundles():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 20_001)
    for trial in range(300):
        m = int(rng.choice([1, 2, 3, 5, 10, 40, 200]))
        scale = float(10.0 ** rng.uniform(-6, 2))
        a, b, c = coeffs(_random_grams(rng, m, scale))
        s, w = envelope(np.arange(m), a, b, c)
        assert s[0] == 0.0 and s[-1] == 1.0 and np.all(np.diff(s) > 0) and w.size == s.size - 1
        assert w.size <= 2 * m - 1                                   # Davenport-Schinzel bound of the paper
        brute = _brute_min(a, b, c, x)
        err = np.abs(envelope_eval(s, w, a, b, c, x) - brute).max()
        assert err <= 1e-12 * scale * 100, (trial, err)
        v, s_star = envelope_max(s, w, a, b, c)
        attained = _brute_min(a, b, c, np.array([s_star]))[0]
        assert abs(attained - v) <= 1e-12 * max(scale, v), (trial, attained, v)
        assert v >= brute.max() - 1e-12 * scale, (trial, v, brute.max())


def test_incremental_equals_rebuild():
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 1.0, 5_001)
    for trial in range(40):
        m = int(rng.integers(6, 400))
        G = _random_grams(rng, m, float(10.0 ** rng.uniform(-4, 1)))
        sel = EnvelopeSelector()
        for n in [1] + list(range(6, m + 1, 5)) + [m]:              # decisions: theta_0, then 5 points at a time
            v_inc, lam = sel.solve(list(G[:n]))
            a, b, c = coeffs(G[:n])
            s, w = envelope(np.arange(n), a, b, c)
            v_reb, s_reb = envelope_max(s, w, a, b, c)
            assert abs(v_inc - v_reb) <= 1e-12 * max(1.0, v_reb), (trial, n, v_inc, v_reb)
            assert np.allclose(envelope_eval(sel.s, sel.w, a, b, c, x), envelope_eval(s, w, a, b, c, x),
                               rtol=1e-12, atol=1e-15)
            assert abs(lam.sum() - 1.0) < 1e-15 and lam.min() >= 0.0


def test_saved_bundles():
    paths = sorted((ROOT / "runs" / "warm_start_k2").glob("*/grams.npz"))
    if not paths:
        print("  (no saved K = 2 bundles; skipped)")
        return
    for path in paths[:4]:
        G = np.load(path)["gram_stack"]
        sel = EnvelopeSelector()
        for n in (1, 6, 51, 201, 801, G.shape[0]):
            v, lam = sel.solve(list(G[:n]))
            g_val, g_w, g_ub = gn_k2(G[:n], grid_points=200_001)
            a, b, c = coeffs(G[:n])
            tol = 1e-13 * float((np.abs(a) + np.abs(b) + np.abs(c)).max())     # rounding at the coefficient scale
            assert g_val - tol <= v <= g_ub + tol, (path.parent.name, n, g_val, v, g_ub, tol)
        print(f"  {path.parent.name}: {G.shape[0]} points, {sel.pieces} envelope pieces; exact max {v:.10e}, "
              f"audit grid value {g_val:.10e}, certified upper bound {g_ub:.10e}")


if __name__ == "__main__":
    for t in (test_random_bundles, test_incremental_equals_rebuild, test_saved_bundles):
        t()
        print(f"{t.__name__}: ok", flush=True)
