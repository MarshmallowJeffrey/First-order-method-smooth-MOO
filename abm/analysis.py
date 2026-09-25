"""From audited runs to the numbers of the paper: plateau test, markers, geometric means, trend fits.

g(t) is the audited worst-case gradient norm at the checkpoints t (budget in gradient calls).  The true value is
non-increasing in t (the bundle only grows) and every audit is a lower bound of it (K = 3; for K = 2 a grid value
within its certified upper bound), so a later audit proves every earlier one low: the series is repaired by its
suffix maximum.

Plateau: a run of budget B has plateaued if g(B/4) <= 1.05 g(B/2) and g(B/2) <= 1.05 g(B) (less than 5 % gained in
each of the last two budget doublings); g(L) is the value at the last checkpoint at or before L.  The levels B/8,
B/4, B/2 are tested as well and B_run is the smallest level from which all larger levels pass (None: the run has
not plateaued).  The figures draw only configurations of which at least two of three seeds plateau (the drawn sets
are listed in config.py).

Marker of a run: y = g(B) and x = the first budget with g <= 1.05 y.  For K = 3, x is the first such checkpoint.
For K = 2, x is located to one segment by bisection over the bundle prefixes (exact meter on 20,001 weights), and
y = the exact meter on 200,001 weights of the whole bundle.  A configuration's marker is the geometric mean over its
seeds (x, y and the wall-clock time at x).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from .meter import gn_k2

TOL = 0.05


def suffix_max(values):
    v = np.asarray(values, dtype=float)
    return np.maximum.accumulate(v[::-1])[::-1]


def geomean(values):
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=float)))))


def value_at(x, g, L, slack=1e-2):
    """g at the last checkpoint at or before L (None if the run is shorter than L; a final checkpoint within 1 %
    below L counts as reaching L, since the budget rule leaves at most one segment unspent)."""
    x = np.asarray(x, float)
    if L > x[-1] * (1.0 + slack):
        return None
    j = int(np.searchsorted(x, L * (1.0 + 1e-9), side="right") - 1)
    return float(g[j])


def level_test(x, g, L, tol=TOL):
    gq, gh, gl = value_at(x, g, L / 4.0), value_at(x, g, L / 2.0), value_at(x, g, L)
    if gl is None:
        return None
    return {"L": L, "g_quarter": gq, "g_half": gh, "g_L": gl,
            "pass": bool(gq <= (1.0 + tol) * gh and gh <= (1.0 + tol) * gl)}


def plateau(x, g, budget, tol=TOL):
    """Level tests at B/8, B/4, B/2, B; returns (tests, B_run)."""
    tests = [t for t in (level_test(x, g, L, tol) for L in (budget / 8.0, budget / 4.0, budget / 2.0, budget))
             if t is not None]
    B_run = None
    for i, t in enumerate(tests):
        if all(u["pass"] for u in tests[i:]):
            B_run = t["L"]
            break
    return tests, B_run


def checkpoint_marker(x, wall, g, B, tol=TOL):
    """y = g(B); x = first checkpoint with g <= (1 + tol) y; wall-clock time at that checkpoint."""
    y = value_at(x, g, B)
    if y is None:
        return None
    x = np.asarray(x, float)
    i = int(np.where((np.asarray(g) <= (1.0 + tol) * y) & (x <= B * (1.0 + 1e-9)))[0][0])
    return {"y": y, "x": float(x[i]), "wall": float(wall[i]), "ck_index": i}


def exact_marker_k2(gram_stack, seg_grads, tol=TOL, grid=20_001, final_grid=200_001):
    """K = 2: y = sqrt(max GN) of the whole bundle (final_grid weights); x = budget at the end of the first
    segment m with sqrt(max GN of the first m bundle points) <= (1 + tol) y (grid weights), found by bisection
    (the value is non-increasing in m)."""
    G = np.asarray(gram_stack, dtype=float)
    M = int(G.shape[0])
    y = float(math.sqrt(max(gn_k2(G, grid_points=final_grid)[0], 0.0)))
    memo = {}

    def g(m):
        if m not in memo:
            memo[m] = float(math.sqrt(max(gn_k2(G[:m], grid_points=grid)[0], 0.0)))
        return memo[m]

    target = (1.0 + tol) * y
    if g(M) > target:
        return {"y": y, "x": None, "m": None}
    a, b = 1, M
    if g(1) <= target:
        b = 1
    while b - a > 1:
        mid = (a + b) // 2
        if g(mid) <= target:
            b = mid
        else:
            a = mid
    return {"y": y, "x": float(seg_grads[b - 1]), "m": int(b), "g": g(b)}


def fit_trend(x, y):
    """Descriptive trend f(x) = c + a (x/s)^(-p), s = median of x, least squares on log f(x_i) - log y_i
    (multistart; the best of the converged fits)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    s = float(np.median(x))
    lo, hi = np.array([-30.0, -40.0, 1e-3]), np.array([30.0, 30.0, 10.0])

    def resid(th):
        la, lc, p = th
        return np.log(np.exp(lc) + np.exp(la) * (x / s) ** (-p)) - np.log(y)

    best = None
    ymin, ymed = float(y.min()), float(np.median(y))
    for p0 in (0.3, 0.7, 1.2, 2.0, 3.0):
        for lc0 in (np.log(ymin / 2), np.log(ymin / 10), -20.0):
            for la0 in (np.log(ymed), np.log(max(ymed - np.exp(lc0), 1e-12))):
                th0 = np.clip([la0, lc0, p0], lo, hi)
                try:
                    r = least_squares(resid, th0, bounds=(lo, hi), method="trf", max_nfev=5000)
                except Exception:
                    continue
                if r.success and (best is None or r.cost < best.cost):
                    best = r
    la, lc, p = best.x
    return {"a": float(np.exp(la)), "c": float(np.exp(lc)), "p": float(p), "s": s,
            "rms_log_error": float(np.sqrt(np.mean(best.fun ** 2))), "n_points": int(len(x)),
            "x_min": float(x.min()), "x_max": float(x.max())}


def trend_curve(fit, n=300):
    xs = np.geomspace(fit["x_min"], fit["x_max"], n)
    return xs, fit["c"] + fit["a"] * (xs / fit["s"]) ** (-fit["p"])
