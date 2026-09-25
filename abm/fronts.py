"""Linear scalarization fronts: the non-dominated objective values among all points a run visited."""

from __future__ import annotations

import numpy as np

LN3 = float(np.log(3.0))       # K = 3 fronts are drawn inside [0, ln 3]^3 (ln K = loss of the uniform prediction)


def nondominated_2d(F):
    """Indices of the non-dominated rows of F (n, 2) (minimization), by increasing F[:, 0]."""
    order = np.lexsort((F[:, 1], F[:, 0]))
    keep, best = [], np.inf
    for i in order:
        if F[i, 1] < best - 1e-15:
            keep.append(i)
            best = F[i, 1]
    return np.asarray(keep, dtype=int)


def nondominated_kd(F, block=512):
    """Indices of the non-dominated rows of F (n, K) (minimization; pairwise, in blocks)."""
    F = np.asarray(F, dtype=float)
    keep = np.ones(F.shape[0], dtype=bool)
    for i in range(0, F.shape[0], block):
        blk = F[i:i + block]
        le = (F[None, :, :] <= blk[:, None, :]).all(axis=2)
        lt = (F[None, :, :] < blk[:, None, :]).any(axis=2)
        keep[i:i + block] &= ~(le & lt).any(axis=1)
    return np.nonzero(keep)[0]


def front_2d(fvals):
    """Front of a K = 2 run, sorted by F_1."""
    F = np.asarray(fvals, dtype=float)
    F = F[np.isfinite(F).all(axis=1)]
    fr = F[nondominated_2d(F)]
    return fr[np.argsort(fr[:, 0])]


def mean_front_2d(fronts, window, n_grid=600):
    """Seed-mean of K = 2 fronts: each front interpolated at common F_1 values (the intersection of their ranges,
    cut at ``window``), then F_2 averaged."""
    lo = max(f[:, 0].min() for f in fronts)
    hi = min(min(f[:, 0].max() for f in fronts), window)
    grid = np.linspace(lo, hi, n_grid)
    vals = np.vstack([np.interp(grid, f[:, 0], f[:, 1]) for f in fronts])
    return grid, vals.mean(axis=0)


def envelope_3d(fr, nbins=18):
    """Lower envelope of a K = 3 front for drawing: on a log grid of (F_1, F_2) cells, the point with the smallest
    F_3 in each cell."""
    lo = max(1e-3, 0.9 * float(fr[:, :2].min()))
    edges = np.geomspace(lo, LN3, nbins + 1)
    xi = np.clip(np.searchsorted(edges, fr[:, 0]) - 1, 0, nbins - 1)
    yi = np.clip(np.searchsorted(edges, fr[:, 1]) - 1, 0, nbins - 1)
    best = {}
    for k in range(fr.shape[0]):
        key = (int(xi[k]), int(yi[k]))
        if key not in best or fr[k, 2] < fr[best[key], 2]:
            best[key] = k
    return fr[np.array(sorted(best.values()))]


def dominated_share(A, B, chunk=400):
    """Share of the rows of A that are weakly dominated by some row of B (all <=, one <)."""
    n = 0
    for i in range(0, len(A), chunk):
        Ai = A[i:i + chunk]
        le = (B[None, :, :] <= Ai[:, None, :]).all(axis=2)
        lt = (B[None, :, :] < Ai[:, None, :]).any(axis=2)
        n += int((le & lt).any(axis=1).sum())
    return n / len(A)
