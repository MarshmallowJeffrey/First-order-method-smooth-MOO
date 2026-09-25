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


def log_cells(fronts, nbins=18):
    """Edges of a log grid in (F_1, F_2) from just below the smallest value of the given K = 3 fronts up to ln 3."""
    lo = max(1e-3, 0.9 * min(float(F[:, :2].min()) for F in fronts))
    return np.geomspace(lo, LN3, nbins + 1)


def cell_minima(F, edges):
    """Per grid cell in (F_1, F_2): the point of the front F with the smallest F_3.  Returns {cell: point}."""
    nb = len(edges) - 1
    xi = np.clip(np.searchsorted(edges, F[:, 0]) - 1, 0, nb - 1)
    yi = np.clip(np.searchsorted(edges, F[:, 1]) - 1, 0, nb - 1)
    best = {}
    for k in range(F.shape[0]):
        key = (int(xi[k]), int(yi[k]))
        if key not in best or F[k, 2] < F[best[key], 2]:
            best[key] = k
    return {key: F[k] for key, k in best.items()}


def mean_front_3d(fronts, edges):
    """Seed-mean of K = 3 fronts for drawing: in every grid cell where each front has a point, the average of the
    fronts' lowest points (smallest F_3)."""
    cells = [cell_minima(np.asarray(F, float), edges) for F in fronts]
    keys = sorted(set.intersection(*[set(c) for c in cells]))
    return np.array([np.mean([c[k] for c in cells], axis=0) for k in keys])


def dominated_share(A, B, chunk=400):
    """Share of the rows of A that are weakly dominated by some row of B (all <=, one <)."""
    n = 0
    for i in range(0, len(A), chunk):
        Ai = A[i:i + chunk]
        le = (B[None, :, :] <= Ai[:, None, :]).all(axis=2)
        lt = (B[None, :, :] < Ai[:, None, :]).any(axis=2)
        n += int((le & lt).any(axis=1).sum())
    return n / len(A)
