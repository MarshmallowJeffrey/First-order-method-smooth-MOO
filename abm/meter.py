"""The worst-case gradient norm of a bundle B = {theta_1, ..., theta_m}: max over lam in Delta_K of min_i ||J_i^T lam||,
J_i the Jacobian at theta_i, computed from the Gram matrices M_i = J_i J_i^T.  The functions below work with the
squared norm min_i lam^T M_i lam; the runs record its square root.

K = 2 (exact).  With lam = (w, 1 - w) every q_i(w) = lam^T M_i lam is a convex quadratic, so GN(w) is the lower
envelope of m parabolas.  The envelope is evaluated exactly on a uniform grid of w, the best grid cell is polished
in closed form (the maximum of an envelope of convex quadratics is at an end point or at a crossing of two of them),
and a certified upper bound follows from the slopes of the active quadratics on every cell.

K = 3 (certified).  At every checkpoint the simplicial branch and bound of abm/certify.py (the global optimization
fallback of Appendix A.1) returns an interval [lower, upper] that contains the squared value, with
upper <= (1 + CERT_GAP) lower.
"""

from __future__ import annotations

import numpy as np

from .certify import certify_k3_prefixes
from .grid import simplex_grid

# ---------------------------------------------------------------------------------------------------------------
#  K = 2: exact
# ---------------------------------------------------------------------------------------------------------------
SEG_CHUNK = 32          # bundle points per envelope update of the incremental audit


def quad_coeffs(Ms):
    """q_i(w) = A_i w^2 + B_i w + C_i for lam = (w, 1 - w)."""
    Ms = np.asarray(Ms, dtype=float)
    a, b, c = Ms[:, 0, 0], Ms[:, 0, 1], Ms[:, 1, 1]
    return a - 2.0 * b + c, 2.0 * (b - c), c


def envelope_at(A, B, C, w):
    w = np.atleast_1d(np.asarray(w, dtype=float))
    Q = A[:, None] * w[None, :] ** 2 + B[:, None] * w[None, :] + C[:, None]
    return Q.min(axis=0)


def _polish_and_bound(A, B, C, grid, env, act):
    """From the grid envelope (env, argmin act) to (value, w*, certified upper bound)."""
    G = grid.size
    jbest = int(env.argmax())
    best_v, best_w = float(env[jbest]), float(grid[jbest])
    h = (float(grid[-1]) - float(grid[0])) / (G - 1)
    wl, wr = max(float(grid[0]), best_w - h), min(float(grid[-1]), best_w + h)
    cand = set()                                    # locally lowest quadratics (top 64 at three points)
    for w0 in (wl, best_w, wr):
        q = A * w0 * w0 + B * w0 + C
        cand.update(np.argsort(q)[:64].tolist())
    cand = np.asarray(sorted(cand), dtype=int)
    ws = [wl, wr, best_w]                           # their pairwise crossings inside [wl, wr]
    Ac, Bc, Cc = A[cand], B[cand], C[cand]
    for i in range(cand.size):
        dA = Ac[i] - Ac[i + 1:]
        dB = Bc[i] - Bc[i + 1:]
        dC = Cc[i] - Cc[i + 1:]
        with np.errstate(all="ignore"):
            disc = dB * dB - 4.0 * dA * dC
            ok = (np.abs(dA) > 1e-300) & (disc >= 0.0)
            sq = np.sqrt(np.where(ok, disc, 0.0))
            for sgn in (+1.0, -1.0):
                r = np.where(ok, (-dB + sgn * sq) / (2.0 * dA), np.nan)
                r = r[(r >= wl) & (r <= wr)]
                ws.extend(float(t) for t in r)
            lin = (~ok) & (np.abs(dB) > 1e-300)
            r = np.where(lin, -dC / np.where(lin, dB, 1.0), np.nan)
            r = r[(r >= wl) & (r <= wr)]
            ws.extend(float(t) for t in r)
    ws = np.asarray(ws, dtype=float)
    env_ws = envelope_at(A, B, C, ws)
    j = int(np.argmax(env_ws))
    if float(env_ws[j]) > best_v:
        best_v, best_w = float(env_ws[j]), float(ws[j])
    # upper bound: on a cell the envelope is below the quadratic active at either end, which moves by at most
    # (largest slope) x h across the cell
    Ai, Bi = A[act], B[act]
    s_here = 2.0 * Ai * grid + Bi
    s_l = np.maximum(np.abs(s_here[:-1]), np.abs(2.0 * Ai[:-1] * grid[1:] + Bi[:-1]))
    s_r = np.maximum(np.abs(s_here[1:]), np.abs(2.0 * Ai[1:] * grid[:-1] + Bi[1:]))
    u_cell = np.minimum(env[:-1] + s_l * h, env[1:] + s_r * h)
    ub = float(max(float(u_cell.max()), float(env[0]), float(env[-1]), best_v))
    return best_v, best_w, ub


def gn_k2(Ms, grid_points=200_001, chunk=250):
    """max over w of min_i lam(w)^T M_i lam(w) for the bundle Ms: (value, w*, certified upper bound)."""
    A, B, C = quad_coeffs(Ms)
    grid = np.linspace(0.0, 1.0, int(grid_points))
    env = np.empty(grid.size)
    act = np.empty(grid.size, dtype=np.int64)
    for lo in range(0, grid.size, int(chunk)):
        w = grid[lo:lo + int(chunk)]
        Q = A[:, None] * w[None, :] ** 2 + B[:, None] * w[None, :] + C[:, None]
        env[lo:lo + w.size] = Q.min(axis=0)
        act[lo:lo + w.size] = Q.argmin(axis=0)
    return _polish_and_bound(A, B, C, grid, env, act)


def gn_k2_prefixes(Ms, ck_m, grid_points=200_001):
    """gn_k2(Ms[:m]) for every m in ck_m (non-decreasing), with one running envelope over the bundle."""
    A, B, C = quad_coeffs(Ms)
    grid = np.linspace(0.0, 1.0, int(grid_points))
    env = np.full(grid.size, np.inf)
    act = np.zeros(grid.size, dtype=np.int64)
    out, pos = [], 0
    for m in ck_m:
        m = int(m)
        while pos < m:
            s1 = min(m, pos + SEG_CHUNK)
            Q = A[pos:s1, None] * grid[None, :] ** 2 + B[pos:s1, None] * grid[None, :] + C[pos:s1, None]
            cmin = Q.min(axis=0)
            carg = Q.argmin(axis=0) + pos
            upd = cmin < env                        # strict: keeps the first index attaining the minimum
            env[upd] = cmin[upd]
            act[upd] = carg[upd]
            pos = s1
        out.append(_polish_and_bound(A[:m], B[:m], C[:m], grid, env, act))
    return out


# ---------------------------------------------------------------------------------------------------------------
#  K = 3: certified bounds (abm/certify.py)
# ---------------------------------------------------------------------------------------------------------------
CERT_GAP = 1e-3            # relative width of the certified interval [lower, upper] for GNS*
CERT_TIME_LIMIT = 900.0    # seconds per checkpoint; the bounds stay valid if it is reached


def audit_k3_certified(Ms, ck_m, gap=CERT_GAP, time_limit=CERT_TIME_LIMIT):
    """Certified bounds of GNS* = max_lam min_i lam^T M_i lam at all checkpoints (simplicial branch and bound).
    Returns (lower, upper, lams, certified), squared units."""
    res = certify_k3_prefixes(Ms, ck_m, gap=gap, time_limit=time_limit)
    return ([r["lower"] for r in res], [r["upper"] for r in res], [r["lam"] for r in res],
            [bool(r["certified"]) for r in res])


def grid_maxmin_k3(Ms, resolution, chunk=4096):
    """Exact max of min_i lam^T M_i lam over the simplex grid of the given resolution (K = 3; used by the tests)."""
    Ms = np.asarray(Ms, dtype=float)
    lams = simplex_grid(3, resolution)
    Q = np.stack([Ms[:, 0, 0], Ms[:, 1, 1], Ms[:, 2, 2],
                  2.0 * Ms[:, 0, 1], 2.0 * Ms[:, 0, 2], 2.0 * Ms[:, 1, 2]], axis=1)
    best, best_lam = -np.inf, None
    for i in range(0, lams.shape[0], chunk):
        Bl = lams[i:i + chunk]
        P = np.stack([Bl[:, 0] ** 2, Bl[:, 1] ** 2, Bl[:, 2] ** 2,
                      Bl[:, 0] * Bl[:, 1], Bl[:, 0] * Bl[:, 2], Bl[:, 1] * Bl[:, 2]], axis=1)
        mins = (P @ Q.T).min(axis=1)
        j = int(np.argmax(mins))
        if float(mins[j]) > best:
            best, best_lam = float(mins[j]), Bl[j].copy()
    return best, best_lam
