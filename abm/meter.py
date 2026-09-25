"""The worst-case gradient norm of a bundle B = {theta_1, ..., theta_m}: max over lam in Delta_K of min_i ||J_i^T lam||,
J_i the Jacobian at theta_i, computed from the Gram matrices M_i = J_i J_i^T.  The functions below work with the
squared norm min_i lam^T M_i lam; the runs record its square root.

K = 2 (exact).  With lam = (w, 1 - w) every q_i(w) = lam^T M_i lam is a convex quadratic, so GN(w) is the lower
envelope of m parabolas.  The envelope is evaluated exactly on a uniform grid of w, the best grid cell is polished
in closed form (the maximum of an envelope of convex quadratics is at an end point or at a crossing of two of them),
and a certified upper bound follows from the slopes of the active quadratics on every cell.

K = 3 (lower bound).  At every checkpoint: the best of two multistart CCP searches (8,192 seeds, 20 polished) and a
simplex grid of resolution 500.  At the last checkpoint at or before B/8, B/4, B/2 and B additionally: SLSQP
from 10 fixed starts and the previous level's maximizer, a CCP search with the row-wise LP transport, and a grid of
resolution 1,000.  Each value is GN at a feasible lambda, hence a lower bound.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .ccp import CCPConfig, CCPSolver
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
#  K = 3: lower bounds
# ---------------------------------------------------------------------------------------------------------------
HEAVY_CCP = dict(N0=8192, r=20)
CHEAP_SEEDS = (1, 2)
GRID_RES, GRID_RES_FULL = 500, 1000


def grid_maxmin_k3(Ms, resolution, chunk=4096):
    """Exact max of min_i lam^T M_i lam over the simplex grid of the given resolution (K = 3)."""
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


def _slsqp_starts(K, prev_lam):
    """Centroid, vertices, near-vertices (0.8), the previous maximizer and the edge midpoints."""
    EPS, NEAR = 1e-8, 0.8
    starts = [np.full(K, 1.0 / K)]
    for k in range(K):
        e = np.full(K, EPS)
        e[k] = 1.0 - (K - 1) * EPS
        starts.append(e)
    for k in range(K):
        e = np.full(K, (1.0 - NEAR) / (K - 1))
        e[k] = NEAR
        starts.append(e)
    if prev_lam is not None:
        starts.append(np.clip(prev_lam, EPS, 1.0))
    for a in range(K):
        for b in range(a + 1, K):
            e = np.full(K, EPS)
            e[a] = 0.5 - (K - 2) * 0.5 * EPS
            e[b] = 0.5 - (K - 2) * 0.5 * EPS
            starts.append(e)
    return starts


def slsqp_max_gn(Ms, prev_lam=None):
    """Multistart SLSQP on max_lam min_i lam^T M_i lam (ftol 1e-6, 100 iterations); every candidate is scored at
    its projection onto the simplex.  Returns (value, lam)."""
    Ms = np.asarray(Ms, dtype=float)
    K = Ms.shape[1]

    def neg_gn(lam):
        return -float(np.min(np.einsum('k,ikl,l->i', lam, Ms, lam)))

    def neg_gn_jac(lam):
        vals = np.einsum('k,ikl,l->i', lam, Ms, lam)
        return -(2.0 * (Ms[int(np.argmin(vals))] @ lam))

    def project(v):
        v = np.maximum(np.asarray(v, dtype=float), 0.0)
        s = float(v.sum())
        return np.full(K, 1.0 / K) if (not np.isfinite(s) or s <= 0.0) else v / s

    constraints = [{"type": "eq", "fun": lambda l: float(np.sum(l) - 1.0), "jac": lambda l: np.ones(K)}]
    starts = _slsqp_starts(K, prev_lam)
    best_val, best_lam = np.inf, project(starts[0])
    for lam0 in starts:
        lam0 = project(lam0)
        v0 = neg_gn(lam0)
        if v0 < best_val:
            best_val, best_lam = float(v0), lam0
        try:
            res = minimize(neg_gn, lam0, jac=neg_gn_jac, method="SLSQP", bounds=[(1e-8, 1.0)] * K,
                           constraints=constraints, options={"ftol": 1e-6, "maxiter": 100})
        except Exception:
            continue
        lam_res = project(res.x)
        v_res = neg_gn(lam_res)
        if np.isfinite(v_res) and v_res < best_val:
            best_val, best_lam = float(v_res), lam_res
    return float(-best_val), best_lam


def _heavy_ccp(Ms, seed, transport):
    return CCPSolver(3, CCPConfig(seed=int(seed), **HEAVY_CCP), transport=transport).solve(Ms)


def audit_checkpoint_k3(Ms):
    """Per-checkpoint lower bound: two CCP searches and the grid of resolution 500.  Returns (value, lam)."""
    Ms = np.asarray(Ms, dtype=float)
    vals, lams = [], []
    for s in CHEAP_SEEDS:
        v, lam = _heavy_ccp(Ms, s, "bulk")
        vals.append(float(v))
        lams.append([float(t) for t in lam])
    v_grid, lam_grid = grid_maxmin_k3(Ms, GRID_RES)
    j = int(np.argmax(vals))
    if v_grid > vals[j]:
        return float(v_grid), [float(t) for t in lam_grid]
    return vals[j], lams[j]


def audit_level_k3(Ms, prev_lam):
    """Additional instruments at the level checkpoints: SLSQP, CCP (row transport, seed 1), grid 1,000."""
    Ms = np.asarray(Ms, dtype=float)
    v_i, lam_i = slsqp_max_gn(Ms, prev_lam)
    v_c, lam_c = _heavy_ccp(Ms, 1, "rows")
    value, lam = (float(v_c), np.asarray(lam_c, float)) if v_c >= v_i else (float(v_i), np.asarray(lam_i, float))
    lam = [float(t) for t in lam]
    v_g, lam_g = grid_maxmin_k3(Ms, GRID_RES_FULL)
    if v_g > value:
        value, lam = float(v_g), [float(t) for t in lam_g]
    return value, lam


def audit_k3(Ms, ck_m, ck_grads, budget):
    """Lower bounds of max_lam min_i lam^T M_i lam at all checkpoints; returns (values, lambdas)."""
    values, lams = [], []
    for m in ck_m:
        v, lam = audit_checkpoint_k3(Ms[:m])
        values.append(v)
        lams.append(lam)
    ck = np.asarray(ck_grads)
    levels = [budget / 8, budget / 4, budget / 2, budget]
    level_idx = sorted(set([int(np.nonzero(ck <= L + 1e-9)[0][-1]) for L in levels if np.any(ck <= L + 1e-9)]
                           + [len(ck_m) - 1]))
    prev = None
    for i in level_idx:
        v, lam = audit_level_k3(Ms[:ck_m[i]], prev)
        if v > values[i]:
            values[i], lams[i] = v, lam
        prev = np.asarray(lams[i], dtype=float)
    return values, lams
