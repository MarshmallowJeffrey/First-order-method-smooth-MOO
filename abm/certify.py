"""Certified worst-case gradient norm for K = 3: simplicial branch and bound (the global optimization fallback of
Appendix A.1 of the paper).

For a bundle with Gram matrices Q_i (m x 3 x 3), phi_i(lam) = lam^T Q_i lam, phi(lam) = min_i phi_i(lam) and
GNS* = max_{lam in Delta_3} phi(lam).  On a triangle T = conv(v_1, v_2, v_3) of the simplex, convexity of phi_i gives
phi_i(lam) <= sum_j mu_j phi_i(v_j) for lam = sum_j mu_j v_j, hence

    max_{lam in T} phi(lam) <= U(T) := min_i max_j phi_i(v_j),

and every vertex value phi(v_j) is a lower bound of GNS*.  Starting from the simplex, the open triangle with the largest
U(T) is bisected at the midpoint of its longest edge until U(T) <= (1 + gap) L for every open triangle, L being the best
vertex value found; [L, max_T U(T)] then contains GNS* and its relative width is at most gap.  A triangle whose U(T) is
already within that tolerance is never split again (L only grows), so only its U(T) is kept.  A row i is dropped from
T (and from its sub-triangles) when its tangent plane at the centroid of T exceeds U(T) on all of T: phi_i then exceeds
max_T phi everywhere in T, so it is never the minimum there and dropping it changes neither phi on T nor the bounds.
"""

from __future__ import annotations

import heapq
import time

import numpy as np

_IU = np.triu_indices(3)
_COEF = np.where(_IU[0] == _IU[1], 1.0, 2.0)
_EDGES = ((0, 1), (1, 2), (0, 2))


def _sym(Q):
    return np.ascontiguousarray(Q[:, _IU[0], _IU[1]] * _COEF)          # phi_i(v) = Qs[i] . _pvec(v)


def _pvec(v):
    return v[_IU[0]] * v[_IU[1]]


def phi_at(Q, lam):
    """phi(lam) on the whole bundle."""
    return float(np.min(_sym(Q) @ _pvec(np.asarray(lam, dtype=float))))


def certify_k3(Q, gap=1e-3, lam0=None, time_limit=None):
    """Branch and bound on the bundle Q (m x 3 x 3).  lam0: a known good point (its value seeds the lower bound).
    Returns a dict: lower and upper (bounds of GNS*, squared units), lam (a point with phi(lam) = lower), certified
    (upper <= (1 + gap) lower), splits, seconds."""
    t0 = time.perf_counter()
    Q = np.ascontiguousarray(np.asarray(Q, dtype=float))
    if Q.ndim != 3 or Q.shape[1:] != (3, 3):
        raise ValueError("certify_k3 needs Gram matrices of shape (m, 3, 3)")
    Qs = _sym(Q)
    best_val, best_lam = -np.inf, None
    if lam0 is not None:
        lam0 = np.asarray(lam0, dtype=float)
        best_val, best_lam = float(np.min(Qs @ _pvec(lam0))), lam0.copy()
    heap, uid, closed = [], 0, -np.inf             # closed: largest U(T) of the triangles that need no split

    def push(V, rows, F):
        nonlocal best_val, best_lam, uid, closed
        ub = float(F.max(axis=1).min())
        vals = F.min(axis=0)
        j = int(np.argmax(vals))
        if vals[j] > best_val:
            best_val, best_lam = float(vals[j]), V[j].copy()
        if ub <= (1.0 + gap) * best_val:
            closed = max(closed, ub)
            return
        c = V.mean(axis=0)
        low = Qs[rows] @ _pvec(c) + ((2.0 * (Q[rows] @ c)) @ (V - c).T).min(axis=1)
        keep = low <= ub + 1e-9 * abs(ub)
        uid += 1
        heapq.heappush(heap, (-ub, uid, V, rows[keep], F[keep]))

    V0 = np.eye(3)
    push(V0, np.arange(Q.shape[0], dtype=np.int32), np.stack([Qs @ _pvec(v) for v in V0], axis=1))
    splits, timed_out = 0, False
    while heap and heap[0][0] < -(1.0 + gap) * best_val:
        if time_limit is not None and time.perf_counter() - t0 > time_limit:
            timed_out = True
            break
        _, _, V, rows, F = heapq.heappop(heap)
        a, b = max(_EDGES, key=lambda e: float(np.sum((V[e[0]] - V[e[1]]) ** 2)))
        o = 3 - a - b
        mid = 0.5 * (V[a] + V[b])
        fm = Qs[rows] @ _pvec(mid)
        for k in (a, b):
            push(np.stack([V[k], V[o], mid]), rows, np.stack([F[:, k], F[:, o], fm], axis=1))
        splits += 1
    upper = max(closed, -heap[0][0] if heap else -np.inf)
    return {"lower": best_val, "upper": upper, "lam": best_lam, "certified": not timed_out,
            "splits": splits, "seconds": time.perf_counter() - t0}


def certify_k3_prefixes(Ms, ck_m, gap=1e-3, time_limit=None):
    """certify_k3 on every bundle prefix Ms[:m], m in ck_m (non-decreasing); the maximizer found for one prefix seeds
    the lower bound of the next.  Returns a list of the result dicts (lam as a list)."""
    Ms = np.asarray(Ms, dtype=float)
    out, lam = [], None
    for m in ck_m:
        r = certify_k3(Ms[:int(m)], gap=gap, lam0=lam, time_limit=time_limit)
        lam = r["lam"]
        out.append(dict(r, lam=[float(x) for x in r["lam"]]))
    return out
