"""Reported metric max_{lambda in simplex} GN(lambda, B), GN(lambda, B) = min_i ||J_i' lambda||.

  K=2  exact (lower envelope of parabolas).
  K=6  maximum over one fixed pool of 23,992 weights (vertices, center, 199 points per edge, 500 random
       points per face with 3..6 objectives), refined by CCP polishing: from each of the POLISH_STARTS best pool
       weights (pairwise distance > 0.08), CCP steps (paper Algorithm 2) to a local maximum, at most POLISH_MAXITER
       LPs.  Every value is GN at an actual weight, so the result is a numerical lower estimate; the evaluator is the
       same for every method.

All evaluators return (gn, argmax weight).
"""
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np

from . import lambda_solvers as ls

POOL_SEED = 20260910
POOL_PER_FACE = 500
POLISH_STARTS, POLISH_MAXITER = 32, 200


def fixed_stratified_weights(K=6, seed=POOL_SEED, per_face=POOL_PER_FACE):
    """The fixed pool of test weights for K>3."""
    rng = np.random.default_rng(seed)
    rows = [np.eye(K), np.ones((1, K)) / K]
    ts = np.linspace(0.0, 1.0, 201)[1:-1]
    for i, j in combinations(range(K), 2):
        w = np.zeros((len(ts), K))
        w[:, i] = ts
        w[:, j] = 1.0 - ts
        rows.append(w)
    for support_size in range(3, K + 1):
        for support in combinations(range(K), support_size):
            w = np.zeros((per_face, K))
            w[:, support] = rng.dirichlet(np.ones(support_size), size=per_face)
            rows.append(w)
    return np.vstack(rows)


def evaluator_sha256():
    """Fingerprint of the K>2 evaluator: the source of this module and of the LP / CCP code, and the fixed pool."""
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(Path(ls.__file__).read_bytes())
    h.update(np.ascontiguousarray(fixed_stratified_weights()).tobytes())
    return h.hexdigest()


def gram(J):
    return J @ J.transpose(0, 2, 1)


def k2_exact_gram(Q):
    solver = ls.Envelope()
    for q in Q:
        solver.add(q)
    value, weight = solver.solve()
    return np.sqrt(value), weight


def polish(Q, weights, values):
    """max over the pool (values = min_i w' Q_i w for every pool weight w) and over the CCP local maxima reached
    from the POLISH_STARTS best pool weights that are more than 0.08 apart: (gn, weight)."""
    i = int(values.argmax()); best, wb = float(values[i]), weights[i]
    if best <= 0:
        return 0.0, np.asarray(wb)
    qs = Q / best  # phi normalized by the pool maximum, as in the CCP of GRAB
    starts = []
    for i in np.argsort(-values, kind="stable"):
        if all(np.linalg.norm(weights[i] - s) > .08 for s in starts):
            starts.append(weights[i])
        if len(starts) >= POLISH_STARTS:
            break
    ls.reset_lp_state()
    top = 1.0
    for w in starts:
        _, value, point, _ = ls.ccp_ascent(qs, w, POLISH_MAXITER)
        if value > top:
            top, wb = value, point
    return float(np.sqrt(top * best)), np.asarray(wb)


def pool_ccp_gn_gram(Q, weights, chunk=2000):
    values = np.concatenate([np.einsum("wk,nkl,wl->wn", weights[s:s + chunk], Q, weights[s:s + chunk],
                                       optimize=True).min(axis=1) for s in range(0, len(weights), chunk)])
    return polish(Q, weights, values)


def pool_ccp_prefixes(final_J, bundle_sizes, weights, chunk=400):
    """The K>2 metric of every prefix B[:n] of one bundle (the adaptive bundles are nested)."""
    Q = gram(final_J)
    values = np.zeros((len(weights), len(bundle_sizes)))
    for start in range(0, len(weights), chunk):
        w = weights[start:start + chunk]
        running = np.minimum.accumulate(np.einsum("wk,nkl,wl->wn", w, Q, w, optimize=True), axis=1)
        values[start:start + chunk] = running[:, np.asarray(bundle_sizes) - 1]
    return np.array([polish(Q[:n], weights, values[:, i])[0] for i, n in enumerate(bundle_sizes)])


def reporting_metric_gram(Q, pool=None):
    K = Q.shape[1]
    if K == 2:
        return k2_exact_gram(Q)
    if pool is None:
        raise ValueError("K>2 needs the fixed weight pool")
    return pool_ccp_gn_gram(Q, pool)
