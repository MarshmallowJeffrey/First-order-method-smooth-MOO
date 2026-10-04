"""Reported metric max_{lambda in simplex} GN(lambda, B), GN(lambda, B) = min_i ||J_i' lambda||.

  K=2  exact (lower envelope of parabolas).
  K=6  maximum over one fixed pool of 23,992 weights (vertices, center, 199 points per edge, 500 random
       points per face with 3..6 objectives): a numerical lower estimate, the same for every method.

All evaluators return (gn, argmax weight, gn_upper).
"""
from itertools import combinations

import numpy as np

from .lambda_solvers import Envelope

POOL_SEED = 20260910
POOL_PER_FACE = 500


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


def gram(J):
    return J @ J.transpose(0, 2, 1)


def k2_exact_gram(Q):
    solver = Envelope()
    for q in Q:
        solver.add(q)
    value, weight = solver.solve()
    return np.sqrt(value), weight, np.sqrt(value)


def pool_gn_gram(Q, weights, chunk=2000):
    best_sq, best_w = 0.0, weights[0]
    for start in range(0, len(weights), chunk):
        w = weights[start:start + chunk]
        values = np.einsum("wk,nkl,wl->wn", w, Q, w, optimize=True).min(axis=1)
        i = int(values.argmax())
        if float(values[i]) > best_sq:
            best_sq, best_w = float(values[i]), w[i]
    return float(np.sqrt(max(best_sq, 0.0))), np.asarray(best_w), float(np.sqrt(max(best_sq, 0.0)))


def pool_gn_prefixes(final_J, bundle_sizes, weights, chunk=400):
    """Pool GN of every prefix B[:n] of one bundle (the adaptive bundles are nested)."""
    Q = gram(final_J)
    answer = np.zeros(len(bundle_sizes), float)
    for start in range(0, len(weights), chunk):
        w = weights[start:start + chunk]
        values = np.einsum("wk,nkl,wl->wn", w, Q, w, optimize=True)
        running = np.minimum.accumulate(values, axis=1)
        for i, n in enumerate(bundle_sizes):
            answer[i] = max(answer[i], float(running[:, n - 1].max()))
    return np.sqrt(answer)


def reporting_metric_gram(Q, pool=None):
    K = Q.shape[1]
    if K == 2:
        return k2_exact_gram(Q)
    if pool is None:
        raise ValueError("K>2 needs the fixed weight pool")
    return pool_gn_gram(Q, pool)
