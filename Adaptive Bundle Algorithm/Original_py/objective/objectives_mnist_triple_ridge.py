"""objectives_mnist_triple_ridge.py — ridge-penalised K = 3 MNIST
digit-triple objectives: F_k_mu(theta) = F_k(theta)
+ (mu/2)*||theta||^2.

NEW FILE (Aug 26, 2026).  User-approved design (Aug-26 Q&A: add a
penalty term to the formal K = 3 experiment; mu = 1e-4 fixed by the
user's direct call).  Per the user's explicit instruction the base
module ``objectives_mnist_triple`` is NOT modified — this file wraps
it; nothing is copied except the thin oracle subclass.

Why a ridge term (motivation of record):

* Without it the boundary of the simplex is ill-posed: at vertex/edge
  lambdas the ignored class's CE diverges (Smoke A measured ~25 / ~15
  after 15 segments) because driving the favoured classes' CE toward
  0 needs ||theta|| -> inf, so the scalarized infimum is not
  attained.  With mu > 0 every scalarization lam^T F_mu is coercive:
  a minimizer exists for ALL lam in the simplex, iterates stay
  bounded, the Pareto front is bounded (grid vertex/edge nodes become
  genuine anchor points instead of runaway arms), and the probe-based
  L is honest on the reachable set.

Exact form (user-approved):

* Same mu for all three objectives; penalty over ALL d = 8,195
  parameters, biases included — the softmax logit-shift direction is
  CE-flat, so coercivity needs every coordinate (deliberate deviation
  from the no-bias-decay ML habit).
* Since sum(lam) = 1 on the simplex, penalising every objective
  equals adding the penalty ONCE to any scalarization:
  lam^T F_mu = lam^T F + (mu/2)*||theta||^2.
* The penalty gradient mu*theta is deterministic and touches no data
  rows: ifo accounting and the segment-cost formula are UNCHANGED.
  In the MSVRG estimator the anchor penalty terms cancel exactly:
  (g_B(y) + mu*y) - (g_B(a) + mu*a) + (g_full(a) + mu*a)
      = classic estimator + mu*y.
* L: the penalty's Hessian shift is exactly mu*I, so L_k + mu is a
  valid smoothness constant; we return the base probe L plus mu
  analytically (an upper bound — safe for the step rule; no
  re-probing, so the base probe cost is paid once as before).
* Reporting split (user-approved): training fvals / Gram stacks /
  GN* certificates are in PENALISED coordinates (that is the problem
  being solved); the official-test evaluation stays RAW CE (the base
  ``evaluate_triple`` / ``_test_eval_stack`` are reused unchanged).
  The penalty adds the SAME value (mu/2)*||theta||^2 to all three
  coordinates of a point, so dominance relations differ between the
  two coordinate systems; raw train values are exactly recoverable at
  plot time from thetas.npz.

mu = 0.0 degenerates to the base problem bit-identically (every
penalty line is guarded on ``mu != 0.0`` and L/joint pass through
untouched) — this is the replica-fidelity gate of the ridge campaign
runner.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from objectives_mnist_triple import (
    TripleStochLamOracle,
    make_mnist_triple,
)


class RidgeTripleStochLamOracle(TripleStochLamOracle):
    """Stochastic lam-scalarized oracle of the PENALISED objectives.

    Identical batch stream and ifo accounting as the base class (same
    seed protocol; the penalty touches no data rows).  Only change:
    ``grad_pair`` adds mu*theta for each net at its own theta; the
    anchor theta is recorded by ``set_anchor`` for that purpose.
    """

    def __init__(self, X_np, labels_np, *, batch_size: int, seed: int,
                 mu: float):
        super().__init__(X_np, labels_np, batch_size=batch_size, seed=seed)
        self.mu = float(mu)
        self._theta_a: np.ndarray | None = None

    def set_anchor(self, theta_a: np.ndarray) -> None:
        theta_a = np.asarray(theta_a, dtype=float)
        self._theta_a = theta_a.copy()
        super().set_anchor(theta_a)

    def grad_pair(self, theta_y, lam, batch):
        theta_y = np.asarray(theta_y, dtype=float)
        g_y, g_a = super().grad_pair(theta_y, lam, batch)
        if self.mu != 0.0:
            g_y = g_y + self.mu * theta_y
            g_a = g_a + self.mu * self._theta_a
        return g_y, g_a


def make_mnist_triple_ridge(digits: Sequence[int], mu: float,
                            per_class: int | None = None, *,
                            batch_size: int = 1024, sampler_seed: int = 41,
                            init_seed: int = 8, n_probes: int = 40,
                            probe_seed: int = 7):
    """Penalised factory — same return signature as make_mnist_triple:
    (objectives, grad_objectives, L, joint_oracle, stoch, meta).

    Delegates everything to the base factory, then wraps values,
    Jacobian rows, per-objective closures and the stochastic oracle
    with the exact penalty terms.  The base stochastic oracle is
    replaced by a Ridge one built on the same arrays with the same
    sampler seed — identical batch stream.  meta gains "mu".
    """
    mu = float(mu)
    (obj0, grad0, L0, joint0, _stoch0, meta) = make_mnist_triple(
        digits, per_class, batch_size=batch_size,
        sampler_seed=sampler_seed, init_seed=init_seed,
        n_probes=n_probes, probe_seed=probe_seed)
    K = meta["K"]

    def joint_oracle(theta: np.ndarray):
        fvals, J = joint0(theta)
        if mu != 0.0:
            th = np.asarray(theta, dtype=float)
            fvals = fvals + 0.5 * mu * float(th @ th)
            J = J + mu * th     # broadcast: every objective row + mu*theta
        return fvals, J

    def _obj(k):
        def f(th):
            v = obj0[k](th)
            if mu != 0.0:
                th_arr = np.asarray(th, dtype=float)
                v += 0.5 * mu * float(th_arr @ th_arr)
            return v
        return f

    def _grad(k):
        def g(th):
            row = grad0[k](th)
            if mu != 0.0:
                row = row + mu * np.asarray(th, dtype=float)
            return row
        return g

    objectives = [_obj(k) for k in range(K)]
    grad_objectives = [_grad(k) for k in range(K)]
    L = (np.asarray(L0, dtype=float) + mu if mu != 0.0
         else np.asarray(L0, dtype=float))       # exact Hessian shift mu*I
    stoch = RidgeTripleStochLamOracle(
        meta["_X"], meta["_y"], batch_size=batch_size,
        seed=sampler_seed, mu=mu)
    meta = dict(meta)
    meta["mu"] = mu
    return objectives, grad_objectives, L, joint_oracle, stoch, meta
