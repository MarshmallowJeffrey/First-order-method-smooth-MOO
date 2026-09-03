"""objectives_mnist_pair_ridge.py — ridge-penalised K = 2 MNIST
digit-pair objectives: F_k_mu(theta) = F_k(theta) + (mu/2)*||theta||^2.

NEW FILE (Sep 2, 2026), the K = 2 port of
``objectives_mnist_triple_ridge.py`` (Aug 26; that file's docstring
carries the full motivation of record).  User decision Sep 2: the K = 2
pair campaign v2 runs on the RIDGE problem only — the unregularized
vertex solutions diverge (runaway arms), which breaks SURF's
bounded-speed assumptions (verbatim SURF collapses onto the vertices);
regularization bounds the front instead of windowing the measurement.
The base module ``objectives_mnist_pair`` is NOT modified — this file
wraps it; nothing is copied except the thin oracle subclass.

Exact form (identical to the K3 ridge design):

* Same mu for both objectives; penalty over ALL d = 8,098 parameters,
  biases included (coercivity needs every coordinate).
* sum(lam) = 1 on the simplex, so penalising every objective equals
  adding the penalty ONCE to any scalarization.
* The penalty gradient mu*theta touches no data rows: ifo accounting
  and the segment-cost formula are UNCHANGED.  In the MSVRG estimator
  the anchor penalty terms cancel exactly:
  (g_B(y) + mu*y) - (g_B(a) + mu*a) + (g_full(a) + mu*a)
      = classic estimator + mu*y.
* L: the penalty's Hessian shift is exactly mu*I, so the base probe L
  plus mu analytically (no re-probing).
* Reporting split: training fvals / Grams / GN in PENALISED
  coordinates (that is the problem being solved); official-test
  evaluation stays RAW CE (the base ``_test_eval_stack`` is reused
  unchanged); raw train values recoverable from thetas.npz.

mu = 0.0 degenerates to the base problem bit-identically (every
penalty line is guarded on ``mu != 0.0``).
"""

from __future__ import annotations

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from objectives_mnist_pair import (
    PairStochLamOracle,
    make_mnist_pair,
)


class RidgePairStochLamOracle(PairStochLamOracle):
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


def make_mnist_pair_ridge(digit_a: int, digit_b: int, mu: float,
                          per_class: int | None = None, *,
                          batch_size: int = 1024, sampler_seed: int = 41,
                          init_seed: int = 8, n_probes: int = 40,
                          probe_seed: int = 7):
    """Penalised factory — same return signature as make_mnist_pair:
    (objectives, grad_objectives, L, joint_oracle, stoch, meta).

    Delegates everything to the base factory, then wraps values,
    Jacobian rows, per-objective closures and the stochastic oracle
    with the exact penalty terms.  The base stochastic oracle is
    replaced by a Ridge one built on the same arrays with the same
    sampler seed — identical batch stream.  meta gains "mu".
    """
    mu = float(mu)
    (obj0, grad0, L0, joint0, _stoch0, meta) = make_mnist_pair(
        digit_a, digit_b, per_class=per_class, batch_size=batch_size,
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
    stoch = RidgePairStochLamOracle(
        meta["_X"], meta["_y"], batch_size=batch_size,
        seed=sampler_seed, mu=mu)
    meta = dict(meta)
    meta["mu"] = mu
    return objectives, grad_objectives, L, joint_oracle, stoch, meta
