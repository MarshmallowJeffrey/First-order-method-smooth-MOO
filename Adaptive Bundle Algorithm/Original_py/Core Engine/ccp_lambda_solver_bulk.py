"""ccp_lambda_solver_bulk.py — the multistart CCP λ-search of
``ccp_lambda_solver.py`` with the inner game LP handed to HiGHS in ONE
block per solve instead of one Python-to-C++ call per coefficient.

NEW FILE (Sep 10, 2026; user-approved).  ``ccp_lambda_solver.py`` is NOT
modified; this module subclasses it.

What changes and what does not
------------------------------
The LP solved at every CCP iterate is unchanged:

    max t   s.t.   Mc[i, :] · lam − t ≥ 0  for every bundle point i,
                   sum(lam) = 1,  lam ≥ 0,

with ``Mc[i, k] = 2 (Q_i lam_c)_k − lam_c^T Q_i lam_c`` (the tangent
linearisation of the convex quadratic phi_i at the current iterate).
The original ``_GameLP`` builds the model with one ``addRow`` call per
bundle point and rewrites the payoff block with one ``changeCoeff`` call
per entry (m·K calls per CCP iterate); at m ≈ 20,000 that is 200,000
pybind11 round trips per iterate, ~40 s per decision, and it dominates
every K = 10 campaign wall clock.  ``_GameLPBulk`` passes the whole LP
through ``Highs.passModel`` (row-wise CSR arrays built in NumPy) and
restores the previous optimal basis with ``Highs.setBasis`` so the dual
simplex still restarts warm.  Same numbers reach HiGHS, same basis,
same optimum; only the transport differs (measured Sep 10: 40 ms vs
1,456 ms per rewrite at m = 20,000, identical solutions).

Additionally, when the bundle has grown since the last solve (the outer
loop delivers a few new points per decision) the stored basis is
extended with the new rows marked basic, so the first LP of a decision
also starts warm (the original rebuilt cold).  A shrunk bundle (never
happens in the campaigns) falls back to a cold start.

Equivalence gate: ``sanity_check/sanity_checks_ccp_bulk.py`` replays the
recorded decision sequence of a campaign leg through both solvers and
compares phi, lambda and the CCP iteration counts.

Usage: replace ``CCPLambdaSolver`` by ``CCPLambdaSolverBulk``; the
configuration object, ``solve`` contract and telemetry are identical
(``stats_last["backend"]`` reads ``"highspy-bulk"``).
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from ccp_lambda_solver import (  # noqa: E402
    CCPConfig,
    CCPLambdaSolver,
    _GameLP,
    _HAS_HIGHSPY,
    _highspy,
)


class _GameLPBulk(_GameLP):
    """``_GameLP`` with block transport of the payoff matrix.

    Rows 0..m-1 are the payoff rows (K lambda coefficients and −1 on the
    t column), row m is the simplex equality — the same layout as the
    parent, so the basis vectors line up one to one.
    """

    def __init__(self, K: int, use_highspy: Optional[bool] = None):
        super().__init__(K, use_highspy=use_highspy)
        self._basis = None          # HighsBasis of the last optimal solve
        self._basis_m = None        # payoff row count that basis belongs to
        self.n_warm = 0             # solves started from a stored basis
        self.n_cold = 0

    # -- LP assembly (NumPy, row-wise CSR) --------------------------------
    def _lp_of(self, M: np.ndarray):
        m, K = M.shape
        INF = _highspy.kHighsInf
        lp = _highspy.HighsLp()
        lp.num_col_ = K + 1
        lp.num_row_ = m + 1
        lp.col_cost_ = np.r_[np.zeros(K), -1.0]          # minimise −t
        lp.col_lower_ = np.r_[np.zeros(K), -INF]         # lam ≥ 0, t free
        lp.col_upper_ = np.full(K + 1, INF)
        lp.row_lower_ = np.r_[np.zeros(m), 1.0]
        lp.row_upper_ = np.r_[np.full(m, INF), 1.0]
        lp.a_matrix_.format_ = _highspy.MatrixFormat.kRowwise
        payoff = np.hstack([M, -np.ones((m, 1))]).ravel()      # m·(K+1)
        lp.a_matrix_.value_ = np.concatenate([payoff, np.ones(K)])
        lp.a_matrix_.index_ = np.concatenate(
            [np.tile(np.arange(K + 1), m), np.arange(K)]).astype(np.int32)
        lp.a_matrix_.start_ = np.concatenate(
            [np.arange(0, m * (K + 1) + 1, K + 1),
             [m * (K + 1) + K]]).astype(np.int32)
        return lp

    def _highs(self):
        if self._h is None:
            h = _highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("presolve", "off")     # keep the basis usable
            h.setOptionValue("solver", "simplex")
            self._h = h
        return self._h

    def _basis_for(self, m: int):
        """Stored basis adapted to a payoff block of m rows (None = cold)."""
        if self._basis is None or self._basis_m is None:
            return None
        if m == self._basis_m:
            return self._basis
        if m > self._basis_m:
            grown = m - self._basis_m
            b = _highspy.HighsBasis()
            b.valid = True
            b.col_status = list(self._basis.col_status)
            rs = list(self._basis.row_status)
            b.row_status = (rs[:-1]
                            + [_highspy.HighsBasisStatus.kBasic] * grown
                            + rs[-1:])
            return b
        return None                                    # shrunk: cold start

    # -- the parent's hooks are replaced wholesale ------------------------
    def _build(self, M: np.ndarray) -> None:      # pragma: no cover
        raise RuntimeError("_GameLPBulk.resolve does not use _build.")

    def _rewrite(self, M: np.ndarray) -> None:    # pragma: no cover
        raise RuntimeError("_GameLPBulk.resolve does not use _rewrite.")

    def resolve(self, M: np.ndarray) -> Tuple[float, np.ndarray]:
        """Solve for payoff M (m, K); returns (t_star, lam_star)."""
        M = np.ascontiguousarray(np.asarray(M, dtype=float))
        m, K = M.shape
        if K != self.K:
            raise ValueError(f"payoff has K={K}, expected {self.K}.")
        if not self.use_highspy:
            return self._resolve_scipy(M)
        h = self._highs()
        basis = self._basis_for(m)
        h.passModel(self._lp_of(M))
        if basis is not None:
            h.setBasis(basis)
            self.n_warm += 1
        else:
            self.n_cold += 1
        h.run()
        self.n_solves += 1
        self.m = m
        status = h.getModelStatus()
        if status != _highspy.HighsModelStatus.kOptimal:
            if not self._warned_status:
                warnings.warn(
                    f"HiGHS returned status {status!r} on the game LP; "
                    "falling back to scipy for this solve.",
                    RuntimeWarning, stacklevel=2,
                )
                self._warned_status = True
            self._basis, self._basis_m = None, None
            return self._resolve_scipy(M)
        self.simplex_iters += int(h.getInfo().simplex_iteration_count)
        self._basis = h.getBasis()
        self._basis_m = m
        x = np.asarray(h.getSolution().col_value, dtype=float)
        return float(x[K]), x[:K].copy()


class CCPLambdaSolverBulk(CCPLambdaSolver):
    """``CCPLambdaSolver`` whose game LP is transported in one block."""

    def __init__(self, K: int, config: Optional[CCPConfig] = None):
        super().__init__(K, config)
        self.lp = _GameLPBulk(self.K, use_highspy=self.cfg.use_highspy)

    def _record(self, *args, **kwargs) -> None:
        super()._record(*args, **kwargs)
        self.stats_last["backend"] = ("highspy-bulk" if self.lp.use_highspy
                                      else "scipy")
        self.stats_last["lp_warm_solves"] = int(self.lp.n_warm)
        self.stats_last["lp_cold_solves"] = int(self.lp.n_cold)


def bulk_backend_available() -> bool:
    """True when highspy is importable (otherwise both solvers use scipy)."""
    return _HAS_HIGHSPY


__all__ = ["CCPLambdaSolverBulk", "CCPConfig", "bulk_backend_available"]
