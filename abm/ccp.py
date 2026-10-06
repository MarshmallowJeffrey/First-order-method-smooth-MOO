"""Helpers of the K = 3 lambda-search (Algorithm 2 with constraint generation, abm/ccp_cg.py).

For a bundle with Gram matrices Q_i = J_i J_i^T (m x K x K), phi_i(lam) = lam^T Q_i lam and
phi(lam) = min_i phi_i(lam), the squared gradient norm of the bundle at lam.

sample_simplex_exp: uniform points on Delta_K (normalized Exp(1) vectors); _phi_terms: Q_i lam and phi_i(lam);
_project_simplex and _active_set; _scipy_game and GameLP: the matrix-game LP max{t : M lam >= t 1, lam in Delta_K},
by scipy and by HiGHS (dual simplex, warm-started).  Algorithm 2 solves its LPs by constraint generation
(abm/ccp_cg.py); it uses GameLP only when run with the full LP, for checks, and _scipy_game when HiGHS fails.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import highspy
import numpy as np
from scipy.optimize import linprog


def sample_simplex_exp(n: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """n uniform points on Delta_K (normalized Exp(1) vectors)."""
    if n <= 0:
        return np.zeros((0, K))
    E = rng.standard_exponential(size=(n, K))
    s = E.sum(axis=1, keepdims=True)
    s[s <= 0.0] = 1.0
    return E / s


def _phi_terms(Q: np.ndarray, lam: np.ndarray):
    """G[i] = Q_i lam (m, K) and phi_i(lam) (m,)."""
    G = Q @ lam
    return G, G @ lam


def _project_simplex(lam: np.ndarray, K: int) -> np.ndarray:
    lam = np.maximum(np.asarray(lam, dtype=float), 0.0)
    s = float(lam.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.full(K, 1.0 / K)
    return lam / s


def _active_set(phis: np.ndarray, tol: float) -> frozenset:
    lo = float(np.min(phis))
    return frozenset(np.nonzero(phis <= lo + tol * max(1.0, abs(lo)))[0].tolist())


def _scipy_game(M: np.ndarray) -> Tuple[float, np.ndarray]:
    m, K = M.shape
    res = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.c_[-M, np.ones(m)], b_ub=np.zeros(m),
                  A_eq=np.r_[np.ones(K), 0.0].reshape(1, -1), b_eq=[1.0],
                  bounds=[(0.0, None)] * K + [(None, None)], method="highs")
    if not res.success:
        raise RuntimeError(f"game LP failed: {res.message}")
    x = np.asarray(res.x, dtype=float)
    return float(x[K]), x[:K].copy()


class GameLP:
    """max{t : M lam >= t 1, sum(lam) = 1, lam >= 0} for successive payoffs M (m x K).
    Columns (lam_1..lam_K, t); rows: the m payoff rows, then the simplex equality."""

    def __init__(self, K: int, transport: str = "bulk"):
        if transport not in ("bulk", "rows"):
            raise ValueError(transport)
        self.K, self.transport = int(K), transport
        self.m = 0
        self._h = None
        self._basis, self._basis_m = None, None
        self._warned = False

    def _new_highs(self):
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", "off")          # keep the basis usable
        h.setOptionValue("solver", "simplex")
        return h

    # -- "rows": one addRow per payoff row; same-size payoffs are rewritten in place --------------------------
    def _build_rows(self, M):
        m, K = M.shape
        INF = highspy.kHighsInf
        h = self._new_highs()
        h.addCols(K + 1, np.r_[np.zeros(K), -1.0], np.r_[np.zeros(K), -INF], np.full(K + 1, INF), 0, [], [], [])
        idx = np.r_[np.arange(K), K].astype(np.int32)
        for i in range(m):
            h.addRow(0.0, INF, K + 1, idx, np.r_[M[i], -1.0])
        h.addRow(1.0, 1.0, K, np.arange(K, dtype=np.int32), np.ones(K))
        self._h, self.m = h, m

    def _rewrite_rows(self, M):
        m, K = M.shape
        for i in range(m):
            row = M[i]
            for k in range(K):
                self._h.changeCoeff(i, k, float(row[k]))

    # -- "bulk": the whole LP in one passModel; the previous basis restored (grown bundles: new rows basic) --
    def _lp_of(self, M):
        m, K = M.shape
        INF = highspy.kHighsInf
        lp = highspy.HighsLp()
        lp.num_col_ = K + 1
        lp.num_row_ = m + 1
        lp.col_cost_ = np.r_[np.zeros(K), -1.0]
        lp.col_lower_ = np.r_[np.zeros(K), -INF]
        lp.col_upper_ = np.full(K + 1, INF)
        lp.row_lower_ = np.r_[np.zeros(m), 1.0]
        lp.row_upper_ = np.r_[np.full(m, INF), 1.0]
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        payoff = np.hstack([M, -np.ones((m, 1))]).ravel()
        lp.a_matrix_.value_ = np.concatenate([payoff, np.ones(K)])
        lp.a_matrix_.index_ = np.concatenate([np.tile(np.arange(K + 1), m), np.arange(K)]).astype(np.int32)
        lp.a_matrix_.start_ = np.concatenate([np.arange(0, m * (K + 1) + 1, K + 1),
                                              [m * (K + 1) + K]]).astype(np.int32)
        return lp

    def _basis_for(self, m):
        if self._basis is None or self._basis_m is None:
            return None
        if m == self._basis_m:
            return self._basis
        if m > self._basis_m:
            b = highspy.HighsBasis()
            b.valid = True
            b.col_status = list(self._basis.col_status)
            rs = list(self._basis.row_status)
            b.row_status = rs[:-1] + [highspy.HighsBasisStatus.kBasic] * (m - self._basis_m) + rs[-1:]
            return b
        return None

    def solve(self, M: np.ndarray) -> Tuple[float, np.ndarray]:
        """Returns (t*, lam*)."""
        M = np.ascontiguousarray(np.asarray(M, dtype=float))
        m, K = M.shape
        if self.transport == "rows":
            if self._h is None or m != self.m:
                self._build_rows(M)
            else:
                self._rewrite_rows(M)
            h = self._h
            h.run()
        else:
            if self._h is None:
                self._h = self._new_highs()
            h = self._h
            basis = self._basis_for(m)
            h.passModel(self._lp_of(M))
            if basis is not None:
                h.setBasis(basis)
            h.run()
            self.m = m
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            if not self._warned:
                warnings.warn("HiGHS did not reach an optimal game LP; using scipy for this solve.", RuntimeWarning)
                self._warned = True
            if self.transport == "bulk":
                self._basis, self._basis_m = None, None
            return _scipy_game(M)
        if self.transport == "bulk":
            self._basis = h.getBasis()
            self._basis_m = m
        x = np.asarray(h.getSolution().col_value, dtype=float)
        return float(x[K]), x[:K].copy()
