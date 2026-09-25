"""The lambda-search of the adaptive bundle method: multistart convex-concave procedure (CCP).

For a bundle with Gram matrices Q_i = J_i J_i^T (m x K x K), phi_i(lam) = lam^T Q_i lam and
phi(lam) = min_i phi_i(lam), the squared gradient norm of the bundle at lam; the search approximates
argmax_{lam in Delta_K} phi(lam).

1. Sandwich: max_k min_i [Q_i]_kk <= max phi <= val(A) with A_ik = [Q_i]_kk; if it closes, the best vertex is
   returned.
2. Seeds: the K vertices, the maximizer of the game A, the pool carried from the previous call and N0 random
   points (normalized Exp(1) vectors).
3. Screening: phi on all seeds; keep the r best seeds that are at least 0.05 apart in l1.
4. Polish each kept seed by CCP: at lam_c, linearize every phi_i (M[i, k] = 2 (Q_i lam_c)_k - phi_i(lam_c)), solve
   the matrix-game LP max{t : M lam >= t 1, lam in Delta_K} with HiGHS and move to its solution; stop when the
   predicted gain is at most tau = 1e-8 max(1, phi(lam_c)) or after T_max = 100 LPs.
5. Pool: deduplicate the polished points and the old pool, keep at most 3 r of them (always the winner); they seed
   the next call.

The game LP is passed to HiGHS either in one block with the previous basis restored ("bulk", used for the
decisions of the adaptive method and the per-checkpoint audits) or row by row with the coefficients rewritten in
place ("rows", used by the heavy CCP of the K = 3 full audits).  Both warm-start the dual simplex.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def phi_batch(Q: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """phi(lam) for a batch of lambdas (one matmul)."""
    m, K = Q.shape[0], Q.shape[1]
    n = lams.shape[0]
    outer = (lams[:, :, None] * lams[:, None, :]).reshape(n, K * K)
    vals = outer @ Q.reshape(m, K * K).T          # (n, m)
    idx = np.argmin(vals, axis=1)
    return vals[np.arange(n), idx]


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


@dataclass
class CCPConfig:
    N0: int = 2000                 # random seeds per call
    r: int = 10                    # seeds polished per call
    pool_cap_factor: int = 3       # pool cap = factor * r
    tau_rel: float = 1e-8          # relative stationarity tolerance
    T_max: int = 100               # CCP iterations per seed
    screen_sep_l1: float = 0.05    # l1 separation of the kept seeds
    dedup_l1_tol: float = 1e-3     # pool: same point if l1-closer than this ...
    dedup_phi_rel: float = 1e-9    # ... or same active set and GN within this relative distance
    active_tol: float = 1e-9       # tolerant active set
    seed: int = 0                  # random seeds


class CCPSolver:
    """Stateful across calls (pool, random stream, LP warm start); one instance per run."""

    def __init__(self, K: int, config: CCPConfig | None = None, transport: str = "bulk"):
        self.K = int(K)
        self.cfg = config if config is not None else CCPConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.lp = GameLP(self.K, transport)
        self.pool: List[Dict] = []

    def _polish(self, Q, lam0):
        cfg = self.cfg
        lam = _project_simplex(lam0, self.K)
        G, phis = _phi_terms(Q, lam)
        phi = float(np.min(phis))
        for _ in range(cfg.T_max):
            tau = cfg.tau_rel * max(1.0, abs(phi))
            t_star, lam_next = self.lp.solve(2.0 * G - phis[:, None])
            if t_star - phi <= tau:
                break
            lam_next = _project_simplex(lam_next, self.K)
            G2, phis2 = _phi_terms(Q, lam_next)
            phi2 = float(np.min(phis2))
            if phi2 < phi - 1e-12 * max(1.0, abs(phi)):
                break                                  # numerical non-ascent: keep the better point
            lam, G, phis, phi = lam_next, G2, phis2, phi2
        return lam, phi, phis

    def _dedup(self, cands):
        cfg = self.cfg
        kept = []
        for cand in sorted(cands, key=lambda cn: -cn["phi"]):
            dup = False
            for k in kept:
                if float(np.abs(cand["lam"] - k["lam"]).sum()) <= cfg.dedup_l1_tol:
                    dup = True
                    break
                if (cand["active"] == k["active"]
                        and abs(cand["phi"] - k["phi"]) <= cfg.dedup_phi_rel * max(1.0, abs(k["phi"]))):
                    dup = True
                    break
            if not dup:
                kept.append(cand)
        return kept

    def solve(self, Q: np.ndarray) -> Tuple[float, np.ndarray]:
        """(phi(lam), lam) of the best point found; phi(lam) is a lower bound of max phi."""
        cfg, K = self.cfg, self.K
        Q = np.ascontiguousarray(np.asarray(Q, dtype=float))

        # 1. sandwich
        A = np.diagonal(Q, axis1=1, axis2=2)
        vertex_vals = A.min(axis=0)
        k_star = int(np.argmax(vertex_vals))
        lower = float(vertex_vals[k_star])
        valA, lam_A = self.lp.solve(A)
        lam_A = _project_simplex(lam_A, K)
        if valA <= lower * (1.0 + 1e-10) + 1e-14:
            lam = np.zeros(K)
            lam[k_star] = 1.0
            self.pool = [{"lam": lam, "phi": lower, "active": _active_set(_phi_terms(Q, lam)[1], cfg.active_tol)}]
            return lower, lam

        # 2. seeds
        fresh = sample_simplex_exp(int(cfg.N0), K, self.rng)
        pool_lams = np.array([p["lam"] for p in self.pool]) if self.pool else np.zeros((0, K))
        seeds = np.vstack([np.eye(K), lam_A[None, :], pool_lams, fresh])

        # 3. screening
        order = np.argsort(-phi_batch(Q, seeds))
        kept_idx: List[int] = []
        for pos in order:
            if all(float(np.abs(seeds[pos] - seeds[q]).sum()) > cfg.screen_sep_l1 for q in kept_idx):
                kept_idx.append(int(pos))
                if len(kept_idx) >= cfg.r:
                    break
        if len(kept_idx) < cfg.r:
            for pos in order:
                if int(pos) not in kept_idx:
                    kept_idx.append(int(pos))
                    if len(kept_idx) >= cfg.r:
                        break

        # 4. polish
        results = []
        for q in kept_idx:
            lam, phi, phis = self._polish(Q, seeds[q])
            results.append({"lam": lam, "phi": phi, "active": _active_set(phis, cfg.active_tol)})
        winner = max(results, key=lambda cn: cn["phi"])

        # 5. pool (old entries re-scored on the current bundle)
        cands = list(results)
        for p in self.pool:
            phis_p = _phi_terms(Q, p["lam"])[1]
            cands.append({"lam": p["lam"], "phi": float(np.min(phis_p)), "active": _active_set(phis_p, cfg.active_tol)})
        cap = cfg.pool_cap_factor * cfg.r
        new_pool = self._dedup(cands)[:cap]
        if not any(cn is winner for cn in new_pool):
            new_pool = [winner] + new_pool[:cap - 1]
        self.pool = new_pool
        return float(winner["phi"]), winner["lam"].copy()
