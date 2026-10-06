"""Step 1 of GRAB for K = 3: the multistart CCP of Algorithm 2 (Appendix A.4), every game LP solved by constraint
generation.  It is the lambda-search of the adaptive method for K = 3 (selector "ccp_cg"); abm/ccp.py holds the
multistart CCP of the K = 3 audits.

For a bundle with Gram matrices Q_i (m x K x K), phi_i(lam) = lam^T Q_i lam and phi(lam) = min_i phi_i(lam).  One call
of ``solve``, step by step as in Algorithm 2:

1. A_ik = [Q_i]_kk and the value val(A) of the zero-sum game A, with a maximizer lam_A (Proposition 10).  With a
   tolerance eps, val(A) <= eps^2 is a valid upper bound certificate and the call ends (``certified``; it returns
   lam = None).  The runs set no eps: GRAB runs until the budget is spent.
2. Seeds: the K vertices, lam_A, N points drawn uniformly from Delta_K and the J best local maximizers of the
   previous call.
3. phi on all seeds in one batched contraction; the r best seeds that are more than sep_l1 apart (l1) are retained.
4. CCP from each retained seed: at lam_c form M^c (M_ik = 2 (Q_i lam_c)_k - phi_i(lam_c)), solve the LP
   max{t : M^c lam >= t 1, lam in Delta_K} and move to its maximizer lam_{c+1}; stop when the predicted improvement
   delta_c = val(M^c) - phi(lam_c) is at most tau = tau_rel max{1, phi(lam_c)} (Appendix A.2), or after T LPs.
5. Return the recorded point of largest phi.  The recorded points without duplicates are the local maximizers that
   seed the next call.

Choices that Algorithm 2 leaves open: the retained seeds are more than sep_l1 apart in l1; the next call is seeded
with all distinct recorded points of this call (J = None); a CCP run ends at the maximizer of its last LP.  By Lemma 2
phi never decreases along a CCP run; ``stats["descents"]`` counts numerical exceptions.

Constraint generation.  The LP has K + 1 variables and one row per bundle point; an optimal basis needs at most K + 1
of the rows.  HiGHS (dual simplex) solves it on a working set: the ``work`` rows smallest at a hint point (for a CCP
step lam_c, where row i of M^c equals phi_i(lam_c); for the game A the previous lam_A) and the smallest row of each
column.  The rows that the working-set solution violates by more than cg_tol max{1, |t|} join the working set, the
``add`` most violated at a time, and HiGHS continues from its basis, until no row is violated.  The working-set value
is an upper bound of the value of the full LP and the returned lam attains it up to that tolerance, so the result
is an optimal solution of the full LP.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Optional

import highspy
import numpy as np

from .ccp import GameLP, _active_set, _project_simplex, _scipy_game, sample_simplex_exp

SCREEN_BLOCK = 2048          # bundle rows per block of the batched contraction


@dataclass
class CCPCGConfig:
    N: int = 2000                  # seeds drawn uniformly from Delta_K per call
    r: int = 10                    # retained seeds per call (each runs CCP)
    J: Optional[int] = None        # local maximizers of the previous call used as seeds (None: all of them, <= r)
    tau_rel: float = 1e-8          # a CCP run stops when delta_c <= tau_rel max{1, phi(lam_c)} ...
    T: int = 100                   # ... or after T LPs
    sep_l1: float = 0.05           # retained seeds are more than this apart (l1)
    dedup_l1_tol: float = 1e-3     # two recorded points are the same local maximizer if l1-closer than this ...
    dedup_phi_rel: float = 1e-9    # ... or with the same active set and phi within this relative distance
    active_tol: float = 1e-9       # active set: phi_i within this relative distance of phi
    eps: Optional[float] = None    # certificate test val(A) <= eps^2 (None: no test)
    lp: str = "cg"                 # "cg": constraint generation; "full": the whole LP (abm.ccp.GameLP), for checks
    work: int = 200                # constraint generation: rows of the first working set ...
    add: int = 200                 # ... rows added per round at most
    cg_tol: float = 1e-12          # ... violation tolerance, relative to max{1, |t|}
    seed: int = 0                  # random stream of the uniform seeds


class CGGameLP:
    """max{t : M lam >= t 1, sum(lam) = 1, lam >= 0} by constraint generation (see the module docstring)."""

    def __init__(self, K: int, work: int = 200, add: int = 200, tol: float = 1e-12):
        self.K, self.work, self.add, self.tol = int(K), int(work), int(add), float(tol)
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("presolve", "off")
        self.h.setOptionValue("solver", "simplex")
        self.solves = 0          # LPs solved
        self.rounds = 0          # working-set solves (one per round)
        self.max_rows = 0        # largest working set
        self.fallbacks = 0       # LPs passed to scipy because HiGHS did not report an optimum
        self._warned = False
        self._cols = np.arange(self.K + 1, dtype=np.int32)

    def _rows(self, Mw):
        """Row-wise coefficients of the payoff rows M_i lam - t >= 0."""
        n = Mw.shape[0]
        values = np.hstack([Mw, -np.ones((n, 1))]).ravel()
        index = np.tile(self._cols, n)
        start = np.arange(0, n * (self.K + 1), self.K + 1, dtype=np.int32)
        return values, index, start

    def _model(self, Mw):
        n, K = Mw.shape
        INF = highspy.kHighsInf
        lp = highspy.HighsLp()
        lp.num_col_ = K + 1
        lp.num_row_ = n + 1
        lp.col_cost_ = np.r_[np.zeros(K), -1.0]
        lp.col_lower_ = np.r_[np.zeros(K), -INF]
        lp.col_upper_ = np.full(K + 1, INF)
        lp.row_lower_ = np.r_[np.zeros(n), 1.0]
        lp.row_upper_ = np.r_[np.full(n, INF), 1.0]
        values, index, start = self._rows(Mw)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.value_ = np.concatenate([values, np.ones(K)])
        lp.a_matrix_.index_ = np.concatenate([index, np.arange(K, dtype=np.int32)])
        lp.a_matrix_.start_ = np.concatenate([start, [n * (K + 1), n * (K + 1) + K]]).astype(np.int32)
        return lp

    def _fallback(self, M):
        if not self._warned:
            warnings.warn("HiGHS did not report an optimal game LP; using scipy on the full LP for this solve.",
                          RuntimeWarning)
            self._warned = True
        self.fallbacks += 1
        return _scipy_game(M)

    def solve(self, M: np.ndarray, hint: np.ndarray):
        """Returns (t*, lam*).  hint[i]: row i at a hint point; the first working set holds the smallest rows."""
        M = np.ascontiguousarray(M, dtype=float)
        m, K = M.shape
        self.solves += 1
        if m <= self.work + K:
            W = np.arange(m)
        else:
            W = np.union1d(np.argpartition(hint, self.work)[:self.work], np.argmin(M, axis=0))
        in_w = np.zeros(m, dtype=bool)
        in_w[W] = True
        h = self.h
        h.passModel(self._model(M[W]))
        h.run()
        while True:
            self.rounds += 1
            if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                return self._fallback(M)
            x = np.asarray(h.getSolution().col_value, dtype=float)
            lam, t = x[:K], float(x[K])
            slack = M @ lam - t
            bad = np.nonzero((slack < -self.tol * max(1.0, abs(t))) & ~in_w)[0]
            if bad.size == 0:
                break
            if bad.size > self.add:
                bad = bad[np.argpartition(slack[bad], self.add)[:self.add]]
            in_w[bad] = True
            values, index, start = self._rows(M[bad])
            h.addRows(bad.size, np.zeros(bad.size), np.full(bad.size, highspy.kHighsInf), values.size, start, index,
                      values)
            h.run()
        self.max_rows = max(self.max_rows, int(np.count_nonzero(in_w)))
        return t, lam.copy()


class CCPCGSelector:
    """Algorithm 2 with constraint generation; stateful across calls (bundle copy, local maximizers of the previous
    call, random stream, LP hint), one instance per run.  solve(grams) takes the bundle's Gram matrices (all of them,
    in bundle order; only the new ones are copied) and returns (phi(lam), lam), as EnvelopeSelector.solve does
    (abm/envelope.py)."""

    def __init__(self, K: int, config: CCPCGConfig | None = None):
        self.K = int(K)
        self.cfg = cfg = config if config is not None else CCPCGConfig()
        if cfg.lp not in ("cg", "full"):
            raise ValueError(f"lp {cfg.lp!r}")
        self.rng = np.random.default_rng(cfg.seed)
        self.lp = CGGameLP(self.K, cfg.work, cfg.add, cfg.cg_tol) if cfg.lp == "cg" else GameLP(self.K, "bulk")
        self.n = 0
        self.Q = np.empty((1024, self.K, self.K))
        self.A = np.empty((1024, self.K))                            # [Q_i]_kk
        self.iu = np.triu_indices(self.K)
        self.Qs = np.empty((1024, self.iu[0].size))                  # [Q_i]_kl, k <= l, off-diagonal doubled
        self._coef = np.where(self.iu[0] == self.iu[1], 1.0, 2.0)
        self.prev = []                                               # local maximizers of the previous call
        self.lam_A = None
        self.upper = None                                            # val(A) of the last call
        self.certified = False
        self.last = None                                             # details of the last call
        self.stats = {"calls": 0, "lps": 0, "capped": 0, "descents": 0,
                      "seconds_game": 0.0, "seconds_screen": 0.0, "seconds_ccp": 0.0}

    # -- bundle ------------------------------------------------------------------------------------------------
    def _update(self, grams):
        m = len(grams)
        if m < self.n:
            raise ValueError("the bundle only grows")
        if m > self.Q.shape[0]:
            size = max(m, 2 * self.Q.shape[0])
            for name in ("Q", "A", "Qs"):
                old = getattr(self, name)
                arr = np.empty((size,) + old.shape[1:])
                arr[:self.n] = old[:self.n]
                setattr(self, name, arr)
        if m > self.n:
            new = np.asarray(grams[self.n:m], dtype=float)
            self.Q[self.n:m] = new
            self.A[self.n:m] = np.diagonal(new, axis1=1, axis2=2)
            self.Qs[self.n:m] = new[:, self.iu[0], self.iu[1]] * self._coef
            self.n = m
        return m

    def _lp(self, M, hint):
        if self.cfg.lp == "cg":
            return self.lp.solve(M, hint)
        return self.lp.solve(M)

    def phi_seeds(self, seeds):
        """phi at every seed: one contraction with the bundle, in blocks of bundle rows."""
        P = np.ascontiguousarray((seeds[:, self.iu[0]] * seeds[:, self.iu[1]]).T)
        Qs = self.Qs[:self.n]
        out = np.full(seeds.shape[0], np.inf)
        for a in range(0, self.n, SCREEN_BLOCK):
            np.minimum(out, (Qs[a:a + SCREEN_BLOCK] @ P).min(axis=0), out=out)
        return out

    # -- one CCP run ----------------------------------------------------------------------------------------------
    def _ccp(self, Q2, m, lam0):
        cfg, K = self.cfg, self.K
        lam = _project_simplex(lam0, K)
        G = (Q2 @ lam).reshape(m, K)
        phis = G @ lam
        phi = float(phis.min())
        n_lp = 0
        while n_lp < cfg.T:
            tau = cfg.tau_rel * max(1.0, abs(phi))
            t, lam_lp = self._lp(2.0 * G - phis[:, None], phis)
            n_lp += 1
            delta = t - phi
            lam = _project_simplex(lam_lp, K)
            G = (Q2 @ lam).reshape(m, K)
            phis_new = G @ lam
            phi_new = float(phis_new.min())
            if phi_new < phi - 1e-12 * max(1.0, abs(phi)):
                self.stats["descents"] += 1
            phis, phi = phis_new, phi_new
            if delta <= tau:
                break
        else:
            self.stats["capped"] += 1
        return lam, phi, phis, n_lp

    def _dedup(self, cands):
        cfg = self.cfg
        kept = []
        for cand in sorted(cands, key=lambda cn: -cn["phi"]):
            if not any(float(np.abs(cand["lam"] - k["lam"]).sum()) <= cfg.dedup_l1_tol
                       or (cand["active"] == k["active"]
                           and abs(cand["phi"] - k["phi"]) <= cfg.dedup_phi_rel * max(1.0, abs(k["phi"])))
                       for k in kept):
                kept.append(cand)
        return kept

    # -- one call -------------------------------------------------------------------------------------------------
    def solve(self, grams):
        cfg, K = self.cfg, self.K
        m = self._update(grams)
        Q2 = self.Q[:m].reshape(m * K, K)
        st = self.stats
        st["calls"] += 1

        # 1. the game A
        t0 = time.perf_counter()
        A = self.A[:m]
        hint = self.lam_A if self.lam_A is not None else np.full(K, 1.0 / K)
        val_A, lam_A = self._lp(A, A @ hint)
        self.lam_A, self.upper = _project_simplex(lam_A, K), float(val_A)
        t1 = time.perf_counter()
        st["seconds_game"] += t1 - t0
        if cfg.eps is not None and val_A <= cfg.eps ** 2:
            self.certified = True
            self.last = {"m": m, "certified": True, "upper": float(val_A)}
            return float(val_A), None

        # 2. seeds and 3. screening
        fresh = sample_simplex_exp(int(cfg.N), K, self.rng)
        prev = np.array([p["lam"] for p in self.prev]).reshape(-1, K)
        seeds = np.vstack([np.eye(K), self.lam_A[None, :], prev, fresh])
        vals = self.phi_seeds(seeds)
        order = np.argsort(-vals, kind="stable")
        blocked = np.zeros(seeds.shape[0], dtype=bool)
        kept = []
        for pos in order:
            if blocked[pos]:
                continue
            kept.append(int(pos))
            if len(kept) >= cfg.r:
                break
            blocked |= np.abs(seeds - seeds[pos]).sum(axis=1) <= cfg.sep_l1
        t2 = time.perf_counter()
        st["seconds_screen"] += t2 - t1

        # 4. CCP from each retained seed
        results, lps = [], []
        for q in kept:
            lam, phi, phis, n_lp = self._ccp(Q2, m, seeds[q])
            results.append({"lam": lam, "phi": phi, "active": _active_set(phis, cfg.active_tol)})
            lps.append(n_lp)
        st["lps"] += sum(lps)
        st["seconds_ccp"] += time.perf_counter() - t2

        # 5. the point of largest phi; the local maximizers seed the next call
        winner = max(results, key=lambda cn: cn["phi"])
        distinct = self._dedup(results)
        self.prev = distinct if cfg.J is None else distinct[:int(cfg.J)]
        self.last = {"m": m, "upper": float(val_A), "lps": lps, "phi": float(winner["phi"]),
                     "best_seed_phi": float(vals[order[0]]), "maximizers": len(distinct)}
        return float(winner["phi"]), winner["lam"].copy()
