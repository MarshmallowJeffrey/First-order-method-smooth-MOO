"""ccp_lambda_solver.py — Multistart convex–concave procedure (CCP) for the
bundle GNS criterion.

NEW FILE (Aug 9, 2026).  Design record: Note/Aug_8_note.md.  Source:
Reference_essay/gns-ccp.pdf (Algorithm 1).  No existing file is modified.

The problem solved here is the λ-search of the adaptive bundle method,

    phi_i(lam) = lam^T Q_i lam,      phi(lam) = min_i phi_i(lam),
    GNS        = max_{lam in Delta_K} phi(lam),

on a precomputed Gram stack ``Q`` of shape (m, K, K) with
``Q[i] = J_i J_i^T`` — exactly ``BundleFast.gram_stack()`` (the paper's
Q_i).  ``Q`` never contains the per-CCP-iterate LP payoff M^(c); that
matrix is named ``Mc`` throughout and is rebuilt at every iteration.

Method (paper Algorithm 1 + Note/Aug_8_note.md extensions):

1.  Sandwich (Prop. 6): A_ik = [Q_i]_kk;  max_k min_i A_ik <= GNS <=
    val(A).  If the sandwich closes, the optimal vertex is returned
    exactly.  Otherwise the game maximiser lam_A becomes a seed.
2.  Seeds: K vertices + lam_A + the carried pool of previous-round
    local maximisers + N_new random points (Exp(1)-normalised or
    scrambled Sobol pushed onto the simplex).
3.  Screening: phi on all seeds in one matmul; retain the r best
    well-separated seeds (greedy by phi, l1 separation).
4.  Polish each retained seed with CCP: linearise at lam_c,

        Mc[i, k] = 2 (Q_i lam_c)_k - lam_c^T Q_i lam_c,

    solve the epigraph matrix-game LP  max{t : Mc lam >= t 1,
    lam in Delta_K}  (HiGHS, warm-started dual simplex; scipy fallback),
    step to the LP maximiser.  Stop when the predicted improvement
    delta_c = t* - phi(lam_c) <= tau, with

        tau = min(tau_rel * max(1, phi(lam_c)), tau_eps_frac * epsilon).

5.  Pool update: deduplicate (l1 distance, tolerant active set,
    phi proximity; greedy keep-best, prefer under-merging), cap at
    pool_cap_factor * r, always keep the winner.  The pool (lambdas
    only) seeds the next round; phi values are re-screened under the
    new bundle every round.
6.  Optional adaptive N_new schedule (rho = h/r rule), default OFF.

The solver is bundle-agnostic: it sees only ``Q``.  Cross-round state
(pool, N_new, LP model, Sobol stream) lives in ``CCPLambdaSolver``;
``solve`` has the same output contract as ``_maximise_GN_fast``:
``(pc_val, lam)`` with ``pc_val = phi(lam)`` scored at the feasible
projection.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog as _sp_linprog
from scipy.stats import qmc as _qmc

# Optional HiGHS backend (warm-started dual simplex).  Guarded like the
# cyipopt import in the algorithm modules: the module loads without it
# and _GameLP falls back to scipy's (cold) HiGHS wrapper.
try:
    import highspy as _highspy
    _HAS_HIGHSPY = True
    _HIGHSPY_IMPORT_ERROR: Optional[BaseException] = None
except (ImportError, OSError) as exc:  # pragma: no cover - environment guard
    _highspy = None
    _HAS_HIGHSPY = False
    _HIGHSPY_IMPORT_ERROR = exc


def highspy_available() -> bool:
    """Return whether the optional highspy/HiGHS backend is available."""
    return _HAS_HIGHSPY


# =====================================================================
#  Simplex samplers
# =====================================================================
def sample_simplex_exp(n: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """n i.i.d. uniform draws on Delta_K via normalised Exp(1) vectors."""
    if n <= 0:
        return np.zeros((0, K))
    E = rng.standard_exponential(size=(n, K))
    s = E.sum(axis=1, keepdims=True)
    s[s <= 0.0] = 1.0  # unreachable in practice; keeps the map total
    return E / s


def sample_simplex_sobol(n: int, K: int, engine: "_qmc.Sobol") -> np.ndarray:
    """n low-discrepancy points on Delta_K: scrambled Sobol in
    [0,1]^{K-1}, sorted, gaps taken (exact uniformity, even coverage).

    The engine is stateful: successive calls continue the sequence, so
    cross-round draws never repeat points.
    """
    if n <= 0:
        return np.zeros((0, K))
    if K == 1:
        return np.ones((n, 1))
    with warnings.catch_warnings():
        # scipy warns when n is not a power of two; balance is not
        # load-bearing here (seeds are screened, not integrated over).
        warnings.simplefilter("ignore", UserWarning)
        U = engine.random(n)
    U = np.sort(U, axis=1)
    P = np.empty((n, K))
    P[:, 0] = U[:, 0]
    if K > 2:
        P[:, 1:K - 1] = np.diff(U, axis=1)
    P[:, K - 1] = 1.0 - U[:, -1]
    return P


# =====================================================================
#  phi evaluation on the Gram stack
# =====================================================================
def phi_batch(Q: np.ndarray, lams: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """phi(lam) = min_i lam^T Q_i lam for a batch of lambdas.

    Implemented as one (n, K^2) x (K^2, m) matmul so that screening
    thousands of seeds is a single BLAS call; memory is O(n*m).

    Returns
    -------
    phis   : (n,) envelope values
    argmin : (n,) index of an active bundle point per lambda
    """
    Q = np.asarray(Q, dtype=float)
    lams = np.asarray(lams, dtype=float)
    m, K = Q.shape[0], Q.shape[1]
    n = lams.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    outer = (lams[:, :, None] * lams[:, None, :]).reshape(n, K * K)
    vals = outer @ Q.reshape(m, K * K).T          # (n, m)
    idx = np.argmin(vals, axis=1)
    return vals[np.arange(n), idx], idx


def _phi_terms(Q: np.ndarray, lam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-bundle-point pieces at one lambda: G[i] = Q_i lam  (m, K) and
    phis[i] = lam^T Q_i lam  (m,)."""
    G = Q @ lam                    # (m, K)
    phis = G @ lam                 # (m,)
    return G, phis


def _project_simplex(lam: np.ndarray, K: int) -> np.ndarray:
    """Clip to lam >= 0 and renormalise to sum 1 (centroid on degenerate
    input).  phi is homogeneous of degree 2, so every candidate must be
    scored at its feasible projection (same rationale as the IPOPT
    path's projector)."""
    lam = np.maximum(np.asarray(lam, dtype=float), 0.0)
    s = float(lam.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.full(K, 1.0 / K)
    return lam / s


def _active_set(phis: np.ndarray, tol: float) -> frozenset:
    """Tolerant active set A_eta(lam) = {i : phi_i <= phi + eta*max(1, phi)}."""
    lo = float(np.min(phis))
    return frozenset(np.nonzero(phis <= lo + tol * max(1.0, abs(lo)))[0].tolist())


# =====================================================================
#  The inner LP:  max{ t : M lam >= t 1, 1^T lam = 1, lam >= 0 }
# =====================================================================
class _GameLP:
    """Persistent epigraph matrix-game LP with HiGHS warm start.

    Columns are (lam_1..lam_K, t); rows are the m payoff rows plus the
    simplex equality.  Consecutive calls with same-shape payoffs rewrite
    coefficients in place (``changeCoeff``) so the dual simplex restarts
    from the previous basis; a grown payoff appends rows (basis stays
    valid); a shrunk payoff rebuilds.  Without highspy every call is a
    cold scipy ``linprog(method="highs")`` solve — same optima, no warm
    start.
    """

    def __init__(self, K: int, use_highspy: Optional[bool] = None):
        self.K = int(K)
        self.m = 0
        if use_highspy is None:
            self.use_highspy = _HAS_HIGHSPY
        else:
            self.use_highspy = bool(use_highspy) and _HAS_HIGHSPY
        self._h = None
        self.simplex_iters = 0     # cumulative across resolves
        self.n_solves = 0
        self._warned_status = False

    # -- highspy path --------------------------------------------------
    def _build(self, M: np.ndarray) -> None:
        m, K = M.shape
        INF = _highspy.kHighsInf
        h = _highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", "off")     # keep the basis usable
        h.setOptionValue("solver", "simplex")
        h.addCols(K + 1,
                  np.r_[np.zeros(K), -1.0],      # minimise -t
                  np.r_[np.zeros(K), -INF],      # lam >= 0, t free
                  np.full(K + 1, INF),
                  0, [], [], [])
        idx = np.r_[np.arange(K), K].astype(np.int32)
        for i in range(m):
            h.addRow(0.0, INF, K + 1, idx, np.r_[M[i], -1.0])
        h.addRow(1.0, 1.0, K, np.arange(K, dtype=np.int32), np.ones(K))
        self._h, self.m = h, m

    def _rewrite(self, M: np.ndarray) -> None:
        # Same shape as the stored model: rewrite the payoff block in
        # place (rows 0..m-1; the simplex equality is row m and is never
        # touched), keeping the basis for a warm dual-simplex restart.
        m, K = M.shape
        h = self._h
        for i in range(m):
            row = M[i]
            for k in range(K):
                h.changeCoeff(i, k, float(row[k]))

    def resolve(self, M: np.ndarray) -> Tuple[float, np.ndarray]:
        """Solve for payoff M (m, K); returns (t_star, lam_star)."""
        M = np.asarray(M, dtype=float)
        m, K = M.shape
        if K != self.K:
            raise ValueError(f"payoff has K={K}, expected {self.K}.")
        if not self.use_highspy:
            return self._resolve_scipy(M)
        if self._h is None or m != self.m:
            # Any row-count change (bundle grew or shrank between outer
            # rounds) rebuilds the model: appending after the trailing
            # simplex-equality row would scramble the row layout that
            # ``_rewrite`` relies on.  Rebuilds happen at most once per
            # outer round; the warm path serves every CCP iterate within
            # the round, which is where the LP volume is.
            self._build(M)
        else:
            self._rewrite(M)
        self._h.run()
        self.n_solves += 1
        status = self._h.getModelStatus()
        if status != _highspy.HighsModelStatus.kOptimal:
            if not self._warned_status:
                warnings.warn(
                    f"HiGHS returned status {status!r} on the game LP; "
                    "falling back to scipy for this solve.",
                    RuntimeWarning, stacklevel=2,
                )
                self._warned_status = True
            return self._resolve_scipy(M)
        self.simplex_iters += int(self._h.getInfo().simplex_iteration_count)
        x = np.asarray(self._h.getSolution().col_value, dtype=float)
        return float(x[K]), x[:K].copy()

    # -- scipy fallback ------------------------------------------------
    def _resolve_scipy(self, M: np.ndarray) -> Tuple[float, np.ndarray]:
        m, K = M.shape
        c = np.r_[np.zeros(K), -1.0]
        A_ub = np.c_[-M, np.ones(m)]               # t - (M lam)_i <= 0
        res = _sp_linprog(
            c, A_ub=A_ub, b_ub=np.zeros(m),
            A_eq=np.r_[np.ones(K), 0.0].reshape(1, -1), b_eq=[1.0],
            bounds=[(0.0, None)] * K + [(None, None)],
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"game LP failed in scipy fallback: {res.message}")
        self.n_solves += 1
        x = np.asarray(res.x, dtype=float)
        return float(x[K]), x[:K].copy()


# =====================================================================
#  Exact K = 2 envelope (test oracle / gold standard)
# =====================================================================
def exact_gns_K2(Q: np.ndarray) -> Tuple[float, np.ndarray]:
    """Exact GNS for K = 2 via the parabola lower envelope.

    With lam = (1-s, s), each phi_i(s) = alpha_i s^2 + beta_i s +
    gamma_i is a convex parabola (alpha_i = ||g_i1 - g_i2||^2 >= 0), so
    every local maximiser of the envelope is an endpoint of [0, 1] or a
    crossing of two parabolas (paper Sec. 6).  Enumerating all O(m^2)
    pairwise crossings is exact and adequate for test-oracle use.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 3 or Q.shape[1:] != (2, 2):
        raise ValueError(f"exact_gns_K2 expects (m, 2, 2); got {Q.shape}.")
    a, b, c = Q[:, 0, 0], Q[:, 0, 1], Q[:, 1, 1]
    alpha, beta, gamma = a - 2.0 * b + c, 2.0 * (b - a), a
    scale = max(1.0, float(np.max(np.abs(np.c_[alpha, beta, gamma]))))
    cands = [0.0, 1.0]
    mm = Q.shape[0]
    for i in range(mm):
        for j in range(i + 1, mm):
            da = alpha[i] - alpha[j]
            db = beta[i] - beta[j]
            dg = gamma[i] - gamma[j]
            if abs(da) > 1e-14 * scale:
                disc = db * db - 4.0 * da * dg
                if disc < 0.0:
                    continue
                rt = np.sqrt(disc)
                roots = ((-db - rt) / (2.0 * da), (-db + rt) / (2.0 * da))
            elif abs(db) > 1e-14 * scale:
                roots = (-dg / db,)
            else:
                continue
            cands.extend(s for s in roots if 0.0 <= s <= 1.0)
    s = np.asarray(cands)
    vals = np.min(alpha[None, :] * s[:, None] ** 2
                  + beta[None, :] * s[:, None] + gamma[None, :], axis=1)
    best = int(np.argmax(vals))
    s_star = float(s[best])
    return float(vals[best]), np.array([1.0 - s_star, s_star])


# =====================================================================
#  Configuration and solver
# =====================================================================
@dataclass
class CCPConfig:
    """Knobs for the multistart CCP lambda-search (Note/Aug_8_note.md)."""
    N0: int = 2000                       # random seeds per round (static mode)
    r: int = 10                          # restarts polished per round
    pool_cap_factor: int = 3             # pool cap = factor * r
    tau_rel: float = 1e-8                # relative stationarity tolerance
    tau_eps_frac: float = 0.01           # safety cap 0.01*epsilon (rarely binds)
    T_max: int = 100                     # CCP iteration cap per restart
    seed_sampler: str = "exp"            # "exp" | "sobol" (default per Aug 9 smoke test)
    adaptive_seed_schedule: bool = False # rho-rule OFF by default (ablation switch)
    n_new_floor_factor: int = 10         # shrink floor = factor * r
    rho_low: float = 0.25                # schedule band edge
    screen_sep_l1: float = 0.05          # l1 separation among retained seeds
    dedup_l1_tol: float = 1e-3           # pool dedup: same point if closer
    dedup_phi_rel: float = 1e-9          # pool dedup: phi proximity (rel)
    active_tol: float = 1e-9             # tolerant active-set threshold (rel)
    collapse_frac: float = 0.5           # pool-collapse trigger fraction
    seed: int = 0                        # rng / Sobol scramble seed
    use_highspy: Optional[bool] = None   # None = auto-detect


class CCPLambdaSolver:
    """Cross-round stateful multistart CCP solver.

    One instance per algorithm run.  ``solve(Q, epsilon)`` returns
    ``(pc_val, lam)`` exactly like ``_maximise_GN_fast``; per-round
    telemetry lands in ``stats_last`` / ``stats_history``.
    """

    def __init__(self, K: int, config: Optional[CCPConfig] = None):
        if not isinstance(K, (int, np.integer)) or isinstance(K, bool) or K < 1:
            raise ValueError(f"K must be a positive integer; got {K!r}.")
        self.K = int(K)
        self.cfg = config if config is not None else CCPConfig()
        if self.cfg.seed_sampler not in {"exp", "sobol"}:
            raise ValueError("seed_sampler must be 'exp' or 'sobol'.")
        if self.cfg.r < 1 or self.cfg.N0 < 1:
            raise ValueError("N0 and r must be positive.")
        self.rng = np.random.default_rng(self.cfg.seed)
        self._sobol: Optional[_qmc.Sobol] = None
        if self.cfg.seed_sampler == "sobol" and self.K >= 2:
            self._sobol = _qmc.Sobol(d=self.K - 1, scramble=True,
                                     seed=self.cfg.seed)
        self.lp = _GameLP(self.K, use_highspy=self.cfg.use_highspy)
        # cross-round state
        self.pool: List[Dict] = []       # dicts: lam, phi (last screen), active
        self.n_new = int(self.cfg.N0)
        self.round_index = 0
        self.last_winner_phi: Optional[float] = None
        self._zero_rho_streak = 0
        self.stats_last: Dict = {}
        self.stats_history: List[Dict] = []

    # -- pieces --------------------------------------------------------
    def _sample(self, n: int) -> np.ndarray:
        if self.cfg.seed_sampler == "sobol" and self._sobol is not None:
            return sample_simplex_sobol(n, self.K, self._sobol)
        return sample_simplex_exp(n, self.K, self.rng)

    def _polish(self, Q: np.ndarray, lam0: np.ndarray,
                epsilon: Optional[float],
                trace: Optional[List[float]] = None
                ) -> Tuple[np.ndarray, float, np.ndarray, int, float]:
        """One CCP ascent from lam0.  Returns (lam, phi, phis, iters, delta)."""
        cfg = self.cfg
        lam = _project_simplex(lam0, self.K)
        G, phis = _phi_terms(Q, lam)
        phi = float(np.min(phis))
        delta = np.inf
        iters = 0
        for _ in range(cfg.T_max):
            tau = cfg.tau_rel * max(1.0, abs(phi))
            if epsilon is not None:
                tau = min(tau, cfg.tau_eps_frac * float(epsilon))
            Mc = 2.0 * G - phis[:, None]
            t_star, lam_next = self.lp.resolve(Mc)
            iters += 1
            delta = t_star - phi
            if trace is not None:
                trace.append(phi)
            if delta <= tau:
                break
            lam_next = _project_simplex(lam_next, self.K)
            G2, phis2 = _phi_terms(Q, lam_next)
            phi2 = float(np.min(phis2))
            if phi2 < phi - 1e-12 * max(1.0, abs(phi)):
                # Numerical non-ascent (theory forbids it): keep the
                # better point rather than lose ground.
                break
            lam, G, phis, phi = lam_next, G2, phis2, phi2
        return lam, phi, phis, iters, float(delta)

    def _dedup(self, cands: List[Dict]) -> List[Dict]:
        """Greedy keep-best dedup: same candidate iff l1-close, OR equal
        active set with matching phi.  Under-merging is the cheap error."""
        cfg = self.cfg
        kept: List[Dict] = []
        for cand in sorted(cands, key=lambda cn: -cn["phi"]):
            dup = False
            for k in kept:
                if float(np.abs(cand["lam"] - k["lam"]).sum()) <= cfg.dedup_l1_tol:
                    dup = True
                    break
                if (cand["active"] == k["active"]
                        and abs(cand["phi"] - k["phi"])
                        <= cfg.dedup_phi_rel * max(1.0, abs(k["phi"]))):
                    dup = True
                    break
            if not dup:
                kept.append(cand)
        return kept

    def _update_seed_schedule(self, rho: float, collapse: bool) -> None:
        """The rho = h/r controller (runs only when the switch is ON and
        a previous round exists).  Shrink needs two consecutive
        zero-rho rounds; expansion fires immediately; collapse resets."""
        cfg = self.cfg
        if collapse:
            self.n_new = int(cfg.N0)
            self._zero_rho_streak = 0
            return
        if rho == 0.0:
            self._zero_rho_streak += 1
            if self._zero_rho_streak >= 2:
                self.n_new = max(cfg.n_new_floor_factor * cfg.r,
                                 self.n_new // 2)
                self._zero_rho_streak = 0
        elif rho <= cfg.rho_low:
            self._zero_rho_streak = 0
        else:
            self._zero_rho_streak = 0
            self.n_new = min(int(cfg.N0), 2 * self.n_new)

    # -- main entry ----------------------------------------------------
    def solve(self, Q: np.ndarray, epsilon: Optional[float] = None
              ) -> Tuple[float, np.ndarray]:
        """argmax_{lam in Delta_K} min_i lam^T Q_i lam  (heuristic-global).

        Returns ``(pc_val, lam)``; ``pc_val = phi(lam)`` at the feasible
        projection, a valid lower bound on GNS.
        """
        t0 = time.perf_counter()
        cfg = self.cfg
        Q = np.ascontiguousarray(np.asarray(Q, dtype=float))
        if Q.ndim != 3 or Q.shape[1] != self.K or Q.shape[2] != self.K:
            raise ValueError(f"Q must have shape (m, {self.K}, {self.K}); "
                             f"got {Q.shape}.")
        m = Q.shape[0]
        if m == 0:
            raise ValueError("Cannot maximise GNS for an empty bundle.")
        K = self.K
        if K == 1:
            lam = np.ones(1)
            phi = float(np.min(Q[:, 0, 0]))
            self._record(m, phi, lam, sandwich_closed=True, rho=0.0,
                         n_new_used=0, retained=0, ccp_iters=0,
                         n_distinct=1, n_dropped=0, t0=t0)
            self.round_index += 1
            self.last_winner_phi = phi
            return phi, lam

        lp_iters0 = self.lp.simplex_iters

        # ---- 1. sandwich (paper Prop. 6 / Algorithm 1 lines 1-3) ----
        A = np.diagonal(Q, axis1=1, axis2=2)      # (m, K), A_ik = [Q_i]_kk
        vertex_vals = A.min(axis=0)               # phi(e_k) = min_i A_ik
        k_star = int(np.argmax(vertex_vals))
        lower = float(vertex_vals[k_star])
        valA, lam_A = self.lp.resolve(A)
        lam_A = _project_simplex(lam_A, K)
        if valA <= lower * (1.0 + 1e-10) + 1e-14:
            lam = np.zeros(K)
            lam[k_star] = 1.0
            self.pool = [{"lam": lam, "phi": lower,
                          "active": _active_set(_phi_terms(Q, lam)[1],
                                                cfg.active_tol),
                          "origin": "vertex"}]
            self._record(m, lower, lam, sandwich_closed=True, rho=0.0,
                         n_new_used=0, retained=0,
                         ccp_iters=0, n_distinct=1, n_dropped=0, t0=t0,
                         val_A=float(valA), lower=lower,
                         lp_iters0=lp_iters0)
            self.round_index += 1
            self.last_winner_phi = lower
            return lower, lam

        # ---- 2. seed set --------------------------------------------
        n_new_used = self.n_new
        fresh = self._sample(n_new_used)
        pool_lams = (np.array([p["lam"] for p in self.pool])
                     if self.pool else np.zeros((0, K)))
        seeds = np.vstack([np.eye(K), lam_A[None, :], pool_lams, fresh])
        # origin codes: 0 = structured (vertices, lam_A), 1 = pool, 2 = fresh
        origin = np.concatenate([
            np.zeros(K + 1, dtype=int),
            np.ones(len(pool_lams), dtype=int),
            np.full(len(fresh), 2, dtype=int),
        ])

        # ---- 3. screening -------------------------------------------
        phis_all, _ = phi_batch(Q, seeds)
        order = np.argsort(-phis_all)
        retained_idx: List[int] = []
        for pos in order:
            lam_c = seeds[pos]
            if all(float(np.abs(lam_c - seeds[q]).sum()) > cfg.screen_sep_l1
                   for q in retained_idx):
                retained_idx.append(int(pos))
                if len(retained_idx) >= cfg.r:
                    break
        if len(retained_idx) < cfg.r:      # separation exhausted the list
            for pos in order:
                if int(pos) not in retained_idx:
                    retained_idx.append(int(pos))
                    if len(retained_idx) >= cfg.r:
                        break
        pool_retained = sum(1 for q in retained_idx if origin[q] == 1)
        h_fresh = sum(1 for q in retained_idx if origin[q] == 2)
        rho = h_fresh / max(1, len(retained_idx))

        # ---- 4. CCP polish ------------------------------------------
        results: List[Dict] = []
        ccp_iters = 0
        for q in retained_idx:
            lam, phi, phis_at, iters, delta = self._polish(
                Q, seeds[q], epsilon)
            ccp_iters += iters
            results.append({"lam": lam, "phi": phi,
                            "active": _active_set(phis_at, cfg.active_tol),
                            "origin": "restart", "delta": delta})
        winner = max(results, key=lambda cn: cn["phi"])

        # ---- 5. pool update -----------------------------------------
        cands = list(results)
        for j, p in enumerate(self.pool):
            # carried entries keep their lambda; phi re-screened under
            # the CURRENT bundle (old values are stale across bundles)
            lam_p = p["lam"]
            phis_p = _phi_terms(Q, lam_p)[1]
            cands.append({"lam": lam_p, "phi": float(np.min(phis_p)),
                          "active": _active_set(phis_p, cfg.active_tol),
                          "origin": "pool"})
        kept = self._dedup(cands)
        n_distinct = len(kept)
        pool_cap = cfg.pool_cap_factor * cfg.r
        new_pool = kept[:pool_cap]
        if not any(cn is winner for cn in new_pool):
            new_pool = [winner] + new_pool[:pool_cap - 1]
        n_dropped = n_distinct - len(new_pool)
        self.pool = new_pool

        # ---- 6. adaptive schedule (switch, default OFF) -------------
        collapse = False
        if self.round_index >= 1 and len(pool_lams) > 0:
            old_pool_best = max(
                (cn["phi"] for cn in cands if cn["origin"] == "pool"),
                default=-np.inf)
            collapse = (pool_retained == 0
                        or (self.last_winner_phi is not None
                            and old_pool_best
                            < cfg.collapse_frac * self.last_winner_phi))
        if cfg.adaptive_seed_schedule and self.round_index >= 1:
            self._update_seed_schedule(rho, collapse)

        self.last_winner_phi = float(winner["phi"])
        self._record(m, float(winner["phi"]), winner["lam"],
                     sandwich_closed=False, rho=rho,
                     n_new_used=n_new_used, retained=len(retained_idx),
                     ccp_iters=ccp_iters, n_distinct=n_distinct,
                     n_dropped=n_dropped, t0=t0, val_A=float(valA),
                     lower=lower, collapse=collapse, lp_iters0=lp_iters0)
        self.round_index += 1
        return float(winner["phi"]), winner["lam"].copy()

    # -- telemetry -----------------------------------------------------
    def _record(self, m: int, phi_best: float, lam: np.ndarray, *,
                sandwich_closed: bool, rho: float, n_new_used: int,
                retained: int, ccp_iters: int, n_distinct: int,
                n_dropped: int, t0: float,
                val_A: Optional[float] = None,
                lower: Optional[float] = None,
                collapse: bool = False,
                lp_iters0: Optional[int] = None) -> None:
        cfg = self.cfg
        self.stats_last = {
            "round": self.round_index,
            "m": int(m),
            "phi_best": float(phi_best),
            "lam": np.asarray(lam, dtype=float).copy(),
            "sandwich_closed": bool(sandwich_closed),
            "val_A": val_A,
            "lower_vertex": lower,
            "rho": float(rho),
            "n_new_used": int(n_new_used),
            "n_new_next": int(self.n_new),
            "n_restarts": int(retained),
            "ccp_iters": int(ccp_iters),
            "lp_simplex_iters": (None if lp_iters0 is None
                                 else int(self.lp.simplex_iters - lp_iters0)),
            "pool_size": len(self.pool),
            "pool_cap": cfg.pool_cap_factor * cfg.r,
            "n_distinct_before_cap": int(n_distinct),
            "n_dropped_by_cap": int(n_dropped),
            "pool_collapse": bool(collapse),
            "backend": "highspy" if self.lp.use_highspy else "scipy",
            "lambda_search_wall_time": time.perf_counter() - t0,
        }
        self.stats_history.append(self.stats_last)
