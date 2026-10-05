"""Preference-vector selection: approximate maximization of min_i lambda' Q_i lambda over the simplex (paper (14)).

Q_i = J_i J_i' is the Gram matrix of bundle point i (J_i: K x d Jacobian); internal values are squared gradient
norms, public GN values are their square roots.

  Envelope  K=2: exact lower envelope of the convex parabolas q_i(x) = lambda' Q_i lambda, lambda = (x, 1 - x)
            (paper Appendix A.4.1).  It is built by inserting one parabola at a time, which gives the same envelope
            as the divide and conquer of Algorithm 3; the selected weight is the piece end of largest value (the
            first one in case of ties).
  CCP       K>2: multi-start convex-concave procedure (paper Algorithm 2).

Every LP max_{lambda in simplex} min_i (M lambda)_i is solved by HiGHS (dual simplex, feasibility tolerances 1e-9),
warm-started from the basis of the previous LP of the same size; an LP with at least CG_MIN_ROWS rows is solved by
constraint generation, which returns an optimum of the full LP.
"""
import warnings
from itertools import combinations

import numpy as np
from scipy.optimize import linprog

CG_MIN_ROWS, CG_INIT, CG_ADD = 400, 40, 40
_HIGHS = {}  # HiGHS module, options and the basis of the previous LP
_CG = {}     # rows active at the previous constraint-generation optimum and its maximizer


def reset_lp_state():
    """Forget the warm-start basis and the constraint-generation rows (at the start of a run)."""
    _HIGHS.pop("basis", None); _HIGHS.pop("basis_shape", None); _CG.clear()


def roots(c):
    c = np.asarray(c, float); scale = np.max(np.abs(c))
    if scale == 0:
        return []
    a, b, d = c / scale
    if abs(a) < 2e-15:
        return [] if abs(b) < 2e-15 else [-d / b]
    disc = b * b - 4 * a * d
    if disc < -2e-14:
        return []
    disc = max(disc, 0.)
    q = -.5 * (b + np.copysign(np.sqrt(disc), b))
    if q == 0:
        return [-b / (2 * a)]
    return [q / a, d / q]


def snake_grid(K, r):
    """All lambda in the simplex with r * lambda integral, in snake (boustrophedon) order."""
    def comps(total, parts, forward=True):
        if parts == 1:
            return [[total]]
        out = []
        for a in range(total + 1):
            out.extend([a] + tail for tail in comps(total - a, parts - 1, forward=(a % 2 == 0)))
        return out if forward else out[::-1]
    return np.asarray(comps(int(r), int(K)), float) / r


class Envelope:
    """Incremental lower envelope of convex quadratics q_i(x), x = lambda_1 in [0, 1].

    The envelope is a list of pieces [PL_j, PR_j] with the index PI_j of the active quadratic and its values VA_j,
    VB_j at the two ends.  add() leaves a piece unchanged when the new quadratic stays above the active one on it by
    a safe margin, and splits the other pieces at the crossing points.
    """
    def __init__(self):
        self._cs = np.empty((64, 3)); self._n = 0
        self.PL = np.empty(0); self.PR = np.empty(0); self.PI = np.empty(0, dtype=np.int64)
        self.VA = np.empty(0); self.VB = np.empty(0)

    def _ends(self, L_, R_, I_):
        C = self._cs[I_]
        return (C[:, 0] * L_ + C[:, 1]) * L_ + C[:, 2], (C[:, 0] * R_ + C[:, 1]) * R_ + C[:, 2]

    def add(self, Q):
        q = np.asarray(Q)
        c = np.array([q[0, 0] - 2 * q[0, 1] + q[1, 1], 2 * (q[0, 1] - q[1, 1]), q[1, 1]])
        new = self._n
        if self._n == len(self._cs):
            self._cs = np.concatenate([self._cs, np.empty_like(self._cs)])
        self._cs[self._n] = c; self._n += 1
        if len(self.PI) == 0:
            self.PL = np.array([0.]); self.PR = np.array([1.]); self.PI = np.array([new], dtype=np.int64)
            self.VA, self.VB = self._ends(self.PL, self.PR, self.PI)
            return
        PL, PR, PI = self.PL, self.PR, self.PI
        D = c[None, :] - self._cs[PI]
        a, b, d = D[:, 0], D[:, 1], D[:, 2]
        mn = np.minimum((a * PL + b) * PL + d, (a * PR + b) * PR + d)
        with np.errstate(divide='ignore', invalid='ignore'):
            xv = np.where(a > 0, -b / (2 * a), np.nan)
        inside = (xv > PL) & (xv < PR)
        mn = np.where(inside, np.minimum(mn, (a * xv + b) * xv + d), mn)
        untouched = mn > 1e-9 * (np.abs(a) + np.abs(b) + np.abs(d))
        touched = np.nonzero(~untouched)[0]
        if len(touched) == 0:
            return
        chunks = []; seg = []; prev = 0

        def flush():
            if seg:
                L_, R_, I_ = (np.array([x[0] for x in seg]), np.array([x[1] for x in seg]),
                              np.array([x[2] for x in seg], dtype=np.int64))
                chunks.append((L_, R_, I_) + self._ends(L_, R_, I_))
                seg.clear()
        for t in touched:
            if t > prev:
                flush(); chunks.append((PL[prev:t], PR[prev:t], PI[prev:t], self.VA[prev:t], self.VB[prev:t]))
            left, right, old = float(PL[t]), float(PR[t]), int(PI[t])
            diff = c - self._cs[old]
            cuts = [left] + sorted(set(r for r in roots(diff) if left + 2e-15 < r < right - 2e-15)) + [right]
            for x0, x1 in zip(cuts[:-1], cuts[1:]):
                mid = (x0 + x1) / 2
                winner = new if (diff[0] * mid + diff[1]) * mid + diff[2] < 0 else old
                if seg and seg[-1][2] == winner:
                    seg[-1] = (seg[-1][0], x1, winner)
                else:
                    seg.append((x0, x1, winner))
            prev = t + 1
        flush()
        if prev < len(PI):
            chunks.append((PL[prev:], PR[prev:], PI[prev:], self.VA[prev:], self.VB[prev:]))
        self.PL = np.concatenate([ch[0] for ch in chunks]).astype(float)
        self.PR = np.concatenate([ch[1] for ch in chunks]).astype(float)
        self.PI = np.concatenate([ch[2] for ch in chunks]).astype(np.int64)
        self.VA = np.concatenate([ch[3] for ch in chunks]); self.VB = np.concatenate([ch[4] for ch in chunks])

    def select(self):
        """The selected weight: the piece end of largest value (a convex quadratic is largest at an end of its
        piece), the first one in the order left_0, right_0, left_1, ..."""
        V = np.empty(2 * len(self.VA)); V[0::2] = self.VA; V[1::2] = self.VB
        k = int(np.argmax(V)); x = float(self.PL[k // 2] if k % 2 == 0 else self.PR[k // 2])
        return np.array([x, 1 - x])

    def solve(self):
        """(max_lambda min_i lambda' Q_i lambda, maximizer): the value is re-evaluated on all quadratics."""
        lam = self.select(); x = float(lam[0])
        cs = self._cs[:self._n]
        return max(float(np.min((cs[:, 0] * x + cs[:, 1]) * x + cs[:, 2])), 0.), lam


def _highs_direct(M, warm=True):
    """The LP of lp() solved by HiGHS with the model and options scipy.optimize.linprog(method='highs') uses, the
    options validated once.  Returns (x, row duals), or None if HiGHS is not reachable this way or does not report
    an optimal model (lp() then calls linprog)."""
    if not _HIGHS:
        try:
            import scipy.optimize._highspy._core as _h
            from scipy.optimize._highspy._core import HighsDebugLevel, simplex_constants as s_c
        except ImportError:
            _HIGHS.update(h=None)
            return None
        o = _h.HighsOptions()
        o.presolve = "on"; o.highs_debug_level = HighsDebugLevel.kHighsDebugLevelNone
        o.dual_feasibility_tolerance = 1e-9; o.log_to_console = False; o.output_flag = False
        o.primal_feasibility_tolerance = 1e-9; o.simplex_strategy = s_c.SimplexStrategy.kSimplexStrategyDual
        o.threads = 1
        _HIGHS.update(h=_h, options=o, inf=_h.kHighsInf)
    if _HIGHS["h"] is None:
        return None
    _h, inf = _HIGHS["h"], _HIGHS["inf"]
    from scipy.sparse import csc_matrix
    m, K = M.shape
    A = csc_matrix(np.vstack((np.column_stack([np.ones(m), -M]), np.array([np.r_[0., np.ones(K)]]))))
    lp_ = _h.HighsLp()
    lp_.num_col_ = K + 1; lp_.num_row_ = m + 1
    lp_.a_matrix_.num_col_ = K + 1; lp_.a_matrix_.num_row_ = m + 1
    lp_.a_matrix_.format_ = _h.MatrixFormat.kColwise
    lp_.col_cost_ = np.r_[-1., np.zeros(K)]
    lp_.col_lower_ = np.r_[-inf, np.zeros(K)]; lp_.col_upper_ = np.r_[inf, np.ones(K)]
    lp_.row_lower_ = np.r_[np.full(m, -inf), 1.]; lp_.row_upper_ = np.r_[np.zeros(m), 1.]
    lp_.a_matrix_.start_ = A.indptr; lp_.a_matrix_.index_ = A.indices; lp_.a_matrix_.value_ = A.data
    highs = _h._Highs()
    if highs.passOptions(_HIGHS["options"]) == _h.HighsStatus.kError or highs.passModel(lp_) == _h.HighsStatus.kError:
        return None
    if warm and _HIGHS.get("basis_shape") == (m, K):
        highs.setBasis(_HIGHS["basis"])
    if highs.run() == _h.HighsStatus.kError or highs.getModelStatus() != _h.HighsModelStatus.kOptimal:
        return None
    if warm:
        _HIGHS["basis"], _HIGHS["basis_shape"] = highs.getBasis(), (m, K)
    sol = highs.getSolution()
    return np.array(sol.col_value), np.array(sol.row_dual)[:m]


def _lp_cg(M):
    """The LP of lp() by constraint generation.  HiGHS solves it on a working set of rows: the rows active at the
    previous optimum, the CG_INIT rows smallest at the previous maximizer (the center at first) and the smallest
    row of every column.  Every row outside the working set with (M lambda)_i < t - 1e-10 (stricter than the
    feasibility tolerance HiGHS applies inside) is violated; the CG_ADD most violated rows are added and the LP
    is solved again.  When no row is violated, lambda is optimal for the full LP and the duals of the working set
    (zero elsewhere) are optimal duals.  Returns (x, row duals) as _highs_direct, or None."""
    m, K = M.shape
    ref = _CG.get("lam")
    ref = ref if ref is not None and len(ref) == K else np.ones(K) / K
    W = np.zeros(m, bool); W[np.argsort(M @ ref, kind='stable')[:CG_INIT]] = True; W[M.argmin(axis=0)] = True
    hint = _CG.get("rows")
    if hint is not None:
        W[hint[hint < m]] = True
    while True:
        rows = np.flatnonzero(W)
        out = _highs_direct(M[rows], warm=False)
        if out is None:
            return None
        x, marg = out
        lam = np.maximum(x[1:], 0); lam /= lam.sum()
        gap = M @ lam - x[0]; gap[W] = 0.
        viol = np.flatnonzero(gap < -1e-10)
        if len(viol) == 0:
            full = np.zeros(m); full[rows] = marg
            _CG["rows"] = rows[marg < 0]; _CG["lam"] = lam
            return x, full
        W[viol[np.argsort(gap[viol], kind='stable')[:CG_ADD]]] = True


def lp(M, return_dual_bound=False):
    """max_{lambda in simplex} min_i (M lambda)_i: value, maximizer (and an upper bound from the duals)."""
    m, K = M.shape
    fast = _lp_cg(M) if m >= CG_MIN_ROWS else None
    if fast is None:
        fast = _highs_direct(M)
    if fast is not None:
        x, marg = fast
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = linprog(np.r_[-1., np.zeros(K)], A_ub=np.column_stack([np.ones(m), -M]), b_ub=np.zeros(m),
                             A_eq=np.array([np.r_[0., np.ones(K)]]), b_eq=[1.], bounds=[(None, None)] + [(0., 1.)] * K,
                             method='highs', options={'threads': 1, 'primal_feasibility_tolerance': 1e-9,
                                                      'dual_feasibility_tolerance': 1e-9})
        if not result.success:
            raise RuntimeError('lambda LP failed: ' + result.message)
        x, marg = result.x, result.ineqlin.marginals
    w = np.maximum(x[1:], 0); w /= w.sum()
    value = float(np.min(M @ w))
    if return_dual_bound:
        dual = np.maximum(-marg, 0.)
        if dual.sum() <= 0:
            raise RuntimeError('Missing feasible LP dual distribution')
        dual /= dual.sum()
        return value, w, float(np.max(dual @ M))
    return value, w


class CCP:
    """Multi-start CCP (paper Algorithm 2) with N = nseeds, r = nstarts and iteration cap maxiter.

    Each selection: (i) the LP for val(A) (paper Proposition 10) gives its maximizer lambda_A and, from its dual,
    an upper bound; it is solved again only when a row added since the last solve cuts lambda_A, otherwise lambda_A,
    val(A) and the bound are unchanged.  (ii) Seeds: the K vertices, lambda_A, N points drawn uniformly from the
    simplex anew at every selection, the polished points of the previous selection and a pool of at most keep_pool
    earlier local maxima (newest first, pairwise distance > 0.08); in addition the center and, for K > 3, the
    boundary_resolution - 1 interior grid points of every edge and the centers of the 3-faces.  phi is evaluated on
    all seeds in one batched contraction; values are divided by the largest phi at the vertices, the center and the
    edge and face points.  If the best seed is within a relative 1e-8 of the upper bound, it is returned.
    (iii) Otherwise the r best seeds more than 0.08 apart are polished by CCP steps (the LP for val(M^c), then a
    move to its maximizer); a start stops when the predicted improvement delta_c is at most 1e-8 max{1, phi}
    (Appendix A) or after maxiter LPs.  (iv) The selected weight is the point of largest phi found; the polished
    points update the pool.
    """
    def __init__(self, K, nseeds, nstarts, maxiter, seed=42, boundary_resolution=None, keep_pool=0):
        self.K = K; self.nseeds = nseeds; self.nstarts = nstarts; self.maxiter = maxiter
        self.rng = np.random.default_rng(seed)
        boundary = []
        if boundary_resolution is not None and K > 3:
            res = max(int(boundary_resolution), 2)
            for i, j in combinations(range(K), 2):
                for numerator in range(1, res):
                    w = np.zeros(K); w[i] = numerator / res; w[j] = 1. - numerator / res
                    boundary.append(w)
            for ijk in combinations(range(K), 3):
                w = np.zeros(K); w[list(ijk)] = 1. / 3.
                boundary.append(w)
        self.seeds = np.vstack([np.eye(K), np.ones((1, K)) / K, np.asarray(boundary).reshape((-1, K))])
        self.scores = np.full(len(self.seeds), np.inf)  # phi of the fixed seeds on the bundle
        self.Q = np.empty((64, K, K)); self.n = 0; self.has_zero = False  # Gram matrices of the bundle
        self.sandwich = None  # (rows at the last val(A) LP, lambda_A, min_i (A lambda_A)_i, dual bound)
        self.previous = []; self.pool = []; self.keep_pool = int(keep_pool); self.lp_count = 0

    def add(self, Q):
        if self.n == len(self.Q):
            self.Q = np.concatenate([self.Q, np.empty_like(self.Q)])
        self.Q[self.n] = Q; self.n += 1
        self.has_zero = self.has_zero or not np.any(Q)
        self.scores = np.minimum(self.scores, np.einsum('ni,ij,nj->n', self.seeds, Q, self.seeds))

    def select(self):
        return self.solve()[1]

    def solve(self):
        """(phi value, selected weight, upper bound), unnormalized."""
        Q = self.Q[:self.n]
        scale = max(float(np.max(self.scores)), 1e-300)
        if self.has_zero:
            return 0., np.eye(self.K)[0], 0.
        qs = Q / scale
        sw = self.sandwich
        if sw is not None and float(np.min(np.diagonal(Q[sw[0]:], axis1=1, axis2=2) @ sw[1])) >= sw[2]:
            wA, ub = sw[1], sw[3] / scale
            self.sandwich = (self.n,) + sw[1:]
        else:
            _, wA, ub = lp(np.diagonal(qs, axis1=1, axis2=2), return_dual_bound=True); self.lp_count += 1
            self.sandwich = (self.n, wA, float(np.min(np.diagonal(Q, axis1=1, axis2=2) @ wA)), ub * scale)
        scores = self.scores / scale
        fresh = self.rng.dirichlet(np.ones(self.K), size=self.nseeds)
        extra = np.vstack([wA] + self.previous + list(self.pool) + [fresh])
        # phi of the other seeds as one matrix product (w w')_flat . (Q_i)_flat
        es = ((extra[:, :, None] * extra[:, None, :]).reshape(len(extra), -1) @ qs.reshape(len(qs), -1).T).min(axis=1)
        seeds = np.vstack([self.seeds, extra]); scores = np.r_[scores, es]
        order = np.argsort(-scores, kind='stable')
        best = float(scores[order[0]]); wb = seeds[order[0]].copy()
        if ub - best <= 1e-8 * max(1., abs(ub)):  # the best seed closes the sandwich
            self.previous = [wb.copy()]
            if self.keep_pool > 0 and (not self.pool or
                                       np.sqrt(((np.asarray(self.pool) - wb) ** 2).sum(axis=1)).min() > .08):
                self.pool = ([wb.copy()] + list(self.pool))[:self.keep_pool]
            return max(0., best * scale), wb, max(best, ub) * scale
        chosen = []; maxima = []
        for idx in order:
            if all(np.linalg.norm(seeds[idx] - p) > .08 for p in chosen):
                chosen.append(seeds[idx])
            if len(chosen) >= self.nstarts:
                break
        for w in chosen:
            w = w.copy()
            for _ in range(self.maxiter):
                qw = qs @ w; vals = qw @ w; old = float(vals.min())
                lower, wn = lp(2 * qw - vals[:, None]); self.lp_count += 1
                new = float(np.min((qs @ wn) @ wn))
                if new > best:
                    best, wb = new, wn.copy()
                if new < old - 2e-8:
                    raise RuntimeError('CCP descent exceeds numerical tolerance')
                w = wn
                if lower - old <= 1e-8 * max(abs(old), 1.):
                    break
            maxima.append(w)
        self.previous = maxima
        if self.keep_pool > 0:  # greedy, newest first: keep a point if it is > 0.08 from every point kept so far
            cand = np.asarray([np.asarray(w, float) for w in maxima + list(self.pool)]).reshape(-1, self.K)
            kept = np.empty_like(cand); n = 0
            for w in cand:
                if n == 0 or np.sqrt(((kept[:n] - w) ** 2).sum(axis=1)).min() > .08:
                    kept[n] = w; n += 1
            self.pool = [kept[i].copy() for i in range(min(n, self.keep_pool))]
        if best > ub + 1e-7:
            raise RuntimeError('CCP bound violation')
        return max(0., best * scale), wb, max(best, ub) * scale
