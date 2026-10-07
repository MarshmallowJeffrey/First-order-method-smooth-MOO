"""Preference-vector selection: approximate maximization of min_i lambda' Q_i lambda over the simplex, problem (14).

Q_i = J_i J_i' is the Gram matrix of bundle point i (J_i: K x d Jacobian); internal values are squared gradient
norms, public GN values are their square roots.

  Envelope  K=2: exact lower envelope of the convex parabolas q_i(x) = lambda' Q_i lambda, lambda = (x, 1 - x)
            (Appendix A.2).  It is built by inserting one parabola at a time, which gives the same envelope as
            the divide and conquer of Algorithms 4 and 5; the selected weight is the piece end of largest value (the
            first one in case of ties), as in Algorithm 3.
  CCP       K>2: multi-start convex-concave procedure (the CCP component of Algorithm 2).

Every LP max_{lambda in simplex} min_i (M lambda)_i is solved by HiGHS (dual simplex, feasibility tolerances 1e-9),
warm-started from the basis of the previous LP of the same size; an LP with at least CG_MIN_ROWS rows is solved by
constraint generation, which returns an optimum of the full LP.  LP options (set by CCP for its own LPs only, see
lp_options; the defaults are used everywhere else, e.g. by the reported metric):
  presolve  HiGHS presolve on (default) or off (a second solver instance; off gives the same solutions here);
  warm_cg   constraint generation starts from the rows of the previous LP whose slacks were nonbasic, with their
            basis statuses (the other rows basic), instead of a cold start; the result is again an optimum;
  relaxed   constraint generation solves the LP on its initial working set only, without checking or adding the
            other rows.  This is a relaxation: its maximizer need not be optimal for the full LP and its value is at
            least the full optimum, so a predicted improvement computed from it is at least the exact one; its dual,
            extended by zeros, still gives a valid upper bound (Proposition 9).  Values are always evaluated
            exactly on every row.
"""
import warnings
from contextlib import contextmanager
from itertools import combinations

import numpy as np
from scipy.optimize import linprog

CG_MIN_ROWS, CG_INIT, CG_ADD = 400, 40, 40
_HIGHS = {}     # HiGHS module, the solver instance (presolve on) and the basis of its previous LP
_HIGHS_NP = {}  # the same with presolve off
_CG = {}        # rows active at the previous constraint-generation optimum and its maximizer
_CGW = {}       # warm_cg: the final working set and basis of the previous constraint-generation LP
_OPT = dict(presolve=True, warm_cg=False, relaxed=False)


@contextmanager
def lp_options(**options):
    """LP options for the LPs solved inside the block (see the module docstring)."""
    saved = dict(_OPT); _OPT.update(options)
    try:
        yield
    finally:
        _OPT.clear(); _OPT.update(saved)


def reset_lp_state():
    """Forget the warm-start bases and the constraint-generation rows (at the start of a run)."""
    for H in (_HIGHS, _HIGHS_NP):
        H.pop("basis", None); H.pop("basis_shape", None)
    _CG.clear(); _CGW.clear()


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


def _highs_presolve_on():
    """HiGHS through SciPy's internal interface, with the options scipy.optimize.linprog(method='highs') uses, and one
    solver instance; None if HiGHS is not reachable this way (lp() then calls linprog)."""
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
        highs = _h._Highs()
        _HIGHS.update(h=None if highs.passOptions(o) == _h.HighsStatus.kError else _h, highs=highs, inf=_h.kHighsInf,
                      options=o)
    return _HIGHS if _HIGHS["h"] is not None else None


def _highs():
    """The solver instance for the current options (presolve off: a second instance with otherwise equal options)."""
    H = _highs_presolve_on()
    if H is None or _OPT["presolve"]:
        return H
    if not _HIGHS_NP:
        _h = H["h"]; o = _h.HighsOptions()
        for k in ("dual_feasibility_tolerance", "primal_feasibility_tolerance", "simplex_strategy", "highs_debug_level"):
            setattr(o, k, getattr(H["options"], k))
        o.presolve = "off"; o.log_to_console = False; o.output_flag = False; o.threads = 1
        highs = _h._Highs(); highs.passOptions(o)
        _HIGHS_NP.update(h=_h, highs=highs, inf=_h.kHighsInf)
    return _HIGHS_NP


def _optimal(H):
    _h, highs = H["h"], H["highs"]
    status = highs.run()
    if status == _h.HighsStatus.kError and _reset_scheduler(_h):
        status = highs.run()
    return status != _h.HighsStatus.kError and highs.getModelStatus() == _h.HighsModelStatus.kOptimal


def _reset_scheduler(_h):
    """HiGHS runs on a global task scheduler fixed by the first solve in the process; if another HiGHS user (e.g.
    scipy.optimize.linprog without a `threads` option) started it with a different number of threads, run() refuses
    our one-thread instance.  Destroys that scheduler so that the next run() starts one with our setting; False if
    this HiGHS build does not offer it."""
    reset = getattr(_h._Highs, "resetGlobalScheduler", None)
    if reset is None:
        return False
    reset(True)
    return True


def _highs_direct(M, H, warm=True, basis=None):
    """The LP of lp() passed to the solver instance H and solved, warm-started from the basis of the previous LP of
    the same size (warm) or from `basis`.  Returns (x, row duals), or None if HiGHS does not report an optimal model."""
    _h, highs, inf = H["h"], H["highs"], H["inf"]
    m, K = M.shape
    # [1, -M; 0, 1'] in column-wise (CSC) form, as scipy.sparse.csc_matrix gives it: column by column, rows in
    # ascending order, exact zeros left out
    AT = np.empty((K + 1, m + 1)); AT[0, :m] = 1.; AT[0, m] = 0.; AT[1:, :m] = -M.T; AT[1:, m] = 1.
    nz = AT != 0
    start = np.zeros(K + 2, dtype=np.int32); np.cumsum(nz.sum(axis=1), out=start[1:])
    lp_ = _h.HighsLp()
    lp_.num_col_ = K + 1; lp_.num_row_ = m + 1
    lp_.a_matrix_.num_col_ = K + 1; lp_.a_matrix_.num_row_ = m + 1
    lp_.a_matrix_.format_ = _h.MatrixFormat.kColwise
    lp_.col_cost_ = np.r_[-1., np.zeros(K)]
    lp_.col_lower_ = np.r_[-inf, np.zeros(K)]; lp_.col_upper_ = np.r_[inf, np.ones(K)]
    lp_.row_lower_ = np.r_[np.full(m, -inf), 1.]; lp_.row_upper_ = np.r_[np.zeros(m), 1.]
    lp_.a_matrix_.start_ = start; lp_.a_matrix_.index_ = np.nonzero(nz)[1].astype(np.int32); lp_.a_matrix_.value_ = AT[nz]
    if highs.passModel(lp_) == _h.HighsStatus.kError:
        return None
    if warm and H.get("basis_shape") == (m, K):
        highs.setBasis(H["basis"])
    if basis is not None:
        highs.setBasis(basis)
    if not _optimal(H):
        return None
    if warm:
        H["basis"], H["basis_shape"] = highs.getBasis(), (m, K)
    sol = highs.getSolution()
    return np.array(sol.col_value), np.array(sol.row_dual)[:m]


def _lp_cg(M):
    """The LP of lp() by constraint generation.  HiGHS solves it on a working set of rows: the rows active at the
    previous optimum, the CG_INIT rows smallest at the previous maximizer (the center at first) and the smallest
    row of every column (warm_cg: also the rows of the previous LP whose slacks were nonbasic, started from their
    basis statuses).  Every row outside the working set with (M lambda)_i < t - 1e-10 (stricter than the feasibility
    tolerance HiGHS applies inside) is violated; the CG_ADD most violated rows are added to the model and the LP is
    solved again from the current basis.  When no row is violated, lambda is optimal for the full LP and the duals
    of the working set (zero elsewhere) are optimal duals.  relaxed: no row is checked or added.  Returns (x, row
    duals) as _highs_direct, or None."""
    H = _highs()
    if H is None:
        return None
    _h, highs = H["h"], H["highs"]
    m, K = M.shape
    ref = _CG.get("lam")
    ref = ref if ref is not None and len(ref) == K else np.ones(K) / K
    W = np.zeros(m, bool); W[np.argpartition(M @ ref, CG_INIT)[:CG_INIT]] = True; W[M.argmin(axis=0)] = True
    hint = _CG.get("rows")
    if hint is not None:
        W[hint[hint < m]] = True
    rows = list(np.flatnonzero(W)); n0 = len(rows)  # model rows: rows[:n0], the simplex row, then rows[n0:]
    basis = None
    if _OPT["warm_cg"] and _CGW.get("K") == K and all(r < m for r in _CGW["nonbasic"]):
        nb = _CGW["nonbasic"]
        W[list(nb)] = True; rows = list(np.flatnonzero(W)); n0 = len(rows)
        basis = _h.HighsBasis(); basis.col_status = list(_CGW["col_status"])
        basis.row_status = [nb.get(r, _h.HighsBasisStatus.kBasic) for r in rows] + [_CGW["simplex_status"]]
        basis.valid = True
    out = _highs_direct(M[rows], H, warm=False, basis=basis)
    if out is None:
        return None
    x, duals = out
    while True:
        lam = np.maximum(x[1:], 0); lam /= lam.sum()
        gap = M @ lam - x[0]; gap[W] = 0.
        viol = np.empty(0, dtype=int) if _OPT["relaxed"] else np.flatnonzero(gap < -1e-10)
        if len(viol) == 0:
            idx = np.asarray(rows); full = np.zeros(m); full[idx] = duals
            _CG["rows"] = idx[duals < 0]; _CG["lam"] = lam
            if _OPT["warm_cg"]:
                fb = highs.getBasis(); rs = list(fb.row_status); order = list(idx[:n0]) + [None] + list(idx[n0:])
                _CGW.update(K=K, col_status=list(fb.col_status), simplex_status=rs[n0],
                            nonbasic={r: st for r, st in zip(order, rs) if r is not None and st != _h.HighsBasisStatus.kBasic})
            return x, full
        new = viol[np.argsort(gap[viol], kind='stable')[:CG_ADD]]; k = len(new)
        W[new] = True; rows.extend(new)
        highs.addRows(k, np.full(k, -H["inf"]), np.zeros(k), k * (K + 1), np.arange(0, k * (K + 1), K + 1, dtype=np.int32),
                      np.tile(np.arange(K + 1, dtype=np.int32), k), np.column_stack([np.ones(k), -M[new]]).ravel())
        if not _optimal(H):
            return None
        sol = highs.getSolution(); x = np.array(sol.col_value); d = np.array(sol.row_dual)
        duals = np.r_[d[:n0], d[n0 + 1:]]


def lp(M, return_dual_bound=False):
    """max_{lambda in simplex} min_i (M lambda)_i: (value, maximizer[, dual bound]).  The value is min_i (M lambda)_i
    at the returned maximizer, evaluated on all rows; the maximizer is optimal for the full LP unless the relaxed
    option is on (then for the working-set LP).  The dual bound is max_k (w' M)_k for the dual distribution w over
    the rows (projected onto the simplex), valid for any such w (Proposition 9, (22)), relaxed or not."""
    m, K = M.shape
    fast = _lp_cg(M) if m >= CG_MIN_ROWS else None
    if fast is None:
        H = _highs()
        fast = None if H is None else _highs_direct(M, H)
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


def ccp_ascent(qs, w, maxiter, tau=1e-8):
    """CCP steps from w (Algorithm 2, inner loop): the LP for val(M^c) of the linearizations at the current
    point, then a move to its maximizer, until the predicted improvement delta_c <= tau max{1, phi} or after maxiter
    LPs.  With exact LPs phi does not decrease; with relaxed LPs (lp_options) a step that lowers phi ends the start.
    Returns (last point, largest phi seen after a step, the point where it was seen, number of LPs)."""
    best, wb = -np.inf, None
    w = np.array(w, float)
    for count in range(1, maxiter + 1):
        qw = qs @ w; vals = qw @ w; old = float(vals.min())
        lower, wn = lp(2 * qw - vals[:, None])
        new = float(np.min((qs @ wn) @ wn))
        if new > best:
            best, wb = new, wn.copy()
        if new < old - 2e-8:
            if not _OPT["relaxed"]:
                raise RuntimeError('CCP descent exceeds numerical tolerance')
            break
        w = wn
        if lower - old <= tau * max(abs(old), 1.):
            break
    return w, best, wb, count


class CCP:
    """Multi-start CCP (the CCP component of Algorithm 2) with N = nseeds, r = nstarts, c_max = maxiter and
    the local stopping tolerance tau.

    A full selection: (i) the LP for val(A) (Proposition 9) gives its maximizer lambda_A (with relaxed LPs: of
    the working-set LP) and, from its dual, an upper bound; it is solved again only when a row added since the last solve cuts lambda_A, otherwise lambda_A,
    val(A) and the bound are unchanged.  (ii) Seeds: the K vertices, lambda_A, N points drawn uniformly from the
    simplex anew at every selection, the polished points of the previous selection and a pool of at most keep_pool
    earlier local maxima (newest first, pairwise distance > 0.08); in addition the center and, for K > 3, the
    boundary_resolution - 1 interior grid points of every edge and the centers of the 3-faces.  phi is evaluated on
    all seeds; values are divided by the largest phi at the vertices, the center and the edge and face points.  If
    the best seed is within a relative 1e-8 of the upper bound, it is returned.  (iii) Otherwise the r best seeds
    more than 0.08 apart are polished by CCP steps (the LP for val(M^c), then a move to its maximizer); a start stops
    when the predicted improvement delta_c is at most tau max{1, phi} (phi normalized as in (ii)) or after maxiter
    LPs.  (iv) The selected weight is the point of largest phi found; the polished points update the pool.

    Options (the defaults give the procedure above):
      lazy, lazy_rho  a full selection at every lazy-th call; at the other calls the selected weight is the best of
                      the seeds of (ii) (the last lambda_A, the previous selection, the pool, N fresh draws), without
                      LPs, unless its phi is below lazy_rho times that of the last full selection (then a full one);
      screen          exact screening of the fresh seeds (r = 1 only): phi of a fresh seed is accumulated over blocks of
                      bundle points, newest first, and the seed is dropped as soon as it cannot exceed the best value
                      of the seeds before it (it cannot be selected); phi of the retained seeds of (ii) is kept from
                      the previous call and updated with the new bundle points only;
      presolve, warm_cg, relaxed   LP options of the LPs of this selection (module docstring).
    """
    def __init__(self, K, nseeds, nstarts, maxiter, seed=42, boundary_resolution=None, keep_pool=0, tau=1e-8,
                 lazy=1, lazy_rho=.5, screen=False, presolve=True, warm_cg=False, relaxed=False):
        self.K = K; self.nseeds = nseeds; self.nstarts = nstarts; self.maxiter = maxiter; self.tau = tau
        self.lazy, self.lazy_rho, self.screen = int(lazy), float(lazy_rho), bool(screen)
        self.lp_opts = dict(presolve=bool(presolve), warm_cg=bool(warm_cg), relaxed=bool(relaxed))
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
        self.calls = 0; self.full_calls = 0; self.last_phi = None; self.seed_cache = {}

    def add(self, Q):
        if self.n == len(self.Q):
            self.Q = np.concatenate([self.Q, np.empty_like(self.Q)])
        self.Q[self.n] = Q; self.n += 1
        self.has_zero = self.has_zero or not np.any(Q)
        self.scores = np.minimum(self.scores, np.einsum('ni,ij,nj->n', self.seeds, Q, self.seeds))

    def select(self):
        return self.solve()[1]

    def solve(self):
        """(phi value, selected weight, upper bound or inf after a selection without LPs), unnormalized."""
        self.calls += 1
        with lp_options(**self.lp_opts):
            if self.lazy > 1 and (self.calls - 1) % self.lazy != 0 and self.last_phi is not None:
                out = self._without_lp()
                if out is not None:
                    return out
            self.full_calls += 1
            out = self._full()
            self.last_phi = out[0]
            return out

    def _phi_seeds(self, extra, qs, bar, nfresh):
        """phi (normalized) of the extra seeds, the last nfresh of them fresh draws; with screening a fresh seed that
        cannot exceed bar or the seeds before it gets -inf (see the class docstring)."""
        E = (extra[:, :, None] * extra[:, None, :]).reshape(len(extra), -1); Qf = qs.reshape(len(qs), -1)
        if not self.screen or self.nstarts != 1 or len(qs) <= 320 or nfresh == 0:
            return (E @ Qf.T).min(axis=1)
        h = len(extra) - nfresh; out = np.full(len(extra), -np.inf); n = len(Qf)
        if h:  # retained seeds: their minimum over the points seen before, updated with the new points only
            new_cache = {}
            for i in range(h):
                key = extra[i].tobytes(); val, seen = self.seed_cache.get(key, (np.inf, 0))
                if seen < n:
                    val = min(val, float((E[i] @ self.Q[seen:n].reshape(n - seen, -1).T).min()))
                new_cache[key] = (val, n); out[i] = val / self._scale
            self.seed_cache = new_cache
            bar = max(bar, float(out[:h].max()))
        alive = np.arange(h, len(extra)); run = np.full(len(alive), np.inf); st = n - 1; block = 256
        while st >= 0:  # newest points first, a first block of 256 points, then 1,024
            lo = max(0, st - block + 1); block = 1024
            run = np.minimum(run, (E[alive] @ Qf[lo:st + 1].T).min(axis=1))
            k = run > bar; alive, run = alive[k], run[k]
            st = lo - 1
            if len(alive) == 0:
                break
        out[alive] = run
        return out

    def _without_lp(self):
        """The best of the fixed seeds, the last lambda_A, the previous selection, the pool and N fresh draws; None
        (a full selection follows) if its phi is below lazy_rho times that of the last full selection."""
        Q = self.Q[:self.n]; scale = max(float(np.max(self.scores)), 1e-300); qs = Q / scale; self._scale = scale
        fresh = self.rng.dirichlet(np.ones(self.K), size=self.nseeds)
        wA = [self.sandwich[1]] if self.sandwich is not None else []
        extra = np.vstack(wA + self.previous + list(self.pool) + [fresh])
        es = self._phi_seeds(extra, qs, float(np.max(self.scores / scale)), len(fresh))
        seeds = np.vstack([self.seeds, extra]); scores = np.r_[self.scores / scale, es]
        k = int(np.argmax(scores)); best = float(scores[k])
        if best * scale < self.lazy_rho * self.last_phi:
            return None
        self.previous = [seeds[k].copy()]
        return max(0., best * scale), seeds[k].copy(), np.inf

    def _full(self):
        Q = self.Q[:self.n]
        scale = max(float(np.max(self.scores)), 1e-300); self._scale = scale
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
        es = self._phi_seeds(extra, qs, float(np.max(scores)), len(fresh))
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
            w, value, point, count = ccp_ascent(qs, w, self.maxiter, self.tau); self.lp_count += count
            if value > best:
                best, wb = value, point
            maxima.append(w)
        self.previous = maxima
        if self.keep_pool > 0:  # greedy, newest first: keep a point if it is > 0.08 from every point kept so far
            cand = np.asarray([np.asarray(w, float) for w in maxima + list(self.pool)]).reshape(-1, self.K)
            far = np.sqrt(((cand[:, None, :] - cand[None, :, :]) ** 2).sum(axis=2)) > .08
            keep = []
            for i in range(len(cand)):
                if not keep or far[i, keep].all():
                    keep.append(i)
            self.pool = [cand[i].copy() for i in keep[:self.keep_pool]]
        if best > ub + 1e-7:
            raise RuntimeError('CCP bound violation')
        return max(0., best * scale), wb, max(best, ub) * scale
