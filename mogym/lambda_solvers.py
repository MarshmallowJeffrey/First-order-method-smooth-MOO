"""Preference-weight solvers for max_{lambda in simplex} min_i lambda' Q_i lambda.

Q_i = J_i J_i' is the Gram matrix of bundle point i (J_i: K x d Jacobian); all internal values are
squared gradient norms, public GN values are their square roots.

  Envelope             K=2: exact lower envelope of parabolas on lambda_1 in [0, 1].
  K3BivariateEnvelope  K=3: simplicial branch-and-bound on the 2-simplex; returns a certified
                       [lower, upper] interval (cell upper bound min_i max_vertex q_i, valid by convexity).
  CCP                  multistart convex-concave procedure (paper appendix, Algorithm "Multi-start CCP").
  PeriodicStrongCCP    K>3: CCP with small settings, and a stronger CCP every `period` selections.
"""
import heapq
import warnings
from itertools import combinations

import numpy as np
from scipy.optimize import linprog

# LP warm start (Algorithm "Multi-start CCP"): when WARM_LP is set, every LP starts from the basis of the
# previous LP of the same size.  The state is module-wide, as in the runs reported in the paper, so the
# checkpoint evaluation (evaluate_gram) and the training solver share it.
WARM_LP = False
_HIGHS = {}


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


def simplex_grid(K, r):
    """All lambda in the simplex with r * lambda integral, in lexicographic order."""
    def rec(k, n):
        if k == 1:
            return [[n]]
        return [[i] + tail for i in range(n + 1) for tail in rec(k - 1, n - i)]
    return np.asarray(rec(K, r), float) / r


def snake_grid(K, r):
    """simplex_grid(K, r) in snake (boustrophedon) order: consecutive weights are 2/r apart in l1."""
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

    The envelope is a list of pieces [PL_j, PR_j] with the index PI_j of the active quadratic.  add()
    leaves a piece unchanged when the new quadratic stays above the active one on it by a safe margin
    and splits the other pieces at the crossing points.
    """
    def __init__(self):
        self.coeff = []
        self._cs = np.empty((64, 3)); self._n = 0
        self.PL = np.empty(0); self.PR = np.empty(0); self.PI = np.empty(0, dtype=np.int64)

    def add(self, Q):
        q = np.asarray(Q)
        c = np.array([q[0, 0] - 2 * q[0, 1] + q[1, 1], 2 * (q[0, 1] - q[1, 1]), q[1, 1]])
        new = len(self.coeff); self.coeff.append(c)
        if self._n == len(self._cs):
            self._cs = np.concatenate([self._cs, np.empty_like(self._cs)])
        self._cs[self._n] = c; self._n += 1
        if len(self.PI) == 0:
            self.PL = np.array([0.]); self.PR = np.array([1.]); self.PI = np.array([new], dtype=np.int64)
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
                chunks.append((np.array([x[0] for x in seg]), np.array([x[1] for x in seg]),
                               np.array([x[2] for x in seg], dtype=np.int64)))
                seg.clear()
        for t in touched:
            if t > prev:
                flush(); chunks.append((PL[prev:t], PR[prev:t], PI[prev:t]))
            left, right, old = float(PL[t]), float(PR[t]), int(PI[t])
            diff = c - self.coeff[old]
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
            chunks.append((PL[prev:], PR[prev:], PI[prev:]))
        self.PL = np.concatenate([ch[0] for ch in chunks]).astype(float)
        self.PR = np.concatenate([ch[1] for ch in chunks]).astype(float)
        self.PI = np.concatenate([ch[2] for ch in chunks]).astype(np.int64)

    def solve(self):
        # a convex quadratic attains its maximum on a piece at an end point; first maximum in the order
        # (left_0, right_0, left_1, right_1, ...)
        C = self._cs[self.PI]
        va = (C[:, 0] * self.PL + C[:, 1]) * self.PL + C[:, 2]
        vb = (C[:, 0] * self.PR + C[:, 1]) * self.PR + C[:, 2]
        V = np.empty(2 * len(va)); V[0::2] = va; V[1::2] = vb
        X = np.empty(2 * len(va)); X[0::2] = self.PL; X[1::2] = self.PR
        k = int(np.argmax(V)); xbest = float(X[k])
        lam = np.array([xbest, 1 - xbest])
        cs = self._cs[:self._n]  # re-evaluate all constraints at the returned weight
        best = float(np.min((cs[:, 0] * xbest + cs[:, 1]) * xbest + cs[:, 2]))
        return max(best, 0.), lam


def _highs_direct(M):
    """The LP of lp() solved by HiGHS with the model and options scipy.optimize.linprog(method='highs')
    uses, the options validated once.  Returns (x, row duals), or None if HiGHS is not reachable this way
    or does not report an optimal model (lp() then calls linprog)."""
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
    if WARM_LP and _HIGHS.get("basis_shape") == (m, K):
        highs.setBasis(_HIGHS["basis"])
    if highs.run() == _h.HighsStatus.kError or highs.getModelStatus() != _h.HighsModelStatus.kOptimal:
        return None
    if WARM_LP:
        _HIGHS["basis"], _HIGHS["basis_shape"] = highs.getBasis(), (m, K)
    sol = highs.getSolution()
    return np.array(sol.col_value), np.array(sol.row_dual)[:m]


def lp(M, return_dual_bound=False):
    """max_{lambda in simplex} min_i (M lambda)_i: value, maximizer (and an upper bound from the duals)."""
    m, K = M.shape
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
    """Multistart CCP (paper appendix).  Seeds: vertices, center, lambda_A of the sandwich LP, the maxima
    of the previous solve, uniform random draws (a fixed batch, or redrawn at every solve with
    fresh_seeds=True), and for K>3 optionally points on the edges and the centers of the 3-faces."""
    def __init__(self, K, nseeds=256, nstarts=4, seed=42, maxiter=100, boundary_seeds=False,
                 boundary_resolution=4, fresh_seeds=False):
        self.K = K; self.nstarts = nstarts; self.maxiter = maxiter
        self.fresh_seeds = bool(fresh_seeds); self.nseeds = nseeds
        self.rng = np.random.default_rng(seed)
        random = self.rng.dirichlet(np.ones(K), size=0 if fresh_seeds else nseeds)
        grid = simplex_grid(K, 20) if K == 3 else np.empty((0, K))
        boundary = []
        if boundary_seeds and K > 3:
            boundary_resolution = max(int(boundary_resolution), 2)
            for i, j in combinations(range(K), 2):
                for numerator in range(1, boundary_resolution):
                    fraction = numerator / boundary_resolution
                    w = np.zeros(K); w[i] = fraction; w[j] = 1. - fraction
                    boundary.append(w)
            for ijk in combinations(range(K), 3):
                w = np.zeros(K); w[list(ijk)] = 1. / 3.
                boundary.append(w)
        extra = np.asarray(boundary).reshape((-1, K))
        self.seeds = np.vstack([np.eye(K), np.ones((1, K)) / K, grid, random, extra])
        self.scores = np.full(len(self.seeds), np.inf)
        self.Q = []; self.previous = []; self.lp_count = 0

    def add(self, Q):
        self.Q.append(np.array(Q))
        self.scores = np.minimum(self.scores, np.einsum('ni,ij,nj->n', self.seeds, Q, self.seeds))

    def solve(self):
        Q = np.asarray(self.Q)
        # normalize near a feasible max-min value so that the LP tolerances stay relative
        scale = max(float(np.max(self.scores)), 1e-300)
        if np.any(np.max(np.abs(Q), axis=(1, 2)) == 0):
            return 0., np.eye(self.K)[0], 0.
        qs = Q / scale
        _, wA, ub = lp(np.diagonal(qs, axis1=1, axis2=2), return_dual_bound=True); self.lp_count += 1
        scores = self.scores / scale
        fresh = (self.rng.dirichlet(np.ones(self.K), size=self.nseeds) if self.fresh_seeds
                 else np.empty((0, self.K)))
        extra = np.vstack([wA] + self.previous + [fresh])
        es = np.einsum('ni,mij,nj->nm', extra, qs, extra).min(axis=1)
        seeds = np.vstack([self.seeds, extra]); scores = np.r_[scores, es]
        order = np.argsort(-scores, kind='stable')
        best = float(scores[order[0]]); wb = seeds[order[0]].copy()
        if ub - best <= 1e-8 * max(1., abs(ub)):  # sandwich closed
            self.previous = [wb.copy()]
            return max(0., best * scale), wb, max(best, ub) * scale
        chosen = []; maxima = []
        for idx in order:
            if all(np.linalg.norm(seeds[idx] - p) > .08 for p in chosen):
                chosen.append(seeds[idx])
            if len(chosen) >= self.nstarts:
                break
        for w in chosen:
            w = w.copy()
            for it in range(self.maxiter):
                qw = qs @ w; vals = qw @ w; old = float(vals.min())
                lower, wn = lp(2 * qw - vals[:, None]); self.lp_count += 1
                new = float(np.min((qs @ wn) @ wn))
                if new > best:
                    best, wb = new, wn.copy()
                if new < old - 2e-8:
                    raise RuntimeError('CCP descent exceeds numerical tolerance')
                w = wn
                if lower - old <= 1e-8 * max(abs(old), 1e-6):
                    break
            maxima.append(w)
        self.previous = maxima
        if best > ub + 1e-7:
            raise RuntimeError('CCP bound violation')
        return max(0., best * scale), wb, max(best, ub) * scale


class K3BivariateEnvelope:
    """K=3: branch-and-bound over triangles of the 2-simplex until the upper bound is within rtol of the
    best feasible value or max_nodes splits were made.  solve() returns (lower, argmax, upper), squared."""
    def __init__(self, rtol=.005, max_nodes=5000, grid_resolution=12):
        self.K = 3; self.rtol = float(rtol); self.max_nodes = int(max_nodes)
        self.grid = simplex_grid(3, int(grid_resolution)); self.Q = []
        self.previous = []; self.solve_count = 0; self.total_splits = 0
        self.last_splits = 0; self.last_relative_gap = np.inf; self._cached = None

    def add(self, Q):
        self.Q.append(np.asarray(Q, float)); self._cached = None

    def solve(self):
        if self._cached is not None:
            val, w, upper = self._cached
            return val, w.copy(), upper
        Q = np.asarray(self.Q, float)
        scale = max(float(np.max(np.abs(Q))), 1e-300); qs = Q / scale
        seeds = np.vstack([np.eye(3), np.ones((1, 3)) / 3, self.grid] + self.previous)
        seed_values = np.einsum('nk,mkl,nl->nm', seeds, qs, seeds, optimize=True).min(axis=1)
        ibest = int(np.argmax(seed_values)); best = max(float(seed_values[ibest]), 0.)
        bestw = seeds[ibest].copy(); padding = 1e-13
        heap = []; counter = 0; splits = 0

        def add_cell(vertices):
            nonlocal counter, best, bestw
            values = np.einsum('lnm,nl->nm', np.einsum('mln->lnm', np.tensordot(qs, vertices, axes=((1,), (1,)))),
                               vertices)
            lower = values.min(axis=1); j = int(np.argmax(lower))
            if lower[j] > best:
                best = float(lower[j]); bestw = vertices[j].copy()
            upper = float(np.min(values.max(axis=0))) + padding  # convexity: q_i <= max over the vertices
            if upper > best:
                heapq.heappush(heap, (-upper, counter, vertices)); counter += 1

        add_cell(np.eye(3))
        target = lambda: best * (1 + self.rtol) ** 2 + padding
        while heap and -heap[0][0] > target() and splits < self.max_nodes:
            _, _, v = heapq.heappop(heap); a, b, c = v
            ab = (a + b) / 2; ac = (a + c) / 2; bc = (b + c) / 2
            for sub in ([a, ab, ac], [ab, b, bc], [ac, bc, c], [ab, bc, ac]):
                add_cell(np.asarray(sub))
            splits += 1
        upper = max(best, -heap[0][0] if heap else best)
        self.previous = [bestw.copy()]; self.solve_count += 1; self.total_splits += splits
        self.last_splits = splits
        self.last_relative_gap = float(np.sqrt(upper / max(best, 1e-300)) - 1)
        self._cached = (best * scale, bestw.copy(), upper * scale)
        return self._cached[0], bestw.copy(), self._cached[2]


class PeriodicStrongCCP:
    """CCP with the `weak` settings (nseeds, nstarts, maxiter) at every selection, and with the `strong`
    settings at selections period, 2 period, ..."""
    def __init__(self, K, period=10, weak=(128, 2, 30), strong=(1024, 8, 100), seed=42,
                 boundary_seeds=False, boundary_resolution=4, fresh_seeds=False):
        self.period = max(int(period), 1); self.calls = 0
        self.weak = CCP(K, nseeds=weak[0], nstarts=weak[1], seed=seed, maxiter=weak[2],
                        boundary_seeds=boundary_seeds, boundary_resolution=boundary_resolution, fresh_seeds=fresh_seeds)
        self.strong = CCP(K, nseeds=strong[0], nstarts=strong[1], seed=seed, maxiter=strong[2],
                          boundary_seeds=boundary_seeds, boundary_resolution=boundary_resolution, fresh_seeds=fresh_seeds)

    def add(self, Q):
        self.weak.add(Q); self.strong.add(Q)

    def solve(self):
        self.calls += 1
        use_strong = self.calls >= self.period and (self.calls - self.period) % self.period == 0
        return self.strong.solve() if use_strong else self.weak.solve()

    @property
    def lp_count(self):
        return self.weak.lp_count + self.strong.lp_count


def evaluate_gram(Q, seed=917, nseeds=2048, nstarts=8):
    """GN value recorded at training checkpoints: exact for K=2, a CCP lower estimate for K>3 (the
    reported K>3 value is the fixed-pool metric of mogym.metrics, computed afterwards)."""
    K = Q.shape[1]
    if K == 2:
        solver = Envelope()
        for q in Q:
            solver.add(q)
        val, w = solver.solve()
        return np.sqrt(val), w, np.sqrt(val)
    solver = CCP(K, nseeds, nstarts, seed)
    for q in Q:
        solver.add(q)
    val, w, upper = solver.solve()
    return np.sqrt(val), w, np.sqrt(upper)
