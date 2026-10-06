"""Exact lower envelope of the K = 2 parabolas and its maximizer (the envelope selection of Appendix A.4.1 of the
paper: Algorithms 3-5).

With lambda(s) = (1 - s, s), s in [0, 1], bundle point i gives the convex parabola

    phi_i(s) = lambda(s)^T Q_i lambda(s) = a_i s^2 + b_i s + c_i,
    a_i = Q11 - 2 Q12 + Q22 >= 0,   b_i = 2 (Q12 - Q11),   c_i = Q11,

and phi(s) = min_i phi_i(s) is their lower envelope, stored as breakpoints 0 = sigma_0 < ... < sigma_J = 1 and labels
w_1, ..., w_J (phi = phi_{w_j} on [sigma_{j-1}, sigma_j]).  Every piece is convex, so the maximum of phi is attained
at a breakpoint (Lemma 4): the selection evaluates phi at all breakpoints, sigma_0 = 0 and sigma_J = 1 included.

envelope() is ENV (Algorithm 4: divide and conquer) and merge() is MERGE (Algorithm 5).  EnvelopeSelector keeps the
envelope of the bundle during a run and merges the envelope of the new points into it at every decision.  The lower
envelope is unique, so this gives the envelope that ENV would rebuild from scratch, at a cost linear in its size.
Ties are broken by the smaller bundle index (labels) and by the smaller s (maximizer).
"""

from __future__ import annotations

import numpy as np

ROOT_TOL = 1e-13          # crossings this close to a segment end are dropped: the end is already a breakpoint


def coeffs(Ms):
    """(a, b, c) of phi_i(s) = a_i s^2 + b_i s + c_i for the Gram matrices Ms (m x 2 x 2)."""
    Ms = np.asarray(Ms, dtype=float).reshape(-1, 2, 2)
    q11, q12, q22 = Ms[:, 0, 0], Ms[:, 0, 1], Ms[:, 1, 1]
    return q11 - 2.0 * q12 + q22, 2.0 * (q12 - q11), q11


def _values(a, b, c, idx, s):
    return (a[idx] * s + b[idx]) * s + c[idx]


def _crossings(da, db, dc):
    """The real roots of da s^2 + db s + dc (per row; nan where there is none), computed without cancellation."""
    r1 = np.full(da.shape, np.nan)
    r2 = np.full(da.shape, np.nan)
    with np.errstate(all="ignore"):
        disc = db * db - 4.0 * da * dc
        quad = (da != 0.0) & (disc >= 0.0)
        sq = np.sqrt(np.where(quad, disc, 0.0))
        qq = -0.5 * (db + np.where(db >= 0.0, sq, -sq))
        r1 = np.where(quad & (qq != 0.0), qq / np.where(da != 0.0, da, 1.0), r1)
        r2 = np.where(quad & (qq != 0.0), dc / np.where(qq != 0.0, qq, 1.0), r2)
        lin = (da == 0.0) & (db != 0.0)
        r1 = np.where(lin, -dc / np.where(db != 0.0, db, 1.0), r1)
    return r1, r2


def merge(s1, w1, s2, w2, a, b, c):
    """MERGE (Algorithm 5): the lower envelope of two envelopes (breakpoints s, labels w; indices into a, b, c)."""
    r = np.union1d(s1, s2)                                      # both contain 0 and 1
    lo, hi = r[:-1], r[1:]
    mid = 0.5 * (lo + hi)
    p = w1[np.searchsorted(s1, mid, side="right") - 1]
    q = w2[np.searchsorted(s2, mid, side="right") - 1]
    x1, x2 = _crossings(a[p] - a[q], b[p] - b[q], c[p] - c[q])
    tol = ROOT_TOL * np.maximum(1.0, hi - lo)
    inside = []
    for x in (x1, x2):
        ok = np.isfinite(x) & (x > lo + tol) & (x < hi - tol)
        inside.append(x[ok])
    s = np.unique(np.concatenate([r] + inside))
    # label every new segment by the lower of its two candidate parabolas at the segment midpoint
    m2 = 0.5 * (s[:-1] + s[1:])
    seg = np.searchsorted(r, m2, side="right") - 1
    pp, qq = p[seg], q[seg]
    vp, vq = _values(a, b, c, pp, m2), _values(a, b, c, qq, m2)
    lab = np.where(vp < vq, pp, np.where(vq < vp, qq, np.minimum(pp, qq)))
    # delete every breakpoint whose two neighbouring segments have the same label
    keep = np.ones(s.size, dtype=bool)
    keep[1:-1] = lab[1:] != lab[:-1]
    s = s[keep]
    first = np.concatenate([[True], lab[1:] != lab[:-1]])
    return s, lab[first]


def envelope(idx, a, b, c):
    """ENV (Algorithm 4): the lower envelope of the parabolas idx by divide and conquer."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 1:
        return np.array([0.0, 1.0]), idx.copy()
    h = idx.size // 2
    return merge(*envelope(idx[:h], a, b, c), *envelope(idx[h:], a, b, c), a, b, c)


def envelope_max(s, w, a, b, c):
    """(max of phi, its argmax s*) over the breakpoints of the envelope (Lemma 4).  The value at a breakpoint is the
    smaller of the two neighbouring pieces (they agree up to rounding at a crossing); ties go to the smaller s."""
    left = np.concatenate([[np.inf], _values(a, b, c, w, s[1:])])        # piece on the left of breakpoint j
    right = np.concatenate([_values(a, b, c, w, s[:-1]), [np.inf]])      # piece on the right of breakpoint j
    v = np.minimum(left, right)
    j = int(np.argmax(v))
    return float(v[j]), float(s[j])


def envelope_eval(s, w, a, b, c, x):
    """phi at the points x (for checks)."""
    x = np.asarray(x, dtype=float)
    j = np.clip(np.searchsorted(s, x, side="right") - 1, 0, w.size - 1)
    return _values(a, b, c, w[j], x)


class EnvelopeSelector:
    """Step 1 of GRAB for K = 2 with the exact envelope.  solve(grams) takes the bundle's Gram matrices (all of them,
    in bundle order), merges the new ones into the stored envelope and returns (phi(lambda*), lambda*) with
    lambda* = (1 - s*, s*): the same interface as CCPCGSelector.solve (abm/ccp_cg.py)."""

    def __init__(self):
        self.n = 0
        self.a = np.empty(1024)
        self.b = np.empty(1024)
        self.c = np.empty(1024)
        self.s = None
        self.w = None

    def _grow(self, need):
        if need > self.a.size:
            size = max(need, 2 * self.a.size)
            for name in ("a", "b", "c"):
                arr = np.empty(size)
                arr[:self.n] = getattr(self, name)[:self.n]
                setattr(self, name, arr)

    def update(self, grams):
        m = len(grams)
        if m <= self.n:
            return
        a, b, c = coeffs(np.asarray(grams[self.n:m], dtype=float))
        self._grow(m)
        self.a[self.n:m], self.b[self.n:m], self.c[self.n:m] = a, b, c
        A, B, C = self.a[:m], self.b[:m], self.c[:m]
        s_new, w_new = envelope(np.arange(self.n, m), A, B, C)
        if self.s is None:
            self.s, self.w = s_new, w_new
        else:
            self.s, self.w = merge(self.s, self.w, s_new, w_new, A, B, C)
        self.n = m

    def solve(self, grams):
        self.update(grams)
        value, s_star = envelope_max(self.s, self.w, self.a[:self.n], self.b[:self.n], self.c[:self.n])
        return value, np.array([1.0 - s_star, s_star])

    @property
    def pieces(self):
        return 0 if self.w is None else int(self.w.size)
