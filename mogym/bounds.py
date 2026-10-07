"""Upper bounds on GRAB's worst-case gradient norm max_lambda GN(lambda, B_t) at every checkpoint (K > 2), by one
simplicial branch and bound carried along the checkpoints.

Bound on a subsimplex.  On a subsimplex S with vertices v_1, ..., v_K, each phi_i(lambda) = lambda' Q_i lambda is
convex, so phi_i(sum_j mu_j v_j) <= sum_j mu_j phi_i(v_j), and max_S min_i phi_i is at most the value of the LP
max_mu min_i (A^S mu)_i with A^S_ij = phi_i(v_j).  Every distribution w over the rows bounds that value by
max_j (w' A^S)_j; w is the LP dual projected onto the simplex, so the bound does not depend on the accuracy of the LP
solution.  The leaves of the branch and bound partition the simplex, and the bound of a bundle is the largest leaf
bound.

Along the checkpoints.  The bundle only grows, so a leaf bound computed for an earlier bundle stays valid (its w,
extended by zeros, is still a distribution over the rows).  At checkpoint k (bundle size n_k, lower estimate l_k of
mogym.metrics) the leaves of largest bound are taken in batches: a leaf bounded for an earlier bundle is re-bounded for
n_k, otherwise it is split at the midpoint of its longest edge and both halves are bounded; a leaf keeps the smaller of
its own bound and its parent's.  The checkpoint stops when the largest bound is at most ((1 + gap) l_k)^2 or its time
is spent.  Leaves whose bound is at most ((1 + gap) min_k l_k)^2 would never be split again and are dropped.  The
reported bound is max(sqrt(largest remaining bound), (1 + gap) min_k l_k), times 1 + GUARD against rounding (float64
Gram matrices, LP duals and vertices); it is non-increasing in k.  The stored leaves are capped; a split adds one
leaf, so a batch splits at most as many leaves as there are free slots, and a checkpoint stops when nothing is left to
split or re-bound.

The leaf bounds are normalized by (min_k l_k)^2.  The time limit makes the bounds depend on the machine; with the
settings of mogym.config most checkpoints stop at the gap.
"""
import heapq
import multiprocessing as mp
import time

import numpy as np

from . import lambda_solvers as ls

GUARD = 1e-9
_Q = None


def _init(npz, scale):
    global _Q
    _Q = np.load(npz)["J"]
    _Q = _Q @ _Q.transpose(0, 2, 1) / scale


def _bound(V, n):
    A = np.einsum("jk,nkl,jl->nj", V, _Q[:n], V)
    return float(ls.lp(A, return_dual_bound=True)[2])


def _split(V):
    d = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    a, b = np.unravel_index(np.argmax(d), d.shape)
    mid = (V[a] + V[b]) / 2
    out = []
    for replaced in (a, b):
        C = V.copy(); C[replaced] = mid; out.append(C)
    return out


def _work(args):
    tasks, n = args
    out = []
    for kind, V in tasks:
        if kind == "rebound":
            out.append((_bound(V, n),))
        else:
            C1, C2 = _split(V)
            out.append((C1, _bound(C1, n), C2, _bound(C2, n)))
    return out


def upper_bounds(npz, checkpoints, lower, *, seconds, gap, workers, max_leaves, log=None):
    """checkpoints: the GRAB checkpoints (dicts with "bundle_size", "component_gradients"), lower: the lower estimate
    at each.  Returns one dict per checkpoint: upper (the bound on max GN), leaves, lps, seconds, stopped ("gap",
    "time" or "leaves")."""
    lmin = float(min(lower)); scale = lmin ** 2; drop = (1 + gap) ** 2
    _init(npz, scale); K = _Q.shape[1]
    verts = np.empty((max_leaves, K, K)); bnd = np.empty(max_leaves); nsz = np.zeros(max_leaves, dtype=np.int64)
    free = list(range(max_leaves - 1, -1, -1)); heap = []; rows = []

    def add(V, b, n):
        if b <= drop:
            return
        i = free.pop(); verts[i] = V; bnd[i] = b; nsz[i] = n
        heapq.heappush(heap, (-b, i))

    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(npz, scale)) as pool:
        for k, c in enumerate(checkpoints):
            n = int(c["bundle_size"]); target = ((1 + gap) * lower[k]) ** 2 / scale
            t0 = time.time(); lps = 0; stopped = "gap"
            if k == 0:
                add(np.eye(K), _bound(np.eye(K), n), n); lps += 1
            while heap and -heap[0][0] > target:
                if time.time() >= t0 + seconds:
                    stopped = "time"; break
                batch, tasks, held, nsplit = [], [], [], len(free)
                while heap and -heap[0][0] > target and len(batch) + len(held) < 32 * workers:
                    i = heapq.heappop(heap)[1]
                    if nsz[i] < n:
                        batch.append(i); tasks.append(("rebound", verts[i].copy()))
                    elif nsplit > 0:
                        batch.append(i); tasks.append(("split", verts[i].copy())); nsplit -= 1
                    else:
                        held.append(i)
                for i in held:
                    heapq.heappush(heap, (-bnd[i], i))
                if not batch:
                    stopped = "leaves"; break
                results = pool.map(_work, [(tasks[j::workers], n) for j in range(workers)])
                for j, res in enumerate(results):
                    for (kind, _), out, i in zip(tasks[j::workers], res, batch[j::workers]):
                        parent = bnd[i]; free.append(i)
                        if kind == "rebound":
                            add(verts[i].copy(), min(parent, out[0]), n); lps += 1
                        else:
                            add(out[0], min(parent, out[1]), n); add(out[2], min(parent, out[3]), n); lps += 2
            top = -heap[0][0] if heap else 0.
            upper = float(np.sqrt(max(top, drop) * scale)) * (1 + GUARD)
            if upper < lower[k] * (1 - 1e-9):
                raise RuntimeError(f"upper bound below the lower estimate at checkpoint {k}")
            rows.append(dict(upper=upper, leaves=len(heap), lps=lps, seconds=time.time() - t0, stopped=stopped))
            if log is not None:
                log(f"{c['component_gradients']:7d} calls  n {n:5d}  lower {lower[k]:.4e}  upper {upper:.4e}  "
                    f"ratio {upper / lower[k]:.3f}  leaves {len(heap):7d}  {time.time() - t0:5.1f} s  ({stopped})")
    return rows
