"""Conflict screening of digit pairs and triples (lookahead affinity of Fifty et al. 2021, used in reverse).

For each candidate set of digits: one training run of 15 segments at equal weights (constant step 0.1 / L with
heavy-ball momentum 0.5, the first 5,421 images of each digit, mu = rho = 0).  At the checkpoints
t in {0, 3, ..., 15} (after an accepted segment) a trial step on class i, theta - (0.05 / L_i) grad L_i, gives

    Z_{i->j} = 1 - L_j(trial point) / L_j(theta_t),

C_{i->j} = mean over the checkpoints of max(0, -Z_{i->j}), c_j = (1/2) sum_{i != j} C_{i->j} and the score
C_bal = min_j c_j (ties broken by C_mean = mean_j c_j).

For pairs the Jacobian is computed after every segment; for triples only at the checkpoints and at the end (the
other anchors use the gradient of the equally weighted loss).  Both give the same gradient up to rounding; they are
kept as run.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_digits
from .model import PatchNet, flatten_grads, initial_point, load_theta
from .objective import StochasticOracle

PER_CLASS = 5421            # the smallest digit class of MNIST: the same data size for every candidate
N_SEG = 15
BATCH = 1024
STEP_CONST, MOMENTUM = 0.1, 0.5
MAX_RETRIES = 4
INIT_SEED, SAMPLER_SEED, PROBE_SEED = 8, 41, 7
N_PROBES = 10
ALPHA = 0.05
CHECKPOINTS = (0, 3, 6, 9, 12, 15)
EPS = 1e-12


class _FullBatch:
    """Plain per-class cross-entropies L_k on the full data (no pooling, no ridge)."""

    def __init__(self, X_np, y_np, K):
        self.K = K
        self.net = PatchNet(K)
        self.X = torch.from_numpy(np.ascontiguousarray(X_np))
        self.rows = [torch.from_numpy(np.nonzero(y_np == k)[0]).long() for k in range(K)]
        self.d = int(sum(p.numel() for p in self.net.parameters()))

    def _losses(self, theta):
        load_theta(self.net, np.asarray(theta, dtype=float))
        Z = self.net(self.X)
        return [F.cross_entropy(Z[self.rows[k]], torch.full((len(self.rows[k]),), k, dtype=torch.long),
                                reduction="mean") for k in range(self.K)]

    def values(self, theta):
        with torch.no_grad():
            return np.array([float(v) for v in self._losses(theta)])

    def joint(self, theta):
        losses = self._losses(theta)
        fv = np.array([float(v.detach()) for v in losses])
        J = np.empty((self.K, self.d))
        for k in range(self.K):
            J[k] = flatten_grads(torch.autograd.grad(losses[k], list(self.net.parameters()),
                                                     retain_graph=(k < self.K - 1)))
        return fv, J

    def scalarized_grad(self, theta, lam):
        losses = self._losses(theta)
        scal = sum(float(w) * v for w, v in zip(lam, losses))
        return flatten_grads(torch.autograd.grad(scal, list(self.net.parameters())))


def _estimate_L(problem):
    rng = np.random.RandomState(PROBE_SEED)
    L = np.zeros(problem.K)
    for _ in range(N_PROBES):
        t1 = initial_point(problem.K, rng.randint(1 << 30)) + 0.5 * rng.randn(problem.d) * 0.1
        t2 = t1 + 0.5 * rng.randn(problem.d)
        _, J1 = problem.joint(t1)
        _, J2 = problem.joint(t2)
        L = np.maximum(L, np.linalg.norm(J2 - J1, axis=1) / float(np.linalg.norm(t2 - t1)))
    return L


def _lookahead(problem, theta, fv, J, L):
    K = problem.K
    Z = np.zeros((K, K))
    for i in range(K):
        f_probe = problem.values(theta - (ALPHA / L[i]) * J[i])
        for j in range(K):
            if i != j:
                Z[i, j] = 1.0 - f_probe[j] / max(fv[j], EPS)
    return Z


def screen(digits):
    """Returns the record of one candidate: Z at the checkpoints, C, c_j, C_bal, C_mean."""
    t0 = time.time()
    digits = tuple(int(v) for v in digits)
    K = len(digits)
    lam = np.full(K, 1.0 / K)
    jacobian_every_segment = (K == 2)
    X_np, y_np = load_digits(digits, PER_CLASS)
    problem = _FullBatch(X_np, y_np, K)
    L = _estimate_L(problem)
    stoch = StochasticOracle(X_np, y_np, K, batch_size=BATCH, seed=SAMPLER_SEED, rho=0.0, mu=0.0,
                             device=torch.device("cpu"))
    epoch_len = max(1, int(np.ceil(X_np.shape[0] / float(BATCH))))
    x = initial_point(K, INIT_SEED)
    f, J = problem.joint(x)
    x = x.copy()
    series = [(0, _lookahead(problem, x, f, J, L))]
    L_scale, retries = 1.0, 0
    for seg in range(1, N_SEG + 1):
        g_a_full = J.T @ lam if J is not None else problem.scalarized_grad(x, lam)
        F_a = float(f @ lam)
        eta = STEP_CONST / (float(lam @ L) * L_scale)
        stoch.set_anchor(x)
        y = x.copy()
        u = np.zeros(problem.d)
        for _ in range(epoch_len):
            g_y_S, g_a_S = stoch.grad_pair(y, lam, stoch.sample_batch())
            u = MOMENTUM * u + (g_y_S - g_a_S + g_a_full)
            y = y - eta * u
        if jacobian_every_segment or seg in CHECKPOINTS or seg == N_SEG:
            f_y, J_y = problem.joint(y)
        else:
            f_y, J_y = problem.values(y), None
        moved = True
        if float(f_y @ lam) > F_a + 1e-10 * (1.0 + abs(F_a)):
            L_scale *= 2.0
            retries += 1
            moved = retries > MAX_RETRIES
        if moved:
            x, f, J, retries = y, f_y, J_y, 0
        if seg in CHECKPOINTS and moved:
            series.append((seg, _lookahead(problem, x, f, J, L)))
    stack = np.stack([Z for _, Z in series])
    Zbar = stack.mean(axis=0)
    C = np.maximum(0.0, -stack).mean(axis=0)
    for M in (Zbar, C):
        np.fill_diagonal(M, 0.0)
    c_j = 0.5 * C.sum(axis=0)
    return {"digits": list(digits), "per_class": PER_CLASS, "L": [float(v) for v in L],
            "checkpoints": [int(t) for t, _ in series], "Z": [Z.tolist() for _, Z in series],
            "Zbar": Zbar.tolist(), "C": C.tolist(), "c_j": c_j.tolist(),
            "C_bal": float(c_j.min()), "C_mean": float(c_j.mean()), "seconds": round(time.time() - t0, 1)}


def candidates(K):
    return list(itertools.combinations(range(10), K))


def ranking(records):
    """Most conflicting first: by C_bal, ties by C_mean."""
    return sorted(records, key=lambda r: (-r["C_bal"], -r["C_mean"]))
