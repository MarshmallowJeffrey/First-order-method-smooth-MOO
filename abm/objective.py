"""The pooled objectives of the MNIST experiments and their oracles.

For the digits (d_1, ..., d_K), L_k is the mean cross-entropy on the images of class k and L_pool the mean
cross-entropy on all images.  Objective k is

    F_k(theta) = L_k(theta) + rho * L_pool(theta) + (mu/2) ||theta||^2,

so F = A L + ridge with A = I + rho * 1 pi^T, pi_k = n_k / n (the class fractions).  The scalarized objective is
F_lambda = sum_k lambda_k F_k for lambda in the simplex.

Oracles
-------
* ``Problem.joint(theta)``: all K values and the K x d Jacobian on the full data (one "joint" call, which the
  budget charges K gradient calls).
* ``Problem.stoch``: the stochastic oracle of the SVRG steps.  A mini-batch holds b_k ~ b n_k / n images of each
  class, drawn without replacement from one RandomState(sampler_seed) stream; ``grad_pair(y, lam, batch)`` returns
  the mini-batch gradients of F_lambda at y and at the anchor on the same batch.  The data terms carry the class
  weights A^T lambda = lambda + rho * sum(lambda) * pi (every class keeps a positive weight), the ridge term is
  added once.  Every row of a batch is charged, for every lambda.
* ``Problem.L``: smoothness estimates of the F_k (40 random parameter pairs); only the constant-step, BB and
  AdaGrad rules use them.

Theta, gradients and Jacobians are NumPy float64 arrays on the host; the data and the networks live on
``device`` ("cpu" or "cuda"; Apple MPS has no float64).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_digits
from .model import PatchNet, flatten_grads, initial_point, load_theta


def resolve_device(device: str = "cpu") -> torch.device:
    d = str(device).lower()
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("mps"):
        raise ValueError("Apple MPS has no float64 support; use cpu or cuda.")
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available.")
    return torch.device(d)


def device_description(dev: torch.device) -> str:
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} {torch.cuda.get_device_name(idx)}"
    return f"cpu ({torch.get_num_threads()} torch threads)"


def class_fractions(labels: np.ndarray, K: int) -> np.ndarray:
    counts = np.array([(np.asarray(labels) == k).sum() for k in range(K)], dtype=float)
    return counts / counts.sum()


def mixing_matrix(K: int, rho: float, fractions: np.ndarray) -> np.ndarray:
    """A = I + rho * 1 pi^T, so that F = A L (+ ridge)."""
    return np.eye(K) + float(rho) * np.outer(np.ones(K), np.asarray(fractions, float))


def effective_weights(lam: np.ndarray, rho: float, fractions: np.ndarray) -> np.ndarray:
    """A^T lambda: the weights of the class losses in F_lambda."""
    lam = np.asarray(lam, dtype=float)
    return lam + float(rho) * float(lam.sum()) * np.asarray(fractions, float)


class StochasticOracle:
    """Stratified mini-batches and the SVRG gradient pair of F_lambda (see the module docstring)."""

    def __init__(self, X_np, labels_np, K: int, *, batch_size: int, seed: int, rho: float, mu: float,
                 device: torch.device):
        self.K = int(K)
        self.n = int(X_np.shape[0])
        self.batch_size = int(batch_size)
        self.rho, self.mu = float(rho), float(mu)
        self.device = device
        self.rng = np.random.RandomState(seed)
        self.X = torch.from_numpy(np.ascontiguousarray(X_np)).to(device)
        self.class_idx: List[np.ndarray] = [np.nonzero(labels_np == k)[0] for k in range(self.K)]
        n_k = np.array([i.size for i in self.class_idx], dtype=float)
        raw = batch_size * n_k / n_k.sum()
        b_k = np.maximum(1, np.floor(raw).astype(int))
        short = batch_size - int(b_k.sum())
        if short > 0:
            for j in np.argsort(-(raw - np.floor(raw)))[:short]:
                b_k[j] += 1
        self.b_k = np.minimum(b_k, n_k.astype(int))
        self.fractions = class_fractions(labels_np, self.K)
        self.net_y = PatchNet(self.K).to(device)
        self.net_a = PatchNet(self.K).to(device)
        self.theta_a = None
        self.ifo_count = 0                  # rows consumed (two gradients per row and step)

    def sample_batch(self) -> List[np.ndarray]:
        out = []
        for i in range(self.K):
            pool, take = self.class_idx[i], int(self.b_k[i])
            out.append(pool.copy() if take >= pool.size else self.rng.choice(pool, size=take, replace=False))
        return out

    def set_anchor(self, theta_a: np.ndarray) -> None:
        theta_a = np.asarray(theta_a, dtype=float)
        self.theta_a = theta_a.copy()
        load_theta(self.net_a, theta_a)

    def _data_grad(self, net, weights, batch) -> np.ndarray:
        for param in net.parameters():
            param.grad = None
        rows = np.concatenate(batch)
        Z = net(self.X[torch.from_numpy(rows).long().to(self.device)])
        loss, off = None, 0
        for i in range(self.K):
            m_i = len(batch[i])
            if m_i == 0:
                continue
            target = torch.full((m_i,), i, dtype=torch.long, device=self.device)
            term = float(weights[i]) * F.cross_entropy(Z[off:off + m_i], target, reduction="mean")
            off += m_i
            loss = term if loss is None else loss + term
        return flatten_grads(torch.autograd.grad(loss, list(net.parameters())))

    def grad_pair(self, theta_y: np.ndarray, lam: np.ndarray, batch):
        """Mini-batch gradients of F_lambda at theta_y and at the anchor (same batch)."""
        weights = effective_weights(lam, self.rho, self.fractions) if self.rho != 0.0 else lam
        theta_y = np.asarray(theta_y, dtype=float)
        load_theta(self.net_y, theta_y)
        g_y = self._data_grad(self.net_y, weights, batch)
        g_a = self._data_grad(self.net_a, weights, batch)
        self.ifo_count += 2 * int(sum(len(b) for b in batch))
        if self.mu != 0.0:
            g_y = g_y + self.mu * theta_y
            g_a = g_a + self.mu * self.theta_a
        return g_y, g_a


@dataclass
class Problem:
    digits: tuple
    K: int
    n: int
    d: int
    rho: float
    mu: float
    L: np.ndarray
    joint: Callable            # theta -> (F values (K,), Jacobian (K, d)), full data
    stoch: StochasticOracle
    device_description: str


def make_problem(digits, rho: float, mu: float, *, per_class: int | None = None, batch_size: int = 1024,
                 sampler_seed: int = 41, n_probes: int = 40, probe_seed: int = 7, device: str = "cpu") -> Problem:
    dev = resolve_device(device)
    digits = tuple(int(v) for v in digits)
    K = len(digits)
    rho, mu = float(rho), float(mu)
    X_np, labels_np = load_digits(digits, per_class)
    n = X_np.shape[0]
    X = torch.from_numpy(X_np).to(dev)
    class_rows = [torch.from_numpy(np.nonzero(labels_np == k)[0]).long().to(dev) for k in range(K)]
    targets = [torch.full((int(class_rows[k].shape[0]),), k, dtype=torch.long, device=dev) for k in range(K)]
    net = PatchNet(K).to(dev)
    d = int(sum(p.numel() for p in net.parameters()))

    def class_losses_and_jacobian(theta):
        """L_k and their gradients (plain cross-entropy per class, full data)."""
        load_theta(net, np.asarray(theta, dtype=float))
        Z = net(X)
        losses = [F.cross_entropy(Z[class_rows[k]], targets[k], reduction="mean") for k in range(K)]
        values = np.array([float(v.detach()) for v in losses])
        J = np.empty((K, d))
        for k in range(K):
            J[k] = flatten_grads(torch.autograd.grad(losses[k], list(net.parameters()), retain_graph=(k < K - 1)))
        return values, J

    A = mixing_matrix(K, rho, class_fractions(labels_np, K))

    def joint(theta):
        f0, J0 = class_losses_and_jacobian(theta)
        th = np.asarray(theta, dtype=float)
        return A @ f0 + 0.5 * mu * float(th @ th), A @ J0 + mu * th

    # smoothness estimates: largest gradient difference over random parameter pairs, plus mu
    rng = np.random.RandomState(probe_seed)
    L = np.zeros(K)
    for _ in range(int(n_probes)):
        t1 = initial_point(K, rng.randint(1 << 30)) + 0.5 * rng.randn(d) * 0.1
        t2 = t1 + 0.5 * rng.randn(d)
        _, J1 = class_losses_and_jacobian(t1)
        _, J2 = class_losses_and_jacobian(t2)
        L = np.maximum(L, np.linalg.norm(A @ (J2 - J1), axis=1) / float(np.linalg.norm(t2 - t1)))
    L = L + mu

    stoch = StochasticOracle(X_np, labels_np, K, batch_size=batch_size, seed=sampler_seed, rho=rho, mu=mu,
                             device=dev)
    return Problem(digits=digits, K=K, n=n, d=d, rho=rho, mu=mu, L=L, joint=joint, stoch=stoch,
                   device_description=device_description(dev))
