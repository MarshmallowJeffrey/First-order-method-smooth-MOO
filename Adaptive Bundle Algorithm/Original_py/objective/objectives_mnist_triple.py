"""objectives_mnist_triple.py — K = 3 MNIST digit-triple per-class CE
objectives on the patch-connected softplus MLP + stochastic
λ-scalarized oracle.

NEW FILE (Aug 26, 2026).  User-approved K = 3 plan (Aug-25/26 Q&A):
the K = 2 pair campaign design (Aug-13 freeze) carried verbatim to a
digit TRIPLE — borrow ONLY the task/data idea from Reddi et al. 2016
(three MNIST digits, pixels /255, official train/test split, no
augmentation, no regularisation); network and algorithm stay the
repo's own.  No existing file is modified — ``objectives_mnist_pair``
is the structural template and ``objectives_mnist_patch`` is imported
for the loader primitives and patch geometry.

Deliberately NOT carried over from the pair file: the
``ah16_faithful`` conflict-ladder option.  The K = 2 Smoke A showed no
degeneration on any candidate pair, and every K = 3 candidate triple
is built from high-conflict pairs, so the ladder is not part of the
K = 3 design (Aug-25 decision).

Architecture (3-logit head; otherwise identical to the pair net):

    input 784 -> patch layer: 64 units, each sees ONE 5x5 block
      (8x8 grid of top-left corners over [0, 23]^2)
      -> softplus -> dense 96 -> softplus -> 3 logits

    d = 64*(25+1) + 96*(64+1) + 3*(96+1) = 8,195

Objectives: F_k(theta) = mean CE over digit_k rows (labels remapped
positionally to 0/1/2); s_k ≡ 1; NO regularisation term.  Everything
is L-smooth (linear maps + softplus + CE).

Data: MNIST train, first ``per_class`` rows of each of the three
digits in dataset order (deterministic, no rng).  ``per_class=None``
takes the balanced maximum min over the three counts — the Aug-13
"take all we have" convention (MNIST train holds 5,421-6,742 rows per
digit, so the usable maximum is triple-dependent).  The test loader
returns ALL official t10k rows of the three digits (unbalanced is
fine: per-class means never mix classes).

The stochastic oracle mirrors ``PairStochLamOracle`` verbatim
(stratified b_k ∝ n_k, persistent current/anchor net pair,
ifo += 2·rows) so the pure-budget executor drives it unchanged.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from objectives_torch import _flatten_grads, _load_theta_into_net
from objectives_mnist_patch import (
    N_PATCH_UNITS,
    PATCH,
    _fetch,
    _read_idx,
    patch_indices,
)

K_TRIPLE = 3


# =====================================================================
#  Data
# =====================================================================
def load_mnist_triple(digits: Sequence[int], per_class: int | None = None,
                      *, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Rows of the three digits, labels remapped positionally to 0/1/2.

    train=True: first ``per_class`` rows of each digit in dataset order
    (None -> balanced maximum min over the three counts).  train=False:
    ALL t10k rows of the three digits (per_class must be None).
    Returns X (n, 784) float64 in [0, 1] and labels (n,) int64, rows
    grouped digit-block by digit-block — deterministic, no rng involved.
    """
    digits = [int(v) for v in digits]
    if len(digits) != K_TRIPLE or len(set(digits)) != K_TRIPLE:
        raise ValueError(f"need three distinct digits, got {digits}.")
    stem = "train" if train else "t10k"
    images = _read_idx(_fetch(f"{stem}-images-idx3-ubyte.gz"))
    labels = _read_idx(_fetch(f"{stem}-labels-idx1-ubyte.gz")).astype(np.int64)
    idx = [np.nonzero(labels == dg)[0] for dg in digits]
    if train:
        take = (min(i.size for i in idx) if per_class is None
                else int(per_class))
        if any(i.size < take for i in idx):
            raise ValueError(f"triple {tuple(digits)}: only "
                             f"{tuple(i.size for i in idx)} rows available, "
                             f"need {take}/class.")
        idx = [i[:take] for i in idx]
    elif per_class is not None:
        raise ValueError("test loader always takes all rows.")
    X = np.concatenate([images[i].reshape(i.size, -1) for i in idx])
    X = np.ascontiguousarray(X.astype(np.float64) / 255.0)
    y = np.concatenate([np.full(i.size, k, dtype=np.int64)
                        for k, i in enumerate(idx)])
    return X, y


# =====================================================================
#  Triple network (3-logit head)
# =====================================================================
class TriplePatchMLP(torch.nn.Module):
    """patch-64(5x5) -> dense 96 -> 3, softplus activations.

    Parameter iteration order: W1 (64,25), b1 (64), fc.weight, fc.bias,
    out.weight, out.bias — fixed by attribute declaration order.
    """

    def __init__(self):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.zeros(N_PATCH_UNITS, PATCH * PATCH,
                                                 dtype=torch.float64))
        self.b1 = torch.nn.Parameter(torch.zeros(N_PATCH_UNITS,
                                                 dtype=torch.float64))
        self.fc = torch.nn.Linear(N_PATCH_UNITS, 96, dtype=torch.float64)
        self.out = torch.nn.Linear(96, K_TRIPLE, dtype=torch.float64)
        self.register_buffer("pidx", torch.from_numpy(patch_indices()))

    def forward(self, x):                       # x: (B, 784) float64
        patches = x[:, self.pidx]               # (B, 64, 25)
        z = torch.einsum("bup,up->bu", patches, self.W1) + self.b1
        z = F.softplus(z)
        z = F.softplus(self.fc(z))
        return self.out(z)                      # logits (B, 3)


def make_triple_initial_point(seed: int = 8) -> np.ndarray:
    """Flat He-style init matching TriplePatchMLP's parameter order
    (hidden biases small positive, output bias zero — the repo's
    convention)."""
    rng = np.random.RandomState(seed)
    parts = [rng.randn(N_PATCH_UNITS, PATCH * PATCH).ravel()
             * np.sqrt(2.0 / (PATCH * PATCH)),
             np.full(N_PATCH_UNITS, 0.01),
             rng.randn(96, N_PATCH_UNITS).ravel()
             * np.sqrt(2.0 / N_PATCH_UNITS),
             np.full(96, 0.01),
             rng.randn(K_TRIPLE, 96).ravel() * np.sqrt(2.0 / 96),
             np.zeros(K_TRIPLE)]
    return np.concatenate(parts)


# =====================================================================
#  Stochastic λ-scalarized oracle (verbatim mirror of PairStochLamOracle)
# =====================================================================
class TripleStochLamOracle:
    def __init__(self, X_np, labels_np, *, batch_size: int, seed: int):
        self.K = K_TRIPLE
        self.n = int(X_np.shape[0])
        self.batch_size = int(batch_size)
        self.rng = np.random.RandomState(seed)
        self.X = torch.from_numpy(np.ascontiguousarray(X_np))
        self.class_idx_np: List[np.ndarray] = [
            np.nonzero(labels_np == k)[0] for k in range(self.K)]
        if any(idx.size == 0 for idx in self.class_idx_np):
            raise ValueError("every class needs at least one sample.")
        n_k = np.array([i.size for i in self.class_idx_np], dtype=float)
        raw = batch_size * n_k / n_k.sum()
        b_k = np.maximum(1, np.floor(raw).astype(int))
        short = batch_size - int(b_k.sum())
        if short > 0:
            for j in np.argsort(-(raw - np.floor(raw)))[:short]:
                b_k[j] += 1
        self.b_k = np.minimum(b_k, n_k.astype(int))
        self.b_total = int(self.b_k.sum())
        self.net_y = TriplePatchMLP()
        self.net_a = TriplePatchMLP()
        self.d = int(sum(p.numel() for p in self.net_y.parameters()))
        self.ifo_count = 0

    def sample_batch(self) -> List[np.ndarray]:
        out = []
        for i in range(self.K):
            pool = self.class_idx_np[i]
            take = int(self.b_k[i])
            out.append(pool.copy() if take >= pool.size
                       else self.rng.choice(pool, size=take, replace=False))
        return out

    def full_batch(self) -> List[np.ndarray]:
        return [idx.copy() for idx in self.class_idx_np]

    def set_anchor(self, theta_a: np.ndarray) -> None:
        _load_theta_into_net(self.net_a, np.asarray(theta_a, dtype=float))

    def _scalarized_grad(self, net, lam, batch) -> np.ndarray:
        for param in net.parameters():
            param.grad = None
        rows = np.concatenate(batch)
        Z = net(self.X[torch.from_numpy(rows).long()])
        loss, off = None, 0
        for i in range(self.K):
            m_i = len(batch[i])
            if m_i == 0:
                continue
            Z_i = Z[off:off + m_i]
            off += m_i
            target = torch.full((m_i,), i, dtype=torch.long)
            term = float(lam[i]) * F.cross_entropy(Z_i, target,
                                                   reduction="mean")
            loss = term if loss is None else loss + term
        grads = torch.autograd.grad(loss, list(net.parameters()))
        return _flatten_grads(net, grads)

    def grad_pair(self, theta_y, lam, batch):
        _load_theta_into_net(self.net_y, np.asarray(theta_y, dtype=float))
        g_y = self._scalarized_grad(self.net_y, lam, batch)
        g_a = self._scalarized_grad(self.net_a, lam, batch)
        self.ifo_count += 2 * int(sum(len(b) for b in batch))
        return g_y, g_a


# =====================================================================
#  Fixed-theta evaluation (train OR test rows) — no gradients involved
# =====================================================================
def evaluate_triple(theta: np.ndarray, X_np: np.ndarray,
                    y_np: np.ndarray) -> dict:
    """Per-class mean CE and per-class error rate at a fixed theta.

    Pure forward evaluation (the test-front primitive): CE_k is the
    same formula as the training objective F_k, just on whatever rows
    are passed in; err_k = fraction of class-k rows with argmax != k
    (i.e. 1 - recall_k)."""
    net = TriplePatchMLP()
    _load_theta_into_net(net, np.asarray(theta, dtype=float))
    with torch.no_grad():
        Z = net(torch.from_numpy(np.ascontiguousarray(X_np)))
        out = {}
        for k in range(K_TRIPLE):
            rows = np.nonzero(y_np == k)[0]
            Zk = Z[torch.from_numpy(rows).long()]
            target = torch.full((rows.size,), k, dtype=torch.long)
            out[f"ce_{k}"] = float(F.cross_entropy(Zk, target,
                                                   reduction="mean"))
            out[f"err_{k}"] = float((Zk.argmax(dim=1) != target)
                                    .to(torch.float64).mean())
            out[f"n_{k}"] = int(rows.size)
    return out


# =====================================================================
#  Factory: full oracles + L probes + stochastic oracle
# =====================================================================
def make_mnist_triple(digits: Sequence[int],
                      per_class: int | None = None, *,
                      batch_size: int = 1024, sampler_seed: int = 41,
                      init_seed: int = 8, n_probes: int = 40,
                      probe_seed: int = 7):
    """Returns (objectives, grad_objectives, L, joint_oracle, stoch, meta).

    meta carries the raw arrays under "_X"/"_y" (pop before JSON) so
    callers can build fresh oracles with identical batch streams."""
    K = K_TRIPLE
    digits = [int(v) for v in digits]
    X_np, labels_np = load_mnist_triple(digits, per_class)
    n = X_np.shape[0]
    X = torch.from_numpy(X_np)
    class_rows = [torch.from_numpy(np.nonzero(labels_np == k)[0]).long()
                  for k in range(K)]
    net = TriplePatchMLP()
    d = int(sum(p.numel() for p in net.parameters()))

    def _per_class_losses(theta: np.ndarray):
        _load_theta_into_net(net, np.asarray(theta, dtype=float))
        Z = net(X)
        losses = []
        for k in range(K):
            Zk = Z[class_rows[k]]
            target = torch.full((len(class_rows[k]),), k, dtype=torch.long)
            losses.append(F.cross_entropy(Zk, target, reduction="mean"))
        return losses

    def joint_oracle(theta: np.ndarray):
        losses = _per_class_losses(theta)
        fvals = np.array([float(l.detach()) for l in losses])
        J = np.empty((K, d))
        for k in range(K):
            grads = torch.autograd.grad(losses[k], list(net.parameters()),
                                        retain_graph=(k < K - 1))
            J[k] = _flatten_grads(net, grads)
        return fvals, J

    def _obj(k):
        return lambda th: float(_per_class_losses(th)[k])

    def _grad(k):
        return lambda th: joint_oracle(th)[1][k]

    objectives = [_obj(k) for k in range(K)]
    grad_objectives = [_grad(k) for k in range(K)]

    # L_i by random parameter-pair probes (the patch factory's recipe):
    # pairs of He-scale random points, L_i = max ||dJ_i|| / ||dtheta||.
    rng = np.random.RandomState(probe_seed)
    L = np.zeros(K)
    for _ in range(n_probes):
        t1 = make_triple_initial_point(rng.randint(1 << 30)) \
            + 0.5 * rng.randn(d) * 0.1
        t2 = t1 + 0.5 * rng.randn(d)
        _, J1 = joint_oracle(t1)
        _, J2 = joint_oracle(t2)
        denom = float(np.linalg.norm(t2 - t1))
        L = np.maximum(L, np.linalg.norm(J2 - J1, axis=1) / denom)
    stoch = TripleStochLamOracle(X_np, labels_np, batch_size=batch_size,
                                 seed=sampler_seed)
    meta = {"K": K, "n": n, "d": d, "digits": digits,
            "per_class": int(n // K),
            "_X": X_np, "_y": labels_np}
    return objectives, grad_objectives, L, joint_oracle, stoch, meta
