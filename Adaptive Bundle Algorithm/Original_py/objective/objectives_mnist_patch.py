"""objectives_mnist_patch.py — MNIST per-class CE objectives on a
patch-connected softplus MLP (AH16-style) + stochastic λ-scalarized oracle.

NEW FILE (Aug 9, 2026).  User-approved trial design (Aug-9 Q&A):
real data + softplus + patch first layer; no existing file modified.

Architecture (trial default, ``ah16_faithful=False``):

    input 784 (28x28 pixels, /255)
      -> patch layer: 64 units, each sees ONE 5x5 pixel block; block
         top-left corners on a deterministic 8x8 grid over [0, 23]^2
      -> softplus -> dense 96 -> softplus -> 10 output logits

    d = 64*(25+1) + 96*(64+1) + 10*(96+1) = 8874 parameters
    (ah16_faithful=True drops the dense layer: patch-64 -> 10, d = 2314).

Objectives: F_i(theta) = mean cross-entropy over class-i rows — the
same per-class definition as ``objectives_torch.make_mlp_nonconvex``.
Everything is L-smooth: patch/dense layers are linear, softplus is
smooth, CE is smooth.  L_i is estimated by random parameter-pair
probes (the original factory's recipe; the executor's L_scale
safeguard backstops any underestimate).

Data: MNIST train set (IDX format, torchvision's S3 mirror), first
``per_class`` rows of each class in dataset order (deterministic — no
rng), pixels /255, flattened to 784.  Cached in
``<repo>/Adaptive Bundle Algorithm/data/mnist/`` (~10 MB, one
download).

The stochastic oracle mirrors ``objectives_torch_fast.StochLamOracle``
verbatim (stratified per-class draws b_k ∝ n_k, persistent
current/anchor net pair, ifo_count += 2·rows) so the pure-budget
executor drives it unchanged.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from objectives_torch import _flatten_grads, _load_theta_into_net

_MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "mnist"
IMG, PATCH, GRID = 28, 5, 8          # image side, patch side, grid side
N_PATCH_UNITS = GRID * GRID          # 64


# =====================================================================
#  Data
# =====================================================================
def _fetch(fname: str) -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _DATA_DIR / fname
    if not path.exists():
        print(f"[mnist] downloading {fname} ...", flush=True)
        urllib.request.urlretrieve(_MNIST_MIRROR + fname, path)
    return path


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, = struct.unpack(">I", fh.read(4))
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, fh.read(4 * ndim))
        return np.frombuffer(fh.read(), dtype=np.uint8).reshape(dims)


def load_mnist_subset(per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    """First ``per_class`` rows of each digit in dataset order.

    Returns X (n, 784) float64 in [0, 1] and labels (n,) int64, rows
    grouped by class (0..9) — deterministic, no rng involved.
    """
    images = _read_idx(_fetch("train-images-idx3-ubyte.gz"))
    labels = _read_idx(_fetch("train-labels-idx1-ubyte.gz")).astype(np.int64)
    X_parts, y_parts = [], []
    for k in range(10):
        idx = np.nonzero(labels == k)[0][:per_class]
        if idx.size < per_class:
            raise ValueError(f"class {k}: only {idx.size} rows available.")
        X_parts.append(images[idx].reshape(idx.size, -1))
        y_parts.append(np.full(idx.size, k, dtype=np.int64))
    X = np.concatenate(X_parts).astype(np.float64) / 255.0
    return np.ascontiguousarray(X), np.concatenate(y_parts)


# =====================================================================
#  Patch network
# =====================================================================
def patch_indices() -> np.ndarray:
    """(64, 25) pixel indices: 8x8 grid of 5x5 blocks, deterministic."""
    corners = np.round(np.linspace(0, IMG - PATCH, GRID)).astype(int)
    idx = []
    for r0 in corners:
        for c0 in corners:
            block = [(r0 + r) * IMG + (c0 + c)
                     for r in range(PATCH) for c in range(PATCH)]
            idx.append(block)
    return np.asarray(idx, dtype=np.int64)


class PatchMLP(torch.nn.Module):
    """patch-64(5x5) -> [dense 96 ->] 10, softplus activations.

    Parameter iteration order: W1 (64,25), b1 (64), [fc.weight, fc.bias],
    out.weight, out.bias — fixed by attribute declaration order.
    """

    def __init__(self, ah16_faithful: bool = False):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.zeros(N_PATCH_UNITS, PATCH * PATCH,
                                                 dtype=torch.float64))
        self.b1 = torch.nn.Parameter(torch.zeros(N_PATCH_UNITS,
                                                 dtype=torch.float64))
        self.ah16_faithful = ah16_faithful
        if not ah16_faithful:
            self.fc = torch.nn.Linear(N_PATCH_UNITS, 96, dtype=torch.float64)
        self.out = torch.nn.Linear(N_PATCH_UNITS if ah16_faithful else 96,
                                   10, dtype=torch.float64)
        self.register_buffer("pidx", torch.from_numpy(patch_indices()))

    def forward(self, x):                       # x: (B, 784) float64
        patches = x[:, self.pidx]               # (B, 64, 25)
        z = torch.einsum("bup,up->bu", patches, self.W1) + self.b1
        z = F.softplus(z)
        if not self.ah16_faithful:
            z = F.softplus(self.fc(z))
        return self.out(z)                      # logits (B, 10)


def make_patch_initial_point(seed: int = 8,
                             ah16_faithful: bool = False) -> np.ndarray:
    """Flat He-style init matching PatchMLP's parameter order (hidden
    biases small positive, output bias zero — the repo's convention)."""
    rng = np.random.RandomState(seed)
    parts = [rng.randn(N_PATCH_UNITS, PATCH * PATCH).ravel()
             * np.sqrt(2.0 / (PATCH * PATCH)),
             np.full(N_PATCH_UNITS, 0.01)]
    if not ah16_faithful:
        parts += [rng.randn(96, N_PATCH_UNITS).ravel()
                  * np.sqrt(2.0 / N_PATCH_UNITS),
                  np.full(96, 0.01)]
        parts += [rng.randn(10, 96).ravel() * np.sqrt(2.0 / 96),
                  np.zeros(10)]
    else:
        parts += [rng.randn(10, N_PATCH_UNITS).ravel()
                  * np.sqrt(2.0 / N_PATCH_UNITS),
                  np.zeros(10)]
    return np.concatenate(parts)


# =====================================================================
#  Stochastic λ-scalarized oracle (mirror of StochLamOracle)
# =====================================================================
class PatchStochLamOracle:
    def __init__(self, X_np, labels_np, *, batch_size: int, seed: int,
                 ah16_faithful: bool = False):
        self.K = 10
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
        self.net_y = PatchMLP(ah16_faithful)
        self.net_a = PatchMLP(ah16_faithful)
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
#  Factory: full oracles + L probes + stochastic oracle
# =====================================================================
def make_mnist_patch(per_class: int = 1000, *, batch_size: int = 1024,
                     sampler_seed: int = 41, init_seed: int = 8,
                     n_probes: int = 40, probe_seed: int = 7,
                     ah16_faithful: bool = False):
    """Returns (objectives, grad_objectives, L, joint_oracle, stoch, meta)."""
    K = 10
    X_np, labels_np = load_mnist_subset(per_class)
    n = X_np.shape[0]
    X = torch.from_numpy(X_np)
    class_rows = [torch.from_numpy(np.nonzero(labels_np == k)[0]).long()
                  for k in range(K)]
    net = PatchMLP(ah16_faithful)
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

    # L_i by random parameter-pair probes (original factory's recipe):
    # pairs of He-scale random points, L_i = max ||dJ_i|| / ||dtheta||.
    rng = np.random.RandomState(probe_seed)
    L = np.zeros(K)
    for _ in range(n_probes):
        t1 = make_patch_initial_point(rng.randint(1 << 30), ah16_faithful) \
            + 0.5 * rng.randn(d) * 0.1
        t2 = t1 + 0.5 * rng.randn(d)
        _, J1 = joint_oracle(t1)
        _, J2 = joint_oracle(t2)
        denom = float(np.linalg.norm(t2 - t1))
        L = np.maximum(L, np.linalg.norm(J2 - J1, axis=1) / denom)
    stoch = PatchStochLamOracle(X_np, labels_np, batch_size=batch_size,
                                seed=sampler_seed,
                                ah16_faithful=ah16_faithful)
    meta = {"K": K, "n": n, "d": d, "per_class": per_class,
            "ah16_faithful": ah16_faithful}
    return objectives, grad_objectives, L, joint_oracle, stoch, meta
