"""objectives_mnist_patch_k10.py — K = 10 MNIST all-digits objectives on
the patch-softplus MLP, for the fixed-budget campaign (Sep 2026).

NEW FILE (Sep 10, 2026).  The Aug-9 base module ``objectives_mnist_patch``
is NOT modified; this file reuses its loader, network and initial point
and adds what the campaign needs:

* **balanced-max data**: ``per_class`` defaults to 5,421 rows per digit
  (digit 5 is the smallest class of the MNIST training set), n = 54,210,
  the same balanced convention as the K = 2 / K = 3 campaigns;
* **ridge penalty**: F_k^mu(theta) = F_k(theta) + (mu/2)·||theta||^2 on
  every parameter (biases included), the K = 3 convention
  (``objectives_mnist_triple_ridge``): since sum(lam) = 1 the penalty
  enters every scalarisation exactly once, its gradient mu·theta is
  deterministic and touches no data rows (budget accounting unchanged),
  and L_k + mu is a valid smoothness constant;
* **per-class forward in the joint oracle**: F_k depends only on the rows
  of digit k, so the K gradients are computed as K independent
  forward+backward passes over n_k rows each (1 + 1 passes per class)
  instead of one forward over all n rows followed by K backward passes
  through the whole graph.  Same numbers (verified Sep 10: fvals
  bit-identical, Jacobian within 1e-16), 7x faster at n = 54,210;
* **device selection**: ``device="auto" | "cpu" | "cuda"``.  Data and the
  two networks live on the device; theta, gradients, the budget meter
  and the outer algorithm stay in NumPy on the host (one host-to-device
  copy of theta and one device-to-host copy of the gradient per call —
  71 KB each at d = 8,874).  float64 everywhere on every device; Apple
  MPS has no float64 and is rejected.

Everything is L-smooth (linear patch/dense layers, softplus, CE, ridge).
The stochastic oracle keeps the base class's batch stream (same
``RandomState`` protocol, stratified b_k ∝ n_k) so a CPU run with mu = 0
reproduces the Aug-9 base oracle exactly.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from objectives_mnist_patch import (  # noqa: E402
    PatchMLP,
    _fetch,
    _read_idx,
    load_mnist_subset,
    make_patch_initial_point,
)
from objectives_torch import _flatten_grads, _load_theta_into_net  # noqa: E402

K = 10
BALANCED_MAX_PER_CLASS = 5421     # digit 5 has 5,421 training rows


# =====================================================================
#  Device
# =====================================================================
def resolve_device(device: str = "auto") -> torch.device:
    """'auto' -> cuda if available else cpu.  MPS is refused (no float64)."""
    d = str(device).lower()
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("mps"):
        raise ValueError("Apple MPS has no float64 support; use cpu or cuda.")
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False.")
    return torch.device(d)


def device_description(dev: torch.device) -> str:
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} {torch.cuda.get_device_name(idx)}"
    return f"cpu ({torch.get_num_threads()} torch threads)"


# =====================================================================
#  Data
# =====================================================================
def load_mnist_k10(per_class: int = BALANCED_MAX_PER_CLASS,
                   train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Training rows: first ``per_class`` rows of every digit (dataset
    order, deterministic).  Test rows (``train=False``): ALL official
    t10k rows, grouped by class — kept for a later test-side evaluation
    of the saved theta stacks."""
    if train:
        return load_mnist_subset(per_class)
    images = _read_idx(_fetch("t10k-images-idx3-ubyte.gz"))
    labels = _read_idx(_fetch("t10k-labels-idx1-ubyte.gz")).astype(np.int64)
    order = np.concatenate([np.nonzero(labels == k)[0] for k in range(K)])
    X = images[order].reshape(order.size, -1).astype(np.float64) / 255.0
    return np.ascontiguousarray(X), labels[order]


# =====================================================================
#  Stochastic λ-scalarized oracle (device-aware, ridge-aware)
# =====================================================================
class PatchStochLamOracleK10:
    """Mirror of ``objectives_mnist_patch.PatchStochLamOracle`` with the
    data and both networks on ``device`` and the ridge gradient added
    at each network's own theta (anchor theta recorded by ``set_anchor``).

    Batch protocol (unchanged): stratified per-class draws b_k ∝ n_k from
    one ``RandomState(seed)`` stream; ``ifo_count += 2·rows`` per pair.
    """

    def __init__(self, X_np, labels_np, *, batch_size: int, seed: int,
                 mu: float = 0.0, device: torch.device = torch.device("cpu"),
                 ah16_faithful: bool = False):
        self.K = K
        self.n = int(X_np.shape[0])
        self.batch_size = int(batch_size)
        self.mu = float(mu)
        self.device = device
        self.rng = np.random.RandomState(seed)
        self.X = torch.from_numpy(np.ascontiguousarray(X_np)).to(device)
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
        self.net_y = PatchMLP(ah16_faithful).to(device)
        self.net_a = PatchMLP(ah16_faithful).to(device)
        self.d = int(sum(p.numel() for p in self.net_y.parameters()))
        self._theta_a: Optional[np.ndarray] = None
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
        theta_a = np.ascontiguousarray(np.asarray(theta_a, dtype=float))
        _load_theta_into_net(self.net_a, theta_a)
        self._theta_a = theta_a.copy()

    def _scalarized_grad(self, net, lam, batch) -> np.ndarray:
        for param in net.parameters():
            param.grad = None
        rows = np.concatenate(batch)
        idx = torch.from_numpy(np.ascontiguousarray(rows)).long().to(self.device)
        Z = net(self.X[idx])
        loss, off = None, 0
        for i in range(self.K):
            m_i = len(batch[i])
            if m_i == 0:
                continue
            Z_i = Z[off:off + m_i]
            off += m_i
            target = torch.full((m_i,), i, dtype=torch.long, device=self.device)
            term = float(lam[i]) * F.cross_entropy(Z_i, target, reduction="mean")
            loss = term if loss is None else loss + term
        grads = torch.autograd.grad(loss, list(net.parameters()))
        return _flatten_grads(net, grads)

    def grad_pair(self, theta_y, lam, batch):
        theta_y = np.ascontiguousarray(np.asarray(theta_y, dtype=float))
        _load_theta_into_net(self.net_y, theta_y)
        g_y = self._scalarized_grad(self.net_y, lam, batch)
        g_a = self._scalarized_grad(self.net_a, lam, batch)
        self.ifo_count += 2 * int(sum(len(b) for b in batch))
        if self.mu != 0.0:
            if self._theta_a is None:
                raise RuntimeError("set_anchor must be called before grad_pair.")
            g_y = g_y + self.mu * theta_y
            g_a = g_a + self.mu * self._theta_a
        return g_y, g_a


# =====================================================================
#  Factory
# =====================================================================
def make_mnist_patch_k10(per_class: int = BALANCED_MAX_PER_CLASS, *,
                         mu: float = 1e-4, batch_size: int = 1024,
                         sampler_seed: int = 41, init_seed: int = 8,
                         n_probes: int = 40, probe_seed: int = 7,
                         device: str = "auto", ah16_faithful: bool = False):
    """Returns (objectives, grad_objectives, L, joint_oracle, stoch, meta).

    Same contract as ``objectives_mnist_patch.make_mnist_patch``; ``L``
    already includes ``mu``; ``meta`` carries per_class, n, d, K, mu and
    the resolved device string.
    """
    dev = resolve_device(device)
    X_np, labels_np = load_mnist_k10(per_class, train=True)
    n = int(X_np.shape[0])
    X = torch.from_numpy(X_np).to(dev)
    Xk = [X[torch.from_numpy(np.nonzero(labels_np == k)[0]).long().to(dev)]
          for k in range(K)]                       # contiguous class slices
    targets = [torch.full((int(Xk[k].shape[0]),), k, dtype=torch.long, device=dev)
               for k in range(K)]
    net = PatchMLP(ah16_faithful).to(dev)
    params = list(net.parameters())
    d = int(sum(p.numel() for p in params))
    mu = float(mu)

    def joint_oracle_base(theta: np.ndarray):
        """Unpenalised per-class losses and Jacobian, per-class forward."""
        theta = np.ascontiguousarray(np.asarray(theta, dtype=float))
        _load_theta_into_net(net, theta)
        fvals = np.empty(K)
        J = np.empty((K, d))
        for k in range(K):
            loss = F.cross_entropy(net(Xk[k]), targets[k], reduction="mean")
            grads = torch.autograd.grad(loss, params)
            fvals[k] = float(loss.detach())
            J[k] = _flatten_grads(net, grads)
        return fvals, J

    def joint_oracle(theta: np.ndarray):
        theta = np.ascontiguousarray(np.asarray(theta, dtype=float))
        fvals, J = joint_oracle_base(theta)
        if mu != 0.0:
            fvals = fvals + 0.5 * mu * float(theta @ theta)
            J = J + mu * theta[None, :]
        return fvals, J

    def _obj(k):
        return lambda th: float(joint_oracle(th)[0][k])

    def _grad(k):
        return lambda th: joint_oracle(th)[1][k]

    objectives = [_obj(k) for k in range(K)]
    grad_objectives = [_grad(k) for k in range(K)]

    # L_i by random parameter-pair probes (the base factory's recipe),
    # on the unpenalised problem; the ridge adds exactly mu to every L_i.
    rng = np.random.RandomState(probe_seed)
    L = np.zeros(K)
    for _ in range(n_probes):
        t1 = make_patch_initial_point(rng.randint(1 << 30), ah16_faithful) \
            + 0.5 * rng.randn(d) * 0.1
        t2 = t1 + 0.5 * rng.randn(d)
        _, J1 = joint_oracle_base(t1)
        _, J2 = joint_oracle_base(t2)
        denom = float(np.linalg.norm(t2 - t1))
        L = np.maximum(L, np.linalg.norm(J2 - J1, axis=1) / denom)
    L = L + mu

    stoch = PatchStochLamOracleK10(X_np, labels_np, batch_size=batch_size,
                                   seed=sampler_seed, mu=mu, device=dev,
                                   ah16_faithful=ah16_faithful)
    meta = {"K": K, "n": n, "d": d, "per_class": int(per_class), "mu": mu,
            "device": str(dev), "device_description": device_description(dev),
            "ah16_faithful": ah16_faithful, "n_probes": int(n_probes),
            "class_rows": [int(Xk[k].shape[0]) for k in range(K)]}
    return objectives, grad_objectives, L, joint_oracle, stoch, meta


# =====================================================================
#  Test-side evaluation of a saved theta stack (for later use)
# =====================================================================
def evaluate_test_stack(thetas: Sequence[np.ndarray], device: str = "auto",
                        ah16_faithful: bool = False):
    """Per-class mean CE and error rate of every theta on the official
    test rows (raw CE, no ridge).  Returns (ce (m, K), err (m, K))."""
    dev = resolve_device(device)
    X_np, y_np = load_mnist_k10(train=False)
    X = torch.from_numpy(X_np).to(dev)
    rows = [torch.from_numpy(np.nonzero(y_np == k)[0]).long().to(dev) for k in range(K)]
    net = PatchMLP(ah16_faithful).to(dev)
    ce = np.empty((len(thetas), K))
    err = np.empty((len(thetas), K))
    with torch.no_grad():
        for j, th in enumerate(thetas):
            _load_theta_into_net(net, np.ascontiguousarray(np.asarray(th, dtype=float)))
            Z = net(X)
            for k in range(K):
                Zk = Z[rows[k]]
                tgt = torch.full((int(Zk.shape[0]),), k, dtype=torch.long, device=dev)
                ce[j, k] = float(F.cross_entropy(Zk, tgt, reduction="mean"))
                err[j, k] = float((Zk.argmax(dim=1) != tgt).double().mean())
    return ce, err


__all__ = ["K", "BALANCED_MAX_PER_CLASS", "resolve_device", "device_description",
           "load_mnist_k10", "PatchStochLamOracleK10", "make_mnist_patch_k10",
           "make_patch_initial_point", "evaluate_test_stack"]
