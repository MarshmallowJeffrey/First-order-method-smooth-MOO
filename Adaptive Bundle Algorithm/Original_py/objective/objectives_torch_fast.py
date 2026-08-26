"""objectives_torch_fast.py  –  Torch MLP testbed + stochastic λ-scalarized oracle.

NEW FILE (July 15, 2026).  Part of the ``_fast`` implementation set; the
original ``objectives_torch.py`` is untouched.

``make_mlp_nonconvex_fast`` returns the ORIGINAL full oracles verbatim
(it calls ``objectives_torch.make_mlp_nonconvex`` — same objectives,
gradients, L estimates, joint oracle, bit-for-bit) plus one new object:

``StochLamOracle`` — the minibatch oracle the Momentum-SVRG inner loop
needs.  Design points:

* **Scalarized gradients only.**  An inner (M)SVRG step needs
  ∇f_{λ,S}(θ) = Σ_k λ_k ∇F_{k,S_k}(θ), not the K×d Jacobian: one forward
  + ONE backward per parameter point (the joint oracle pays K backwards).
* **Stratified sampling.**  F_k is the mean loss over class-k rows and the
  classes are disjoint, so the minibatch draws b_k ≈ b·n_k/n rows per
  class (each without replacement within the class) and averages within
  the class before λ-weighting — an unbiased estimator of ∇F_λ.
* **Anchor net pair.**  SVRG evaluates every batch at the current point y
  AND at the anchor a.  Two persistent ``nn.Module``s avoid re-loading the
  anchor parameters at every step: ``set_anchor`` loads them once per
  segment.
* **IFO accounting.**  Every call adds 2·b_total to ``ifo_count``
  (per-sample gradient units; a full joint call costs n on this problem
  because the classes partition the samples).  The caller converts to the
  legacy axis via grad_equiv = IFO · K / n.

Data identity: the constructor re-runs ``_sample_planted_data`` with a
fresh ``RandomState(seed)`` — the exact first rng consumer inside the
original factory — so X/labels match the original problem instance
bit-for-bit (asserted cheaply here, and again end-to-end by
``sanity_checks_fast.py``).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from objectives_torch import (
    make_mlp_nonconvex,
    _sample_planted_data,
    _build_mlp,
    _param_shapes_and_total,
    _load_theta_into_net,
    _flatten_grads,
)


class StochLamOracle:
    """Stratified minibatch oracle for λ-scalarized gradients.

    Parameters
    ----------
    X_np, labels_np : the planted dataset (must equal the instance the full
                      oracles were built on; verified by the factory).
    K, p, hidden_sizes, activation : network architecture (matches original).
    batch_size      : total minibatch size b (split across classes ∝ n_k).
    seed            : sampler seed (reproducible draws, independent of the
                      data/init seeds).
    """

    def __init__(self, X_np: np.ndarray, labels_np: np.ndarray, *,
                 K: int, p: int, hidden_sizes: Sequence[int],
                 activation: str, batch_size: int, seed: int) -> None:
        if batch_size < K:
            raise ValueError(
                f"batch_size must be >= K={K} (one row per class); "
                f"got {batch_size}."
            )
        self.K = int(K)
        self.n = int(X_np.shape[0])
        self.batch_size = int(batch_size)
        self.rng = np.random.RandomState(seed)

        self.X = torch.from_numpy(np.ascontiguousarray(X_np))
        self.class_idx_np: List[np.ndarray] = [
            np.where(labels_np == i)[0] for i in range(K)
        ]
        if any(idx.size == 0 for idx in self.class_idx_np):
            raise ValueError("Every class must contain at least one sample.")

        # Per-class draw sizes b_k ∝ n_k, at least 1 each.
        n_k = np.array([idx.size for idx in self.class_idx_np], dtype=float)
        raw = batch_size * n_k / n_k.sum()
        b_k = np.maximum(1, np.floor(raw).astype(int))
        # Distribute the remainder to the largest fractional parts.
        short = batch_size - int(b_k.sum())
        if short > 0:
            order = np.argsort(-(raw - np.floor(raw)))
            for j in order[:short]:
                b_k[j] += 1
        self.b_k = np.minimum(b_k, n_k.astype(int))
        self.b_total = int(self.b_k.sum())

        # Two persistent nets: current point and anchor.
        self.net_y = _build_mlp(p, list(hidden_sizes), K, activation=activation)
        self.net_a = _build_mlp(p, list(hidden_sizes), K, activation=activation)
        _, d_y = _param_shapes_and_total(self.net_y)
        self.d = d_y

        self.ifo_count = 0  # per-sample gradient units consumed so far

    # -- sampling ------------------------------------------------------
    def sample_batch(self) -> List[np.ndarray]:
        """Draw one stratified minibatch: per-class GLOBAL row indices."""
        batch = []
        for i in range(self.K):
            pool = self.class_idx_np[i]
            take = int(self.b_k[i])
            if take >= pool.size:
                batch.append(pool.copy())
            else:
                batch.append(self.rng.choice(pool, size=take, replace=False))
        return batch

    def full_batch(self) -> List[np.ndarray]:
        """The whole dataset as one 'batch' (degeneration tests: b = n)."""
        return [idx.copy() for idx in self.class_idx_np]

    # -- anchor management ---------------------------------------------
    def set_anchor(self, theta_a: np.ndarray) -> None:
        """Load the anchor parameters once (call at segment start)."""
        _load_theta_into_net(self.net_a, np.asarray(theta_a, dtype=float))

    # -- gradients -------------------------------------------------------
    def _scalarized_grad(self, net, lam: np.ndarray,
                         batch: List[np.ndarray]) -> np.ndarray:
        """∇_θ Σ_k λ_k · mean-CE(class-k rows of `batch`) for one net."""
        for param in net.parameters():
            param.grad = None
        rows = np.concatenate(batch)
        X_b = self.X[torch.from_numpy(rows).long()]
        Z = net(X_b)
        loss = None
        off = 0
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

    def grad_pair(self, theta_y: np.ndarray, lam: np.ndarray,
                  batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """(∇f_{λ,S}(y), ∇f_{λ,S}(anchor)) on the SAME batch S.

        The anchor net keeps whatever ``set_anchor`` loaded last; only the
        current point is re-loaded here.  Adds 2·(rows in S) to
        ``ifo_count``.
        """
        _load_theta_into_net(self.net_y, np.asarray(theta_y, dtype=float))
        g_y = self._scalarized_grad(self.net_y, lam, batch)
        g_a = self._scalarized_grad(self.net_a, lam, batch)
        self.ifo_count += 2 * int(sum(len(b) for b in batch))
        return g_y, g_a


def make_mlp_nonconvex_fast(
    K: int = 3,
    p: int = 4,
    n: int = 60,
    hidden_sizes: Optional[Sequence[int]] = None,
    seed: int = 7,
    w_true_scale: float = 1.0,
    n_probes: int = 40,
    activation: str = "relu",
    *,
    batch_size: int = 1024,
    sampler_seed: int = 0,
) -> Tuple[List[Callable], List[Callable], np.ndarray, Callable,
           StochLamOracle]:
    """Original full oracles (verbatim) + a ``StochLamOracle``.

    Returns ``objectives, grad_objectives, L, joint_oracle, stoch_oracle``.
    The first four come straight from ``objectives_torch.make_mlp_nonconvex``
    with identical arguments, so full-oracle behaviour (and the problem
    instance) is bit-for-bit the original.
    """
    if hidden_sizes is None:
        hidden_sizes = [8]
    objectives, grad_objectives, L_arr, joint_oracle = make_mlp_nonconvex(
        K=K, p=p, n=n, hidden_sizes=list(hidden_sizes), seed=seed,
        w_true_scale=w_true_scale, n_probes=n_probes, activation=activation,
    )

    # Reproduce the dataset: _sample_planted_data is the FIRST rng consumer
    # inside the original factory, so a fresh RandomState(seed) yields the
    # identical draw sequence and therefore identical X / labels.
    rng = np.random.RandomState(seed)
    X_np, labels_np, _W_true = _sample_planted_data(
        K=K, p=p, n=n, rng=rng, w_true_scale=w_true_scale,
    )

    cap = min(int(batch_size), int(n))
    stoch = StochLamOracle(
        X_np, labels_np, K=K, p=p, hidden_sizes=list(hidden_sizes),
        activation=activation, batch_size=cap, seed=sampler_seed,
    )
    return objectives, grad_objectives, L_arr, joint_oracle, stoch
