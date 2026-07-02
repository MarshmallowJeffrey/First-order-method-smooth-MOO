"""Non-convex multi-layer MLP multi-objective testbed (PyTorch backend).

Interface (matches ``make_mlp_nonconvex`` in the NumPy one):
    objectives      : list of K callables  F_i(theta) -> float
    grad_objectives : list of K callables  grad F_i(theta) -> ndarray (d,)
    L               : estimated smoothness constants, shape (K,)
    joint_oracle    : theta -> (fv (K,), gv (K, d)) in a single pass;
                      joint_oracle.fused is the same (kept for API parity).

theta is a NumPy array throughout (the optimisation code — bundle.py,
algorithm.py, baseline.py — is NumPy). All neural-network math is
performed in torch with autograd; we convert at the boundary.

Key design choices
------------------
- ``hidden_sizes`` is a list, so a 1-layer MLP (default ``[h]``) and a
  deep MLP use the same code path.
- We store one persistent ``nn.Module`` (``_net``) and, at every call,
  copy the flat ``theta`` into its parameters.
- Gradients come from ``torch.autograd.grad``, not from a manual backprop
  formula. This is the whole point of switching to torch: multi-layer
  backprop is written for us.
- ``torch.float64`` matches NumPy's default precision, so sanity-check
  agreement with the reference NumPy implementation is possible down to
  ~1e-12 rather than the ~1e-6 you'd get in float32.
"""
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(max(1, torch.get_num_threads()))


# ====================================================================
# Data generation
# ====================================================================
def _sample_planted_data(
    K: int, p: int, n: int, rng: np.random.RandomState, w_true_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (X, y, W_true) from the linear-softmax planted model.

    W_true ~ Uniform[-w_true_scale, w_true_scale]^{K x p}
    X      ~ N(0, I_p)^n
    y_j    ~ Categorical(softmax(W_true @ x_j))
    """
    W_true = rng.randn(K, p)                                  # match reference (not uniform)
    X = rng.randn(n, p)
    logits = X @ W_true.T                                     # (n, K)
    logits -= logits.max(axis=1, keepdims=True)               # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)
    u = rng.uniform(size=(n, 1))                              # match reference draw form
    labels = (u < cdf).argmax(axis=1)
    return X, labels, W_true


# ====================================================================
# Network builder
# ====================================================================
def _build_mlp(p: int, hidden_sizes: List[int], K: int) -> nn.Module:
    """Build p -> h1 -> ... -> hL -> K with ReLU between hidden layers.

    Matches the original when ``hidden_sizes=[h]`` (one hidden layer,
    ReLU, then linear-to-K).
    """
    layers: List[nn.Module] = []
    prev = p
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, K))                          # final logits layer
    return nn.Sequential(*layers)


def _param_shapes_and_total(net: nn.Module) -> Tuple[List[Tuple[str, torch.Size]], int]:
    """Return the ordered list of (name, shape) for net.parameters() and total count.

    Order matches the order ``net.parameters()`` iterates in, which is the
    order we'll use for flatten / unflatten. Not user-facing, but it's the
    contract by which theta_flat maps to network parameters.
    """
    shapes = [(name, p.shape) for name, p in net.named_parameters()]
    d = sum(p.numel() for p in net.parameters())
    return shapes, d


# ====================================================================
# Flat theta  <->  network parameters
# ====================================================================
def _load_theta_into_net(net: nn.Module, theta: np.ndarray) -> None:
    """Copy the flat NumPy vector theta into net.parameters() in place.

    Uses ``.data.copy_`` so we don't build any autograd history on this copy.
    After this call, each parameter tensor has ``requires_grad=True`` and
    is ready to be differentiated through in a forward pass.
    """
    t = torch.from_numpy(theta)                                # zero-copy view of NumPy buffer
    offset = 0
    for param in net.parameters():
        n_params = param.numel()
        chunk = t[offset: offset + n_params].view_as(param)
        param.data.copy_(chunk)
        offset += n_params
    assert offset == t.numel(), (
        f"theta length {t.numel()} does not match network parameter count {offset}"
    )


def _flatten_grads(net: nn.Module, grads: Tuple[torch.Tensor, ...]) -> np.ndarray:
    """Flatten a tuple of per-parameter gradient tensors into one NumPy vector.

    Iteration order matches ``net.parameters()`` — same as the flatten
    direction — so ``theta`` and ``grad`` share the same layout convention.
    """
    return torch.cat([g.reshape(-1) for g in grads]).detach().cpu().numpy()


# ====================================================================
# Main factory
# ====================================================================
def make_mlp_nonconvex(
    K: int = 3,
    p: int = 4,
    n: int = 60,
    h: int = None, # backwards-compat shorthand; same as hidden_sizes=[h]
    hidden_sizes = None,               # list[int]; default [8] = original 1-layer setting
    seed: int = 7,
    w_true_scale: float = 1.0,
) -> Tuple[List[Callable], List[Callable], np.ndarray, Callable]:
    """Create K per-class cross-entropy objectives for a multi-layer MLP.

    Architecture
    ------------
    Input x_j in R^p
      -> hidden layers with ReLU, widths given by ``hidden_sizes``
      -> output layer in R^K (logits z_j)
      -> softmax probabilities

    Per-class loss (the i-th MOO objective):

        F_i(theta) = (1 / n_i) * sum_{j: y_j = i} { -z_j[i] + logsumexp(z_j) }

    Parameters
    ----------
    K            : number of classes.
    p            : feature dimension.
    n            : total number of training samples.
    hidden_sizes : list of hidden-layer widths. Default [8] reproduces the
                   original 1-hidden-layer MLP. Use e.g. [16, 16] for two
                   hidden layers of width 16.
    seed         : random seed (data + weight-init for L estimation).
    w_true_scale : ground-truth planted-model weight range.

    Returns
    -------
    objectives, grad_objectives, L, joint_oracle
        Same contract as the original NumPy implementation.
    """
    if hidden_sizes is None:
        hidden_sizes = [h] if h is not None else [8]
    elif h is not None:
        raise ValueError("Pass either `h` or `hidden_sizes`, not both.")

    rng = np.random.RandomState(seed)

    # ---- planted data ----
    X_np, labels_np, _W_true = _sample_planted_data(
        K=K, p=p, n=n, rng=rng, w_true_scale=w_true_scale,
    )
    class_idx_np = [np.where(labels_np == i)[0] for i in range(K)]
    n_i = np.array([max(len(idx), 1) for idx in class_idx_np], dtype=np.float64)

    # ---- persistent torch objects (built once, reused every call) ----
    net = _build_mlp(p, hidden_sizes, K)
    shapes, d = _param_shapes_and_total(net)

    # Pre-move data to torch. class_idx as long tensors for indexing.
    X = torch.from_numpy(X_np)                                       # (n, p)
    class_idx = [torch.from_numpy(idx).long() for idx in class_idx_np]

    # =================================================================
    # Per-class F_i and grad_F_i  (each does one forward on class-i rows)
    # =================================================================
    def _F_i(theta: np.ndarray, i: int) -> float:
        """Evaluate F_i(theta) without building autograd graph."""
        _load_theta_into_net(net, theta)
        with torch.no_grad():
            X_i = X[class_idx[i]]                                    # (n_i, p)
            Z_i = net(X_i)                                            # (n_i, K)
            # F_i = (1/n_i) sum_j [-z_j[i] + logsumexp(z_j)]
            #     = mean cross-entropy of a target class fixed at i
            target = torch.full((X_i.shape[0],), i, dtype=torch.long)
            loss = F.cross_entropy(Z_i, target, reduction="mean")
        return float(loss.item())

    def _grad_F_i(theta: np.ndarray, i: int) -> np.ndarray:
        """Evaluate grad F_i(theta) via autograd."""
        _load_theta_into_net(net, theta)
        # Zero any lingering .grad (defensive; we use autograd.grad which
        # doesn't populate .grad, but this keeps the module clean).
        for param in net.parameters():
            param.grad = None
        X_i = X[class_idx[i]]
        Z_i = net(X_i)
        target = torch.full((X_i.shape[0],), i, dtype=torch.long)
        loss = F.cross_entropy(Z_i, target, reduction="mean")
        grads = torch.autograd.grad(loss, list(net.parameters()))
        return _flatten_grads(net, grads)

    def _F_and_grad_F_i(theta: np.ndarray, i: int) -> Tuple[float, np.ndarray]:
        """Return (F_i(theta), grad F_i(theta)) from a single forward pass."""
        _load_theta_into_net(net, theta)
        for param in net.parameters():
            param.grad = None
        X_i = X[class_idx[i]]
        Z_i = net(X_i)
        target = torch.full((X_i.shape[0],), i, dtype=torch.long)
        loss = F.cross_entropy(Z_i, target, reduction="mean")
        grads = torch.autograd.grad(loss, list(net.parameters()))
        return float(loss.item()), _flatten_grads(net, grads)

    # =================================================================
    # Joint oracle: (fv (K,), gv (K, d)) in one shared forward pass
    # =================================================================
    def _joint_oracle(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute all K losses and gradients from one forward pass.

        We forward on the FULL training set X once, then compute each F_i
        by masking to class-i rows. Each F_i still needs its own backward
        (autograd.grad with respect to different scalars requires separate
        backward passes) — but the forward is shared, which is where most
        of the cost sits for small MLPs. This mirrors the original one's
        ``_joint_oracle_fused``.
        """
        _load_theta_into_net(net, theta)
        for param in net.parameters():
            param.grad = None
        # Forward once on the whole batch. retain_graph=True on each grad
        # call so we can backward K times without rebuilding.
        Z_all = net(X)                                                # (n, K)
        fv = np.zeros(K)
        gv = np.zeros((K, d))
        for i in range(K):
            idx_i = class_idx[i]
            Z_i = Z_all[idx_i]
            target = torch.full((Z_i.shape[0],), i, dtype=torch.long)
            loss = F.cross_entropy(Z_i, target, reduction="mean")
            grads = torch.autograd.grad(
                loss, list(net.parameters()), retain_graph=(i < K - 1),
            )
            fv[i] = float(loss.item())
            gv[i] = _flatten_grads(net, grads)
        return fv, gv

    objectives = [lambda theta, i=i: _F_i(theta, i) for i in range(K)]
    grad_objectives = [lambda theta, i=i: _grad_F_i(theta, i) for i in range(K)]
    joint_oracle = _joint_oracle
    joint_oracle.fused = _joint_oracle                                # API parity

    # =================================================================
    # Estimate smoothness constants L_i (same probe-based scheme)
    # =================================================================
    n_probes = 40
    L_arr = np.zeros(K)
    for i in range(K):
        max_ratio = 0.0
        for _ in range(n_probes):
            t1 = rng.randn(d) * 0.5
            t2 = t1 + rng.randn(d) * 0.1
            g1 = grad_objectives[i](t1)
            g2 = grad_objectives[i](t2)
            diff_g = np.linalg.norm(g1 - g2)
            diff_t = np.linalg.norm(t1 - t2)
            if diff_t > 1e-12:
                max_ratio = max(max_ratio, diff_g / diff_t)
        L_arr[i] = max_ratio * 2.0                                     # safety factor of 2

    return objectives, grad_objectives, L_arr, joint_oracle