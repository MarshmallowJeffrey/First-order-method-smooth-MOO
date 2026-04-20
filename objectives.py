"""
objectives.py  –  Multi-class logistic regression objectives for MOO
=====================================================================

All objectives are variants of multi-class logistic regression, following
the notation below.

Notation
--------
- K classes, each with its own weight vector  w^i ∈ R^p,  i ∈ [K].
- n labelled training samples  {(y_1, x_1), …, (y_n, x_n)}  where
  y_j ∈ [K]  and  x_j = (x_{j1}, …, x_{jp}) ∈ R^p.
- n_i = |{j : y_j = i}|  is the number of samples in class i.
- The decision variable is  W = [w^1, …, w^K],  stored as a flat
  vector in R^{Kp}  (row-major: first p entries are w^1, etc.).

Model
-----
The conditional probability of class i given feature vector x is:

    P(Y = i | X = x)  =  exp(⟨w^i, x⟩) / Σ_{l=1}^{K} exp(⟨w^l, x⟩)

Per-class loss (the i-th objective in the MOO problem):

    F_i(W)  =  (1/n_i)  Σ_{j: y_j = i}  { −log  exp(⟨w^i, x_j⟩) / Σ_l exp(⟨w^l, x_j⟩) }
            =  (1/n_i)  Σ_{j: y_j = i}  { −⟨w^i, x_j⟩ + log Σ_l exp(⟨w^l, x_j⟩) }

Three objective families
------------------------
1. **Regularised multi-class logistic regression** (strongly convex):
       F_i(W)  =  (1/n_i) Σ_{j: y_j=i} {−⟨w^i,x_j⟩ + log Σ_l exp(⟨w^l,x_j⟩)}
                  + (reg/2) ‖W‖²
   The ℓ₂ term makes each F_i  reg-strongly convex  (µ_i = reg).

2. Single layer MLP

3. **Generic non-convex** (Single layer MLP).


Illustrative example
--------------------
With K = 3 classes and p = 4 features, the decision variable is
W ∈ R^{12}.  Suppose class counts are n_1 = 20, n_2 = 20, n_3 = 20.
At W = 0 (all weights zero), every sample has predicted probability
P(Y = i | x_j; W) = 1/3 for all i.  So

    F_i(0) = (1/n_i) Σ_{j: y_j=i} {−0 + log(3)} = log(3) ≈ 1.099

for each class i.
"""

from __future__ import annotations
import numpy as np
from typing import List, Callable, Tuple


# ====================================================================
# Softmax utilities
# ====================================================================
def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Parameters
    ----------
    logits : shape (n, K)  –  row j contains ⟨w^1, x_j⟩, …, ⟨w^K, x_j⟩.

    Returns
    -------
    probs  : shape (n, K),  probs[j, i] = P(Y=i | X=x_j; W).
    """
    shift = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shift)
    return e / e.sum(axis=-1, keepdims=True)


def _logsumexp(logits: np.ndarray) -> np.ndarray:
    """Numerically stable  log Σ_l exp(⟨w^l, x_j⟩)  for each sample j.

    Parameters
    ----------
    logits : shape (n, K)

    Returns
    -------
    out    : shape (n,),  out[j] = log Σ_{l=1}^K exp(⟨w^l, x_j⟩).
    """
    m = logits.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))).squeeze(-1)


# ====================================================================
# 1.  Regularised multi-class logistic regression  (strongly convex)
# ====================================================================
def make_logreg_strongly_convex(
    K: int = 5,
    p: int = 4,
    n: int = 60,
    reg: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Callable], List[Callable], np.ndarray, np.ndarray]:
    """Create K per-class logistic regression objectives with ℓ₂ regulariser.

    The i-th objective is:

        F_i(W) = (1/n_i) Σ_{j: y_j=i} { −⟨w^i, x_j⟩ + log Σ_l exp(⟨w^l, x_j⟩) }
                 + (reg/2) ‖W‖²

    where  W = [w^1, …, w^K] ∈ R^{Kp}  and  n_i = |{j : y_j = i}|.

    The ℓ₂ term makes every F_i  reg-strongly convex  (µ_i = reg).

    Smoothness:  Each F_i is L_i-smooth with  L_i ≤ ‖X‖²/(4 n_i) + reg,
    where X ∈ R^{n×p} is the data matrix.  This follows from the fact
    that the Hessian of the softmax cross-entropy loss w.r.t. W has
    spectral norm bounded by  ‖x_j‖² / 4  per sample.

    Parameters
    ----------
    K    : number of classes.
    p    : feature dimension (number of features per sample).
    n    : total number of training samples.
    reg  : ℓ₂ regularisation strength  (= µ_i for all i).
    seed : random seed.

    Returns
    -------
    objectives      : list of K callables  F_i(W) → float.
    grad_objectives : list of K callables  ∇F_i(W) → ndarray of shape (Kp,).
    L               : smoothness constants, shape (K,).
    mu              : strong-convexity constants, shape (K,).

    Illustrative example
    --------------------
    >>> objs, grads, L, mu = make_logreg_strongly_convex(K=3, p=4, n=60)
    >>> W = np.zeros(12)       # all weight vectors zero
    >>> objs[0](W)             # F_1(0) = log(3) ≈ 1.099  (uniform predictions)
    >>> grads[0](W).shape      # (12,)
    """
    rng = np.random.RandomState(seed)
    d = K * p

    # Generate data matrix  X ∈ R^{n×p}  and labels  y ∈ [K]^n
    X = rng.randn(n, p)
    labels = np.array([j % K for j in range(n)])    # balanced classes
    rng.shuffle(labels)

    # Class index sets and counts
    #   class_idx[i] = sorted array of sample indices j with y_j = i
    #   n_i = len(class_idx[i])
    class_idx = [np.where(labels == i)[0] for i in range(K)]
    n_i = np.array([len(idx) for idx in class_idx], dtype=float)
    n_i = np.maximum(n_i, 1.0)   # guard against empty classes

    # Smoothness and strong-convexity constants
    X_op_norm_sq = np.linalg.norm(X, ord=2) ** 2
    L_arr = np.array([X_op_norm_sq / (4.0 * n_i[i]) + reg for i in range(K)])
    mu_arr = np.full(K, reg)

    def _F_i(W_flat: np.ndarray, i: int) -> float:
        """Evaluate F_i(W).

        Steps:
          1. Reshape W to (K, p):  row l is w^l.
          2. Compute logits:  logits[j, l] = ⟨w^l, x_j⟩  for all j, l.
          3. For samples j with y_j = i, compute
               loss_j = −⟨w^i, x_j⟩ + log Σ_l exp(⟨w^l, x_j⟩)
          4. Average over class i:  (1/n_i) Σ_{j: y_j=i} loss_j.
          5. Add regulariser:  + (reg/2) ‖W‖².
        """
        W = W_flat.reshape(K, p)                     # (K, p)
        idx = class_idx[i]                           # indices j with y_j = i
        X_i = X[idx]                                 # (n_i, p)

        logits_i = X_i @ W.T                         # (n_i, K)
        lse_i = _logsumexp(logits_i)                 # (n_i,)
        class_logits_i = logits_i[:, i]              # (n_i,)  =  ⟨w^i, x_j⟩

        losses = -class_logits_i + lse_i             # (n_i,)
        avg_loss = losses.sum() / n_i[i]

        return float(avg_loss + 0.5 * reg * np.dot(W_flat, W_flat))

    def _grad_F_i(W_flat: np.ndarray, i: int) -> np.ndarray:
        """Compute ∇F_i(W).

        The gradient ∂F_i/∂w^l  for each class weight w^l is:

            l = i:  (1/n_i) Σ_{j: y_j=i} (P(Y=i|x_j;W) − 1) x_j  + reg · w^i
            l ≠ i:  (1/n_i) Σ_{j: y_j=i}  P(Y=l|x_j;W)       x_j  + reg · w^l

        Derivation:
          ∂/∂w^l [−⟨w^i, x_j⟩ + log Σ_l' exp(⟨w^{l'}, x_j⟩)]
            = −1[l=i] x_j  +  P(Y=l|x_j;W) x_j
            = (P(Y=l|x_j;W) − 1[l=i]) x_j

        Averaging over j: y_j = i  and adding the regulariser gives the above.
        """
        W = W_flat.reshape(K, p)                     # (K, p)
        idx = class_idx[i]
        X_i = X[idx]                                 # (n_i, p)

        logits_i = X_i @ W.T                         # (n_i, K)
        probs_i = _softmax(logits_i)                 # (n_i, K)

        # Gradient w.r.t. each w^l:
        #   (1/n_i) Σ_{j ∈ class i} P(Y=l|x_j;W) x_j   for all l
        # Then subtract  (1/n_i) Σ_{j ∈ class i} x_j  from the l=i block.
        grad_W = (probs_i.T @ X_i) / n_i[i]         # (K, p)
        grad_W[i] -= X_i.sum(axis=0) / n_i[i]       # subtract indicator for l=i

        return grad_W.ravel() + reg * W_flat

    objectives = [lambda W, i=i: _F_i(W, i) for i in range(K)]
    grad_objectives = [lambda W, i=i: _grad_F_i(W, i) for i in range(K)]

    return objectives, grad_objectives, L_arr, mu_arr


# ====================================================================
# 2.  Single Layer MLP  (interpolation + PL)
# ====================================================================



# ====================================================================
# 3.  Single-hidden-layer neural network  (generic non-convex)
# ====================================================================
def make_mlp_nonconvex(
    K: int = 3,
    p: int = 4,
    n: int = 60,
    h: int = 8,
    seed: int = 7,
) -> Tuple[List[Callable], List[Callable], np.ndarray]:
    """Create K per-class cross-entropy objectives for a 1-hidden-layer MLP.

    Architecture
    ------------
    Input  x_j ∈ R^p
      →  hidden layer:   a_j = σ(W_1 x_j + b_1) ∈ R^h      (σ = ReLU)
      →  output layer:   z_j = W_2 a_j + b_2 ∈ R^K
      →  softmax:        P(Y = i | x_j; θ) = exp(z_j^{(i)}) / Σ_l exp(z_j^{(l)})

    Parameters  θ = (W_1, b_1, W_2, b_2)  flattened into a vector of
    dimension  d = h·p + h + K·h + K.

    Per-class loss (the i-th MOO objective):

        F_i(θ) = (1/n_i) Σ_{j: y_j=i} { −z_j^{(i)} + log Σ_l exp(z_j^{(l)}) }

    This is non-convex due to the composition of the linear output layer
    with the ReLU hidden layer (the product W_2 · σ(W_1 x + b_1) is
    non-convex in (W_1, b_1, W_2) jointly).

    Smoothness
    ----------
    No closed-form L_i is available for a neural network.  We estimate
    L_i by computing the gradient at several random points and measuring
    the maximum ratio  ‖∇F_i(θ₁) − ∇F_i(θ₂)‖ / ‖θ₁ − θ₂‖.

    Parameters
    ----------
    K    : number of classes.
    p    : feature dimension.
    n    : total number of training samples.
    h    : number of hidden units.
    seed : random seed.

    Returns
    -------
    objectives      : list of K callables  F_i(θ) → float.
    grad_objectives : list of K callables  ∇F_i(θ) → ndarray of shape (d,).
    L               : estimated smoothness constants, shape (K,).
                      (no mu — the objectives are non-convex.)

    Illustrative example
    --------------------
    With K=3, p=4, h=8 the parameter vector θ has dimension
    d = 8·4 + 8 + 3·8 + 3 = 67.  The three objectives F_1, F_2, F_3
    measure the per-class cross-entropy through the neural network.
    At random initialisation, each F_i ≈ log(K) ≈ 1.099.

    >>> objs, grads, L = make_mlp_nonconvex(K=3, p=4, n=60, h=8)
    >>> d = 8*4 + 8 + 3*8 + 3  # = 67
    >>> theta = np.zeros(d)
    >>> objs[0](theta)           # F_1 at zero weights
    """
    rng = np.random.RandomState(seed)
    d = h * p + h + K * h + K     # total parameter count

    # Generate data
    X = rng.randn(n, p)
    labels = np.array([j % K for j in range(n)])
    rng.shuffle(labels)

    class_idx = [np.where(labels == i)[0] for i in range(K)]
    n_i = np.array([max(len(idx), 1) for idx in class_idx], dtype=float)

    # ---- parameter packing / unpacking ----
    def _unpack(theta: np.ndarray):
        """Unpack θ into (W_1, b_1, W_2, b_2).

        Layout in θ (contiguous blocks):
          W_1 : h × p  (rows of the input-to-hidden weight matrix)
          b_1 : h      (hidden biases)
          W_2 : K × h  (rows of the hidden-to-output weight matrix)
          b_2 : K      (output biases)
        """
        idx = 0
        W1 = theta[idx: idx + h * p].reshape(h, p);  idx += h * p
        b1 = theta[idx: idx + h];                     idx += h
        W2 = theta[idx: idx + K * h].reshape(K, h);   idx += K * h
        b2 = theta[idx: idx + K];                      idx += K
        return W1, b1, W2, b2

    # ---- forward pass ----
    def _forward(theta: np.ndarray, X_batch: np.ndarray):
        """Compute hidden activations and output logits.

        Returns
        -------
        A      : (n_batch, h)  hidden activations  σ(X_batch @ W_1^T + b_1)
        Z      : (n_batch, K)  output logits  A @ W_2^T + b_2
        pre_A  : (n_batch, h)  pre-activation  X_batch @ W_1^T + b_1
        W1, b1, W2, b2 : unpacked parameters
        """
        W1, b1, W2, b2 = _unpack(theta)
        pre_A = X_batch @ W1.T + b1              # (n_batch, h)
        A = np.maximum(pre_A, 0.0)                # ReLU
        Z = A @ W2.T + b2                         # (n_batch, K)
        return A, Z, pre_A, W1, b1, W2, b2

    # ---- per-class objective ----
    def _F_i(theta: np.ndarray, i: int) -> float:
        """Evaluate F_i(θ).

        F_i(θ) = (1/n_i) Σ_{j: y_j=i} { −z_j^{(i)} + log Σ_l exp(z_j^{(l)}) }

        Steps:
          1. Forward pass on class-i samples to get logits z_j ∈ R^K.
          2. For each such sample, compute  −z_j[i] + logsumexp(z_j).
          3. Average over n_i samples.
        """
        idx = class_idx[i]
        X_i = X[idx]                               # (n_i, p)
        _, Z_i, _, _, _, _, _ = _forward(theta, X_i)
        lse = _logsumexp(Z_i)                      # (n_i,)
        losses = -Z_i[:, i] + lse                  # (n_i,)
        return float(losses.sum() / n_i[i])

    # ---- per-class gradient via backpropagation ----
    def _grad_F_i(theta: np.ndarray, i: int) -> np.ndarray:
        """Compute ∇F_i(θ) via backpropagation.

        Backprop derivation for a single sample j with y_j = i:

          Forward:
            pre_a = W_1 x_j + b_1           ∈ R^h
            a      = σ(pre_a)               ∈ R^h   (ReLU)
            z      = W_2 a + b_2            ∈ R^K
            loss   = −z[i] + logsumexp(z)

          Output layer gradient:
            ∂loss/∂z = softmax(z) − e_i      ∈ R^K
            (where e_i is the i-th standard basis vector)

            ∂loss/∂W_2 = (∂loss/∂z) a^T     ∈ R^{K×h}
            ∂loss/∂b_2 = ∂loss/∂z            ∈ R^K

          Hidden layer gradient:
            δ_hidden = W_2^T (∂loss/∂z) ⊙ σ'(pre_a)   ∈ R^h
            (where σ'(pre_a) = 1[pre_a > 0] for ReLU)

            ∂loss/∂W_1 = δ_hidden x_j^T     ∈ R^{h×p}
            ∂loss/∂b_1 = δ_hidden            ∈ R^h

          Average over all samples j with y_j = i, dividing by n_i.
        """
        idx_i = class_idx[i]
        X_i = X[idx_i]                             # (n_i, p)
        ni = n_i[i]

        A_i, Z_i, pre_A_i, W1, b1, W2, b2 = _forward(theta, X_i)

        # Output-layer gradient:  ∂loss/∂z = softmax(z) − e_i
        probs_i = _softmax(Z_i)                    # (n_i, K)
        dZ = probs_i.copy()                        # (n_i, K)
        dZ[:, i] -= 1.0                            # subtract indicator

        # Gradients w.r.t. W_2, b_2
        dW2 = (dZ.T @ A_i) / ni                    # (K, h)
        db2 = dZ.sum(axis=0) / ni                  # (K,)

        # Backprop through ReLU to hidden layer
        dA = dZ @ W2                                # (n_i, h)
        relu_mask = (pre_A_i > 0).astype(float)    # (n_i, h)
        dH = dA * relu_mask                         # (n_i, h)

        # Gradients w.r.t. W_1, b_1
        dW1 = (dH.T @ X_i) / ni                    # (h, p)
        db1 = dH.sum(axis=0) / ni                  # (h,)

        # Pack into flat gradient vector (same layout as θ)
        grad = np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])
        return grad

    objectives = [lambda theta, i=i: _F_i(theta, i) for i in range(K)]
    grad_objectives = [lambda theta, i=i: _grad_F_i(theta, i) for i in range(K)]

    # ---- estimate smoothness constants L_i ----
    # Sample random parameter pairs and measure gradient Lipschitz ratio.
    n_probes = 20
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
        L_arr[i] = max_ratio * 2.0    # safety factor of 2

    return objectives, grad_objectives, L_arr
