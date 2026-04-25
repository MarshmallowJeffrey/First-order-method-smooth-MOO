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

Data generation (planted-model scheme)
--------------------------------------
To avoid the "pure randomization" pitfall of i.i.d. uniform labelling
(where there is no statistical relationship between X and y), we sample
labels from a *ground-truth* softmax model:

  1. Sample a ground-truth weight matrix  W_true ∈ R^{K×p}  with entries
     drawn i.i.d. from Uniform[-1, 1].  W_true is hidden from the
     learner — it is only used to generate labels.

  2. Sample the feature matrix  X ∈ R^{n×p}  with entries from N(0, 1).

  3. For each sample j, sample its label from the categorical
     distribution that the paper itself specifies:

         P(Y = i | X = x_j)
            = exp(⟨w_true^i, x_j⟩) / Σ_l exp(⟨w_true^l, x_j⟩),

     i.e.   y_j  ~  Categorical( softmax(X W_true^T)_j ).

This gives the per-class objectives F_i a genuine statistical signal:
the Pareto front now has non-trivial curvature (there are real
trade-offs between misclassifying class 1 vs. class 2, etc.), and the
learned W should approximately recover (a scaled version of) W_true.

Model learned by the optimiser
-------------------------------
The conditional probability of class i given feature vector x under the
learned weights W is

    P(Y = i | X = x; W)  =  exp(⟨w^i, x⟩) / Σ_{l=1}^{K} exp(⟨w^l, x⟩).

Per-class loss (the i-th objective in the MOO problem):

    F_i(W)  =  (1/n_i) Σ_{j: y_j=i} { −log P(Y = i | x_j ; W) }
            =  (1/n_i) Σ_{j: y_j=i} { −⟨w^i, x_j⟩ + log Σ_l exp(⟨w^l, x_j⟩) }.

Two objective families
----------------------
1. **Regularised multi-class logistic regression** (strongly convex):
       F_i(W) = (1/n_i) Σ_{j: y_j=i} {−⟨w^i,x_j⟩ + log Σ_l exp(⟨w^l,x_j⟩)}
                  + (reg/2) ‖W‖²
   The ℓ₂ term makes each F_i  reg-strongly convex  (µ_i = reg).
   The planted model is well-specified for this hypothesis class.

2. **Generic non-convex** – Single-hidden-layer MLP with ReLU and
   softmax cross-entropy.  Fit to the same planted-model (X, y) from
   above.  The MLP is an over-parameterised non-convex hypothesis class
   that contains the true linear-softmax model as a special case.

Illustrative example
--------------------
With K = 3 classes and p = 4 features, the decision variable is
W ∈ R^{12}.  Because labels are sampled from softmax(X W_true^T) rather
than assigned uniformly, class counts are approximately (but not
exactly) balanced and P(Y=i | x_j) varies meaningfully across j.

At W = 0 (all learned weights zero) the learned softmax is uniform,
so F_i(0) = log(K) ≈ 1.099 for every class i.  As optimisation
proceeds, F_i(W) strictly decreases because there is real signal in
the data to exploit — unlike the pure-noise labelling case, where
the regulariser term dominates and the optimiser essentially stays
near zero.
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
# Planted-model data generation
# ====================================================================
def _sample_planted_data(
    K: int,
    p: int,
    n: int,
    rng: np.random.RandomState,
    w_true_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (X, y) from a linear-softmax planted model.

    Procedure
    ---------
    1. W_true ~ Uniform[-w_true_scale, w_true_scale]^{K × p}
    2. X      ~ N(0, 1)^{n × p}
    3. For each j:  y_j ~ Categorical( softmax(X W_true^T)_j )

    Parameters
    ----------
    K, p, n      : number of classes, features, samples.
    rng          : numpy RandomState (for reproducibility).
    w_true_scale : range of the ground-truth weights is [-s, s].
                   Larger s  ⇒  more separable classes (labels less noisy).

    Returns
    -------
    X      : shape (n, p)   feature matrix.
    labels : shape (n,)     integer labels in [0, K).
    W_true : shape (K, p)   ground-truth weights (returned for diagnostics;
                            the learner should never see these).

    Notes on class balance
    ----------------------
    With X ~ N(0, 1) and W_true ~ U[-1, 1], the true logits have
    std ≈ sqrt(p/3)  per coordinate, so the softmax outputs are moderately
    peaked.  Class counts will be close to, but not exactly, n/K.  Empty
    classes are unlikely for balanced K but are handled by the callers
    via an n_i ≥ 1 guard.
    """
    W_true = rng.uniform(-w_true_scale, w_true_scale, size=(K, p))
    X = rng.randn(n, p)
    true_logits = X @ W_true.T                   # (n, K)
    true_probs = _softmax(true_logits)           # (n, K)

    # Vectorised categorical sampling via inverse-CDF on cumulative probs.
    # For each row j, draw u_j ~ U(0, 1) and pick the smallest class
    # index i such that cumprob[j, i] ≥ u_j.
    cumprob = np.cumsum(true_probs, axis=1)      # (n, K)
    u = rng.uniform(size=(n, 1))                 # (n, 1)
    labels = (u < cumprob).argmax(axis=1)        # (n,)
    #print('labels:',labels)
    return X, labels, W_true


# ====================================================================
# 1.  Regularised multi-class logistic regression  (strongly convex)
# ====================================================================
def make_logreg_strongly_convex(
    K: int = 5,
    p: int = 4,
    n: int = 60,
    reg: float = 0.1,
    seed: int = 42,
    w_true_scale: float = 1.0,
) -> Tuple[List[Callable], List[Callable], np.ndarray, np.ndarray]:
    """Create K per-class logistic regression objectives with ℓ₂ regulariser.

    The i-th objective is:

        F_i(W) = (1/n_i) Σ_{j: y_j=i} { −⟨w^i, x_j⟩ + log Σ_l exp(⟨w^l, x_j⟩) }
                 + (reg/2) ‖W‖²

    where  W = [w^1, …, w^K] ∈ R^{Kp}  and  n_i = |{j : y_j = i}|.

    The ℓ₂ term makes every F_i  reg-strongly convex  (µ_i = reg).

    Data generation (planted model)
    -------------------------------
    - W_true ~ Uniform[-w_true_scale, w_true_scale]^{K × p}
    - X      ~ N(0, 1)^{n × p}
    - y_j    ~ Categorical( softmax(X W_true^T)_j )

    The learner never sees W_true; it only sees (X, y).  This replaces
    the previous i.i.d. uniform labelling, which produced no signal
    between X and y (so the unregularised loss was essentially flat
    and the optimum was dominated by the regulariser near 0).

    Smoothness:  Each F_i is L_i-smooth with  L_i ≤ ‖X‖²/(4 n_i) + reg,
    where X ∈ R^{n×p} is the data matrix.  This follows from the fact
    that the Hessian of the softmax cross-entropy loss w.r.t. W has
    spectral norm bounded by  ‖x_j‖² / 4  per sample.

    Parameters
    ----------
    K            : number of classes.
    p            : feature dimension (number of features per sample).
    n            : total number of training samples.
    reg          : ℓ₂ regularisation strength  (= µ_i for all i).
    seed         : random seed.
    w_true_scale : range of the ground-truth weights (default 1.0,
                   so W_true ~ Uniform[-1, 1]).  Larger values make
                   labels less noisy / classes more separable.

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

    # ---- planted-model data generation ----
    X, labels, _W_true = _sample_planted_data(
        K=K, p=p, n=n, rng=rng, w_true_scale=w_true_scale,
    )

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


# 2.  Single-hidden-layer MLP  (interpolation + PL)
# ====================================================================
def make_mlp_interpolation_pl(
    K: int = 3,
    p: int = 4,
    h: int = None,
    n_per_class: int = 20,
    seed: int = 11,
) -> Tuple[List[Callable], List[Callable], np.ndarray, np.ndarray]:
    """Create K per-class squared-loss objectives for a 1-hidden-layer MLP
    in a setting where both the interpolation (Asn 5.1) and the PL
    condition (Asn 5.2) are satisfied **globally** with µ_i = 1.

    Architecture (simplest choice satisfying both assumptions)
    ----------------------------------------------------------
    Input  x_j ∈ R^p
      →  hidden layer:   a_j = σ(W_1 x_j + b_1) ∈ R^h    (σ = identity)
      →  output layer:   z_j = W_2 a_j + b_2 ∈ R^K
      →  squared loss:   L_j = ½ ‖z_j − t_{y_j}‖²        (t_k = e_k ∈ R^K)

    Parameters  θ = (W_1, b_1, W_2, b_2),  d = h·p + h + K·h + K.

    Per-class objective:

        F_i(θ) = 1/(2 n_i) Σ_{j: y_j=i} ‖z(x_j; θ) − e_i‖² ≥ 0.

    Why σ = identity
    ----------------
    A nonlinear activation (ReLU, tanh, …) introduces kinks / saturation
    regions that break PL globally even in the interpolation regime.
    With σ = identity the loss is a 4th-degree polynomial in θ and
    admits a one-line global PL proof (see below).

    Data (K orthonormal class prototypes)
    -------------------------------------
    1. Draw a random orthogonal Q ∈ R^{p×p}, let μ_k = Q[:, k]  (k=1..K).
       Then  ‖μ_k‖ = 1  and  μ_k^T μ_l = δ_{kl}.
    2. Each class has n_per_class identical copies of its prototype:
          X_j = μ_{y_j},    y_j ∈ [K].

    Closed-form interpolator  θ*
    ----------------------------
    Set  W_1* = M := [μ_1 | ... | μ_K]^T  ∈ R^{K×p}
         b_1* = 0 ∈ R^K,
         W_2* = I_K ∈ R^{K×K},
         b_2* = 0 ∈ R^K.
    Then  W_1* μ_k = e_k  (rows of M are the μ_k's, and M M^T = I_K), so
         z(μ_k; θ*) = I_K · e_k + 0 = e_k = t_k.
    Hence every F_i(θ*) = 0 simultaneously  ⇒  (Asn 5.1) holds.

    Global PL condition  (µ_i = 1, proved in closed form)
    -----------------------------------------------------
    Let r_i(θ) := z(μ_i; θ) − e_i ∈ R^K, so F_i = ½ ‖r_i‖².  Writing the
    four partial derivatives directly (σ' ≡ 1):

        ∂F_i/∂W_1 = W_2^T r_i μ_i^T,      ∂F_i/∂b_1 = W_2^T r_i,
        ∂F_i/∂W_2 = r_i (W_1 μ_i + b_1)^T, ∂F_i/∂b_2 = r_i.

    Taking Frobenius norms and using  ‖μ_i‖² = 1:

        ‖∇F_i‖² = 2‖W_2^T r_i‖² + ‖r_i‖²(‖W_1 μ_i + b_1‖² + 1)
                ≥ ‖r_i‖² = 2 F_i.

    Thus  F_i(θ) ≤ ½ ‖∇F_i(θ)‖²  for every θ ∈ R^d, which is Asn 5.2
    with µ_i = 1.  No neighbourhood / sublevel-set restriction.

    Smoothness
    ----------
    F_i is a 4th-degree polynomial in θ, so it is *not* globally L-smooth.
    We report a numerical estimate of L_i obtained by probing gradient
    Lipschitz ratios on a ball around 0 (the natural init in experiments.py).
    The bundle algorithm only needs L_i to be valid along its iterates.

    Parameters
    ----------
    K            : number of classes (= output dim = hidden width).
    p            : input dimension,  must satisfy p ≥ K.
    h            : hidden width; must equal K  (default: K).
    n_per_class  : number of identical copies of each prototype.
    seed         : RNG seed for the orthonormal basis and L-probing.

    Returns
    -------
    objectives      : list of K callables F_i(θ) → float.
    grad_objectives : list of K callables ∇F_i(θ) → ndarray (d,).
    L               : smoothness constants, shape (K,)  (numerical).
    mu              : PL constants,       shape (K,)  (= 1, analytic).

    Illustrative example
    --------------------
    With K=3, p=4, h=3:  d = 3·4 + 3 + 3·3 + 3 = 27.
    F_i(θ*) = 0, F_i(0) = ½ ‖0 − e_i‖² = 0.5.
    PL check over 600 random (θ,i) pairs yields
    ‖∇F_i(θ)‖² − 2 F_i(θ) > 0 with large slack (see tests).

    >>> objs, grads, L, mu = make_mlp_interpolation_pl(K=3, p=4, n_per_class=20)
    >>> mu
    array([1., 1., 1.])
    """
    if h is None:
        h = K
    assert p >= K, "need p ≥ K for K orthonormal prototypes"
    assert h == K, "closed-form interpolator requires h = K"

    rng = np.random.RandomState(seed)
    d = h * p + h + K * h + K

    # ---- K orthonormal prototypes in R^p ----
    Q, _ = np.linalg.qr(rng.randn(p, p))
    M = Q[:, :K].T                                    # (K, p),  M M^T = I_K

    # ---- data: n_per_class copies of each prototype ----
    X = np.repeat(M, n_per_class, axis=0)             # (K·n_per_class, p)
    labels = np.repeat(np.arange(K), n_per_class)
    T = np.eye(K)                                     # targets, row k = e_k

    class_idx = [np.where(labels == i)[0] for i in range(K)]
    n_i = np.array([len(idx) for idx in class_idx], dtype=float)

    # ---- parameter packing / unpacking (same layout as make_mlp_nonconvex) ----
    def _unpack(theta: np.ndarray):
        idx = 0
        W1 = theta[idx: idx + h * p].reshape(h, p);  idx += h * p
        b1 = theta[idx: idx + h];                     idx += h
        W2 = theta[idx: idx + K * h].reshape(K, h);   idx += K * h
        b2 = theta[idx: idx + K];                      idx += K
        return W1, b1, W2, b2

    # ---- forward pass (identity activation) ----
    def _forward(theta: np.ndarray, X_batch: np.ndarray):
        W1, b1, W2, b2 = _unpack(theta)
        A = X_batch @ W1.T + b1                       # σ = id
        Z = A @ W2.T + b2
        return A, Z, W1, b1, W2, b2

    # ---- per-class squared-loss objective ----
    def _F_i(theta: np.ndarray, i: int) -> float:
        X_i = X[class_idx[i]]
        _, Z_i, *_ = _forward(theta, X_i)
        R = Z_i - T[i]
        return 0.5 * float((R ** 2).sum()) / n_i[i]

    # ---- per-class gradient via backprop (σ' = 1) ----
    def _grad_F_i(theta: np.ndarray, i: int) -> np.ndarray:
        X_i = X[class_idx[i]]
        ni  = n_i[i]
        A_i, Z_i, W1, b1, W2, b2 = _forward(theta, X_i)
        dZ  = (Z_i - T[i]) / ni                       # (n_i, K)
        dW2 = dZ.T @ A_i                              # (K, h)
        db2 = dZ.sum(axis=0)                          # (K,)
        dA  = dZ @ W2                                 # (n_i, h)
        dW1 = dA.T @ X_i                              # (h, p)
        db1 = dA.sum(axis=0)                          # (h,)
        return np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])

    objectives      = [lambda th, i=i: _F_i(th, i)      for i in range(K)]
    grad_objectives = [lambda th, i=i: _grad_F_i(th, i) for i in range(K)]

    # ---- numerical smoothness estimate (local, for the bundle algorithm) ----
    n_probes = 30
    L_arr = np.zeros(K)
    for i in range(K):
        best = 0.0
        for _ in range(n_probes):
            t1 = rng.randn(d)
            t2 = t1 + 0.05 * rng.randn(d)
            g1 = grad_objectives[i](t1)
            g2 = grad_objectives[i](t2)
            dt = np.linalg.norm(t1 - t2)
            if dt > 1e-12:
                best = max(best, np.linalg.norm(g1 - g2) / dt)
        L_arr[i] = 2.0 * best                         # safety factor of 2

    # PL constant:  µ_i = 1 globally (proved in docstring).
    mu_arr = np.ones(K)

    return objectives, grad_objectives, L_arr, mu_arr


# ====================================================================
# 3.  Single-hidden-layer neural network  (generic non-convex)
# ====================================================================
def make_mlp_nonconvex(
    K: int = 3,
    p: int = 4,
    n: int = 60,
    h: int = 8,
    seed: int = 7,
    w_true_scale: float = 1.0,
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

    Data generation (planted model)
    -------------------------------
    Uses the same linear-softmax planted model as the logistic-regression
    generator for consistency across experiments:

        W_true ~ Uniform[-w_true_scale, w_true_scale]^{K × p},
        X      ~ N(0, 1)^{n × p},
        y_j    ~ Categorical( softmax(X W_true^T)_j ).

    The MLP is an over-parameterised non-convex hypothesis class that
    can represent the true linear-softmax model exactly (take W_1 = I,
    b_1 sufficiently negative-shifted or W_1 such that pre-activations
    stay positive, then set W_2 to recover W_true).  This makes it a
    clean non-convex testbed: well-specified in theory, but optimisation
    must still contend with ReLU kinks and the bilinear W_2 W_1 product.

    Smoothness
    ----------
    No closed-form L_i is available for a neural network.  We estimate
    L_i by computing the gradient at several random points and measuring
    the maximum ratio  ‖∇F_i(θ₁) − ∇F_i(θ₂)‖ / ‖θ₁ − θ₂‖.

    Parameters
    ----------
    K            : number of classes.
    p            : feature dimension.
    n            : total number of training samples.
    h            : number of hidden units.
    seed         : random seed.
    w_true_scale : range of the ground-truth weights (default 1.0).

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

    # ---- planted-model data generation ----
    X, labels, _W_true = _sample_planted_data(
        K=K, p=p, n=n, rng=rng, w_true_scale=w_true_scale,
    )

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
