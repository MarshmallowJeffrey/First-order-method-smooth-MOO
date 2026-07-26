"""objectives_bandit_toy.py — SURF offline-bandit toy for the MSVRG pair.

NEW FILE (July 26, 2026).  Nothing in the original files is required to
change for this module; it plugs the SURF offline-bandit toy example
(SURF paper Appendix F.1 == our paper Section 5.2) into the existing
Momentum-SVRG pair:

* ``baseline_svrg_certified_without_256_checkpoints.baseline_svrg_certified``
* ``algorithm_fast_without_256_checkpoints.algorithm_adaptive_fast``

Problem (both papers' shared setting)
-------------------------------------
Single-state bandit, A = 5 arms, x_a = linspace(0, 1, A), true rewards
R1(a) = x_a and R2(a) = 1 - x_a^4, uniform reference policy pi_ref,
KL coefficient tau = 0.05 (SURF's beta).  A fixed BALANCED offline
dataset of T rows (a_t, r1_t, r2_t) with Gaussian reward noise
(std 0.5) yields per-arm sample means (R1_hat, R2_hat); every quantity
the solvers see is built from these plug-in estimates.

Parameterization (user-approved July 26): reduced logits
    theta in R^{A-1},   z = [theta, 0],   pi = softmax(z),
so the problem is unconstrained and smooth, with
    F_k(theta) = tau * KL(pi || pi_ref) - <pi, Rk_hat>,
    grad_theta F_k = [J (tau*z - Rk_hat)]_{1..A-1},   J = Diag(pi) - pi pi^T.
The reduced softmax is a diffeomorphism onto the simplex interior, so
the unique stationary point of every scalarization is its global
optimum and equals the closed-form softmax solution.

Finite-sum view for SVRG
------------------------
With N_a the arm counts,  <pi, Rk_hat> = (1/T) sum_t (T/N_{a_t}) *
pi(a_t) * r_{k,t}, so
    F_k = (1/T) sum_t f_{k,t},
    f_{k,t}(theta) = tau*KL(pi||pi_ref) - (T/N_{a_t}) * pi(a_t) * r_{k,t}.
The KL term appears EXACTLY in every row (zero variance); minibatch
variance comes only from the reward rows.  ``BanditStochOracle`` mirrors
the ``StochLamOracle`` surface used by both methods (attributes
``n / b_total / ifo_count / K``, methods ``sample_batch / full_batch /
set_anchor / grad_pair``).  The batch is a K-list of row-index arrays,
one INDEPENDENT draw per component from the same T-row pool (b_total/K
rows each), so the baseline's ``_support_batch`` zero-weight dropping
stays exact.  IFO accounting mirrors the MLP oracle verbatim:
``grad_pair`` adds 2 * (rows in batch), and the grad-equivalent
conversion joint*K + IFO*K/n applies with n = T.

Closed-form oracle (never timed)
--------------------------------
    pi_star(w)   = softmax(log pi_ref + (w*R1_hat + (1-w)*R2_hat)/tau)
    theta_star(w) equivalent reduced logits
    f_pf(w)      = (F1, F2) at pi_star(w)          (exact plug-in PF)
    F_w(theta_star(w))                              (scalarized optimum)
    v(w)         = SURF Eq. (9) plug-in traversal speed
    Phi_hat      = normalized arc-length CDF; arc-uniform weights (Rule 1)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "BanditToyProblem",
    "BanditStochOracle",
    "make_bandit_toy",
    "make_bandit_toy_K",
    "calibrate_L",
]


# =====================================================================
#  Core softmax helpers (float64, numerically guarded)
# =====================================================================
def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def _safe_log(p: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(p, 1e-300))


# =====================================================================
#  Problem container
# =====================================================================
class BanditToyProblem:
    """Offline-bandit toy instance built from one fixed balanced dataset.

    ``R_true`` = None (default) gives the papers' bi-objective pair
    R1 = x, R2 = 1 - x^alpha (K = 2, bit-identical to the July-26 runs).
    Passing an explicit (K, A) reward matrix generalises the instance to
    K objectives; the closed-form softmax oracle holds for every K, only
    the SURF arc-length layer (speed / CDF / Rule 1) is bi-objective and
    stays guarded behind K == 2.
    """

    def __init__(self, *, A: int, tau: float, alpha: float,
                 T: int, noise_std: float, data_seed: int,
                 R_true: Optional[np.ndarray] = None) -> None:
        if T % A != 0:
            raise ValueError(f"T={T} must be divisible by A={A} for the "
                             "balanced dataset.")
        self.A = int(A)
        self.d = int(A - 1)          # reduced logits
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.T = int(T)
        self.noise_std = float(noise_std)
        self.data_seed = int(data_seed)

        self.x_vals = np.linspace(0.0, 1.0, A)
        if R_true is None:
            self.R_true = np.stack([self.x_vals,
                                    1.0 - self.x_vals ** alpha])   # (2, A)
        else:
            self.R_true = np.asarray(R_true, dtype=float)
            if self.R_true.ndim != 2 or self.R_true.shape[1] != A:
                raise ValueError(f"R_true must have shape (K, {A}); got "
                                 f"{self.R_true.shape}.")
        self.K = int(self.R_true.shape[0])
        self.pi_ref = np.full(A, 1.0 / A)

        # ---- balanced offline dataset (SURF F.1 protocol) ----
        rng = np.random.RandomState(data_seed)
        per_arm = T // A
        self.actions = np.repeat(np.arange(A), per_arm)
        noise = rng.normal(0.0, noise_std, size=(self.K, T))
        self.rewards = self.R_true[:, self.actions] + noise     # (K, T)

        self.counts = np.bincount(self.actions, minlength=A).astype(float)
        R_hat = np.zeros((self.K, A))
        for k in range(self.K):
            R_hat[k] = np.bincount(self.actions, weights=self.rewards[k],
                                   minlength=A) / self.counts
        self.R_hat = R_hat

        # Per-row scale (T / N_{a_t}) used by the finite-sum view.
        self.row_scale = (self.T / self.counts)[self.actions]   # (T,)

        # Callables the bundle machinery consumes.
        self.objectives: List[Callable] = [
            (lambda th, k=k: self._F(th, k)) for k in range(self.K)
        ]
        self.grad_objectives: List[Callable] = [
            (lambda th, k=k: self._gradF(th, k)) for k in range(self.K)
        ]

    # ---------------- deterministic objective surface ----------------
    def _pi_z(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta, dtype=float)
        z = np.append(theta, 0.0)
        return _softmax(z), z

    def pi_of_theta(self, theta: np.ndarray) -> np.ndarray:
        return self._pi_z(theta)[0]

    def _kl(self, pi: np.ndarray) -> float:
        return float(pi @ (_safe_log(pi) - _safe_log(self.pi_ref)))

    def _F(self, theta: np.ndarray, k: int) -> float:
        pi, _ = self._pi_z(theta)
        return self.tau * self._kl(pi) - float(pi @ self.R_hat[k])

    def _gradF(self, theta: np.ndarray, k: int) -> np.ndarray:
        pi, z = self._pi_z(theta)
        g_full = pi * (self.tau * z - self.R_hat[k])
        g_full -= pi * float(pi @ (self.tau * z - self.R_hat[k]))
        return g_full[: self.d]

    def joint_oracle(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fused oracle: theta -> (fvals (K,), grads (K, d)); one softmax."""
        pi, z = self._pi_z(theta)
        kl = self._kl(pi)
        fvals = np.array([self.tau * kl - float(pi @ self.R_hat[k])
                          for k in range(self.K)])
        grads = np.empty((self.K, self.d))
        for k in range(self.K):
            u = self.tau * z - self.R_hat[k]
            g_full = pi * u - pi * float(pi @ u)
            grads[k] = g_full[: self.d]
        return fvals, grads

    # ---------------- closed-form (plug-in) oracle, any K -------------
    def pi_star_lam(self, lam: np.ndarray, use_true: bool = False) -> np.ndarray:
        R = self.R_true if use_true else self.R_hat
        lam = np.asarray(lam, dtype=float)
        logits = _safe_log(self.pi_ref) + (lam @ R) / self.tau
        return _softmax(logits)

    def theta_star_lam(self, lam: np.ndarray,
                       use_true: bool = False) -> np.ndarray:
        z = _safe_log(self.pi_star_lam(lam, use_true))
        return (z[: self.d] - z[self.d]).astype(float)

    def f_vec_lam(self, lam: np.ndarray) -> np.ndarray:
        """All K plug-in objective values at pi_star(lam)."""
        pi = self.pi_star_lam(lam)
        kl = self._kl(pi)
        return self.tau * kl - self.R_hat @ pi

    def scalarized_opt_lam(self, lam: np.ndarray) -> float:
        lam = np.asarray(lam, dtype=float)
        return float(lam @ self.f_vec_lam(lam))

    def oracle_batch(self, lams: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised closed-form oracle over a (n, K) stack of lambdas.

        Returns (fvecs (n, K), f_star_scal (n,)).
        """
        lams = np.asarray(lams, dtype=float)
        logits = _safe_log(self.pi_ref)[None, :] + (lams @ self.R_hat) / self.tau
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        kl = np.sum(P * (_safe_log(P) - _safe_log(self.pi_ref)[None, :]),
                    axis=1)
        fvecs = self.tau * kl[:, None] - P @ self.R_hat.T        # (n, K)
        f_star_scal = np.einsum("nk,nk->n", lams, fvecs)
        return fvecs, f_star_scal

    # ---------------- bi-objective (w-parameterised) layer ------------
    def _require_K2(self, what: str) -> None:
        if self.K != 2:
            raise ValueError(f"{what} is bi-objective (K=2) only; this "
                             f"instance has K={self.K}. Use the *_lam API.")

    def pi_star(self, w: float, use_true: bool = False) -> np.ndarray:
        self._require_K2("pi_star(w)")
        R = self.R_true if use_true else self.R_hat
        logits = _safe_log(self.pi_ref) + (w * R[0] + (1.0 - w) * R[1]) / self.tau
        return _softmax(logits)

    def theta_star(self, w: float, use_true: bool = False) -> np.ndarray:
        z = _safe_log(self.pi_star(w, use_true))
        return (z[: self.d] - z[self.d]).astype(float)

    def f_pf(self, w: float) -> np.ndarray:
        """Exact plug-in PF point (F1, F2) at weight w."""
        pi = self.pi_star(w)
        kl = self._kl(pi)
        return np.array([self.tau * kl - float(pi @ self.R_hat[k])
                         for k in range(2)])

    def scalarized_opt(self, w: float) -> float:
        f = self.f_pf(w)
        return float(w * f[0] + (1.0 - w) * f[1])

    def speed(self, w: float, use_true: bool = False) -> float:
        """SURF Eq. (9): tau^{-1} sqrt(w^2+(1-w)^2) * dR^T Cov(pi_w) dR."""
        R = self.R_true if use_true else self.R_hat
        pi = self.pi_star(w, use_true)
        dR = R[0] - R[1]
        quad = float(dR @ (pi * dR) - (pi @ dR) ** 2)
        return max((1.0 / self.tau) * np.sqrt(w * w + (1.0 - w) ** 2) * quad,
                   0.0)

    def arc_cdf(self, n_dense: int = 5000,
                use_true: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """(w_dense, Phi(w_dense)) with trapezoid integration of the speed."""
        w = np.linspace(0.0, 1.0, n_dense)
        v = np.array([self.speed(wi, use_true) for wi in w])
        s = np.concatenate([[0.0],
                            np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(w))])
        total = s[-1]
        if total <= 0.0:
            return w, w.copy()
        return w, s / total

    def arc_uniform_weights(self, n_target: int = 12,
                            n_dense: int = 5000) -> np.ndarray:
        """SURF Rule 1 plug-in: w_n = Phi_hat^{-1}(n/(n_target-1))."""
        w, cdf = self.arc_cdf(n_dense)
        q = np.linspace(0.0, 1.0, n_target)
        return np.interp(q, cdf, w)

    def cdf_sup_gap(self, n_dense: int = 5000) -> float:
        """|| Phi_hat - Phi ||_inf on the dense grid (statistical layer)."""
        w, cdf_hat = self.arc_cdf(n_dense, use_true=False)
        _, cdf_true = self.arc_cdf(n_dense, use_true=True)
        return float(np.max(np.abs(cdf_hat - cdf_true)))

    def reward_sup_gap(self) -> float:
        return float(np.max(np.abs(self.R_hat - self.R_true)))


def make_bandit_toy(*, T: int = 1000, noise_std: float = 0.5,
                    data_seed: int = 7, A: int = 5, tau: float = 0.05,
                    alpha: float = 4.0) -> BanditToyProblem:
    return BanditToyProblem(A=A, tau=tau, alpha=alpha, T=T,
                            noise_std=noise_std, data_seed=data_seed)


def make_bandit_toy_K(*, K: int = 5, T: int = 1000, noise_std: float = 0.5,
                      data_seed: int = 7, A: int = 5, tau: float = 0.05,
                      alpha: float = 4.0) -> BanditToyProblem:
    """K-objective bandit (user-approved Jul 26 design).

    Centered-quartic rewards on the arm feature grid:
        R_k(a) = 1 - |x_a - c_k|^alpha,   c_k = linspace(0, 1, K),
    so objective k pulls towards its own centre (diagonal 1 when the
    centres coincide with the arms, i.e. A = K); the K = 2, centres
    {0, 1} member of this family is the papers' R2 = 1 - x^alpha with a
    quartic (instead of linear) R1 — the original bi-objective pair
    itself stays available through ``make_bandit_toy``.
    """
    x = np.linspace(0.0, 1.0, A)
    c = np.linspace(0.0, 1.0, K)
    R_true = 1.0 - np.abs(x[None, :] - c[:, None]) ** alpha    # (K, A)
    return BanditToyProblem(A=A, tau=tau, alpha=alpha, T=T,
                            noise_std=noise_std, data_seed=data_seed,
                            R_true=R_true)


# =====================================================================
#  Stochastic oracle (StochLamOracle-compatible surface)
# =====================================================================
class BanditStochOracle:
    """Minibatch oracle over the offline dataset rows.

    Batch = K-list of row-index arrays, one independent draw per
    component (b_total/K rows each, without replacement).  Estimator per
    component k (skipping components whose list is empty, exactly like
    the MLP oracle, which keeps ``_support_batch`` exact):

        g_k(theta; S_k) = tau*(J z)_red
                          - J_red @ c_k,
        c_k(a) = mean over t in S_k of (T/N_{a_t}) * r_{k,t} * 1{a_t = a}

    and the scalarized estimator is sum_k lam_k * g_k.  Unbiased for
    grad F_lam; the exact-KL part carries zero variance.
    """

    def __init__(self, problem: BanditToyProblem, *, batch_size: int,
                 seed: int) -> None:
        if batch_size < problem.K:
            raise ValueError(f"batch_size must be >= K={problem.K}; "
                             f"got {batch_size}.")
        self.problem = problem
        self.K = problem.K
        self.n = problem.T
        self.batch_size = int(batch_size)
        b_k = batch_size // self.K
        self.b_k = np.full(self.K, b_k, dtype=int)
        self.b_k[: batch_size - b_k * self.K] += 1
        self.b_total = int(self.b_k.sum())
        self.rng = np.random.RandomState(seed)
        self.ifo_count = 0
        self._theta_a: Optional[np.ndarray] = None

    # -- batches ---------------------------------------------------------
    def sample_batch(self) -> List[np.ndarray]:
        return [self.rng.choice(self.n, size=int(b), replace=False)
                for b in self.b_k]

    def full_batch(self) -> List[np.ndarray]:
        return [np.arange(self.n) for _ in range(self.K)]

    # -- anchor ----------------------------------------------------------
    def set_anchor(self, theta_a: np.ndarray) -> None:
        self._theta_a = np.asarray(theta_a, dtype=float).copy()

    # -- gradients -------------------------------------------------------
    def _scalarized_grad(self, theta: np.ndarray, lam: np.ndarray,
                         batch: List[np.ndarray]) -> np.ndarray:
        p = self.problem
        pi, z = p._pi_z(theta)
        d = p.d
        g = np.zeros(d)
        kl_full = pi * (p.tau * z) - pi * float(pi @ (p.tau * z))
        for k in range(self.K):
            rows = batch[k]
            if len(rows) == 0:
                continue
            coef = np.bincount(
                p.actions[rows],
                weights=p.row_scale[rows] * p.rewards[k, rows],
                minlength=p.A,
            ) / float(len(rows))
            rew_full = pi * coef - pi * float(pi @ coef)
            g += float(lam[k]) * (kl_full[:d] - rew_full[:d])
        return g

    def grad_pair(self, theta_y: np.ndarray, lam: np.ndarray,
                  batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """(grad_S(y), grad_S(anchor)) on the SAME batch; IFO += 2*rows."""
        if self._theta_a is None:
            raise RuntimeError("set_anchor must be called before grad_pair.")
        g_y = self._scalarized_grad(theta_y, lam, batch)
        g_a = self._scalarized_grad(self._theta_a, lam, batch)
        self.ifo_count += 2 * int(sum(len(b) for b in batch))
        return g_y, g_a


# =====================================================================
#  Smoothness calibration (numerical, safety-factored)
# =====================================================================
def calibrate_L(problem: BanditToyProblem, *, n_w: int = 101,
                n_seg: int = 9, n_rand: int = 100, rand_scale: float = 2.0,
                safety: float = 1.5, seed: int = 0,
                fd_h: float = 1e-6) -> Dict:
    """Numerically bound L_k = safety * max ||Hess F_k|| over the region
    the iterates can roam: the segments {s * theta_star(w)} from the
    origin (theta0 = 0) to the solution path, plus random perturbations.

    Hessians are central finite differences of the ANALYTIC gradient
    (d = A-1 is tiny, so this is exact to ~1e-9 and costs nothing).
    """
    rng = np.random.RandomState(seed)
    thetas = [np.zeros(problem.d)]
    if problem.K == 2:
        # Original K=2 sample path (kept verbatim so the July-26 K=2
        # calibration — and therefore every K=2 trajectory — is
        # reproduced bit-for-bit).
        ws = np.linspace(0.0, 1.0, n_w)
        for w in ws:
            th_star = problem.theta_star(w)
            for s in np.linspace(0.0, 1.0, n_seg)[1:]:
                thetas.append(s * th_star)
        for _ in range(n_rand):
            w = rng.rand()
            thetas.append(problem.theta_star(w)
                          + rng.normal(0.0, rand_scale, size=problem.d))
    else:
        # K-general sample: vertices + centroid + Dirichlet draws, each
        # with the same origin-to-solution segments and perturbations.
        lam_list = [np.eye(problem.K)[k] for k in range(problem.K)]
        lam_list.append(np.full(problem.K, 1.0 / problem.K))
        lam_list.extend(rng.dirichlet(np.ones(problem.K), size=n_w))
        for lam in lam_list:
            th_star = problem.theta_star_lam(lam)
            for s in np.linspace(0.0, 1.0, n_seg)[1:]:
                thetas.append(s * th_star)
        for _ in range(n_rand):
            lam = rng.dirichlet(np.ones(problem.K))
            thetas.append(problem.theta_star_lam(lam)
                          + rng.normal(0.0, rand_scale, size=problem.d))

    d = problem.d
    L_max = np.zeros(problem.K)
    argmax_theta = [None] * problem.K
    for th in thetas:
        for k in range(problem.K):
            H = np.empty((d, d))
            for j in range(d):
                e = np.zeros(d)
                e[j] = fd_h
                H[:, j] = (problem._gradF(th + e, k)
                           - problem._gradF(th - e, k)) / (2.0 * fd_h)
            H = 0.5 * (H + H.T)
            spec = float(np.linalg.norm(H, 2))
            if spec > L_max[k]:
                L_max[k] = spec
                argmax_theta[k] = np.asarray(th, dtype=float).copy()
    return {
        "L": safety * L_max,
        "L_raw_max": L_max.copy(),
        "safety": float(safety),
        "n_hessians": len(thetas),
        "argmax_theta": argmax_theta,
    }
