"""objectives_bandit_toy_mv.py — mean-variance offline-bandit toy (K=2).

NEW FILE (July 31, 2026).  User-ordered follow-up to the July-26 K=2
convex instance: replace the linear-in-pi reward term by a MEAN-VARIANCE
utility so the closed-form softmax oracle dies and the scalarizations
become genuinely nonconvex.  Design agreed in the July-26/27 session
Q&A (ledger: "to kill the closed form use mean-variance; ground truth
then = never-timed multi-start reference solves").

Objective
---------
Same instance data as ``objectives_bandit_toy`` (balanced offline
dataset, plug-in per-arm means R_hat), plus per-arm SECOND moments
    S_hat[k, a] = mean over rows t with a_t = a of r_{k,t}^2.
With pi = softmax([theta, 0]) (reduced logits, unchanged):

    F_k(theta) = tau*KL(pi||pi_ref) - <pi, Rk_hat> + gamma*VarHat_k(pi),
    VarHat_k(pi) = <pi, Sk_hat> - <pi, Rk_hat>^2.

VarHat_k is the plug-in variance of the reward collected under pi
(within-arm noise + between-arm spread); minimising F_k trades mean
reward against reward risk.  gamma = 0 reproduces the July-26 objective
EXACTLY (bit-for-bit, sanity-checked).

Why the closed form dies: the scalarized objective

    F_lam = tau*KL - <pi, m_lam> + gamma * sum_k lam_k (<pi,Sk> - <pi,Rk>^2)

is no longer "convex + linear" in pi — each -gamma*lam_k*<pi,Rk>^2 term
is a CONCAVE quadratic — so no softmax closed form exists, F_lam can be
nonconvex in pi, and a scalarization can carry multiple local minima
(commit-to-high-R1 vs commit-to-high-R2 basins).  Stationary points
remain interior: any stationary pi satisfies a softmax fixed-point
equation with bounded effective logits, so theta stays bounded.

Finite-sum view for SVRG (Option B, exact-deterministic part)
-------------------------------------------------------------
    f_{k,t}(theta) = tau*KL + gamma*VarHat_k(pi)
                     - (T/N_{a_t}) * pi(a_t) * r_{k,t},
so F_k = (1/T) sum_t f_{k,t} exactly.  The KL AND the variance penalty
are arm-level statistics available to the solver (same plug-in principle
that already put the exact KL in every row of the July-26 oracle); the
minibatch variance comes only from the mean-reward rows, so the MSVRG
variance-reduction identity is untouched and the estimator is unbiased:
in v = g_S(y) - g_S(anchor) + mu(anchor), the exact MV part cancels
between g_S(anchor) and mu(anchor) and survives exactly at y.

Ground truth (never timed)
--------------------------
No closed form exists, so every oracle quantity (theta_star, f_pf,
scalarized_opt, the front, the arc CDF) is served from a REFERENCE
TABLE built by a multi-start solver:

    1. vectorised Adam over (dense w-grid x ~29 structured+random
       starts) run simultaneously,
    2. scipy L-BFGS-B polish of the per-w winner (analytic gradients),
    3. ascending + descending relax sweeps (polish from the neighbour's
       winner, accept improvements) until no change, so a basin found
       anywhere propagates along the whole path.

The table is a BEST-KNOWN pool, not a certificate.  After the methods
run, ``refresh_with_points`` polishes from every method-delivered point
at every query weight; any improvement > 1e-9 updates the table + cache
and is counted/reported (delta_value stays >= -1e-9 by construction).
The SURF speed formula (Eq. 9) was closed-form-specific; the arc CDF is
now the normalised CHORDAL arc length of the reference front, and the
Rule-1 arc-uniform weights read that CDF.  All of this is off both cost
axes.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from objectives_bandit_toy import (BanditStochOracle, BanditToyProblem,
                                   _safe_log, _softmax)

__all__ = [
    "BanditToyMVProblem",
    "BanditStochOracleMV",
    "make_bandit_toy_mv",
]


# =====================================================================
#  Problem
# =====================================================================
class BanditToyMVProblem(BanditToyProblem):
    """Mean-variance bandit toy.  K = 2 only (the reference machinery is
    w-parameterised); the closed-form *_lam API of the parent is
    poisoned to fail loudly."""

    def __init__(self, *, gamma: float, ref_n_dense: int = 5000,
                 ref_n_random_starts: int = 12, ref_adam_steps: int = 1500,
                 ref_seed: int = 0, ref_coarse_every: int = 25, **kw) -> None:
        super().__init__(**kw)
        if self.K != 2:
            raise ValueError("BanditToyMVProblem is K=2 only.")
        self.gamma = float(gamma)

        # Plug-in per-arm second moments + their true counterparts.
        S_hat = np.zeros((self.K, self.A))
        for k in range(self.K):
            S_hat[k] = np.bincount(self.actions,
                                   weights=self.rewards[k] ** 2,
                                   minlength=self.A) / self.counts
        self.S_hat = S_hat
        self.S_true = self.R_true ** 2 + self.noise_std ** 2

        # Reference-solver settings (recorded into the cache).
        self.ref_cfg = {
            "n_dense": int(ref_n_dense),
            "n_random_starts": int(ref_n_random_starts),
            "adam_steps": int(ref_adam_steps),
            "seed": int(ref_seed),
            "coarse_every": int(ref_coarse_every),
        }
        self._ref: Optional[Dict] = None        # plug-in table
        self._ref_true: Optional[Dict] = None   # true-parameter table
        self._query_memo: Dict = {}
        self._cache_path: Optional[str] = None

    # ---------------- deterministic objective surface -----------------
    def _uv(self, pi: np.ndarray, k: int,
            use_true: bool = False) -> Tuple[float, float]:
        R = self.R_true if use_true else self.R_hat
        S = self.S_true if use_true else self.S_hat
        u = float(pi @ R[k])
        return u, float(pi @ S[k]) - u * u

    def _F(self, theta: np.ndarray, k: int) -> float:
        pi, _ = self._pi_z(theta)
        u, var = self._uv(pi, k)
        return self.tau * self._kl(pi) - u + self.gamma * var

    def _gradF(self, theta: np.ndarray, k: int) -> np.ndarray:
        pi, z = self._pi_z(theta)
        u = float(pi @ self.R_hat[k])
        v = (self.tau * z - self.R_hat[k]
             + self.gamma * (self.S_hat[k] - 2.0 * u * self.R_hat[k]))
        g_full = pi * v - pi * float(pi @ v)
        return g_full[: self.d]

    def joint_oracle(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pi, z = self._pi_z(theta)
        kl = self._kl(pi)
        fvals = np.empty(self.K)
        grads = np.empty((self.K, self.d))
        for k in range(self.K):
            u, var = self._uv(pi, k)
            fvals[k] = self.tau * kl - u + self.gamma * var
            v = (self.tau * z - self.R_hat[k]
                 + self.gamma * (self.S_hat[k] - 2.0 * u * self.R_hat[k]))
            g_full = pi * v - pi * float(pi @ v)
            grads[k] = g_full[: self.d]
        return fvals, grads

    def scalarized_value_grad(self, theta: np.ndarray, w: float,
                              use_true: bool = False
                              ) -> Tuple[float, np.ndarray]:
        """(F_w, grad F_w) at theta with F_w = w*F1 + (1-w)*F2."""
        R = self.R_true if use_true else self.R_hat
        S = self.S_true if use_true else self.S_hat
        pi, z = self._pi_z(theta)
        val = self.tau * self._kl(pi)
        vcoef = self.tau * z.copy()
        for k, lk in ((0, float(w)), (1, 1.0 - float(w))):
            u = float(pi @ R[k])
            s = float(pi @ S[k])
            val += lk * (-u + self.gamma * (s - u * u))
            vcoef += lk * (-R[k] + self.gamma * (S[k] - 2.0 * u * R[k]))
        g_full = pi * vcoef - pi * float(pi @ vcoef)
        return val, g_full[: self.d]

    # ---------------- closed-form API is dead under MV ----------------
    def _no_closed_form(self, name: str):
        raise NotImplementedError(
            f"{name}: no closed form under mean-variance (gamma="
            f"{self.gamma:g}); use the reference API.")

    def pi_star_lam(self, lam, use_true: bool = False):
        self._no_closed_form("pi_star_lam")

    def theta_star_lam(self, lam, use_true: bool = False):
        self._no_closed_form("theta_star_lam")

    def f_vec_lam(self, lam):
        self._no_closed_form("f_vec_lam")

    def scalarized_opt_lam(self, lam):
        self._no_closed_form("scalarized_opt_lam")

    def oracle_batch(self, lams):
        self._no_closed_form("oracle_batch")

    def _theta_star_gamma0(self, w: float) -> np.ndarray:
        """Closed-form solution of the gamma = 0 (July-26) objective —
        used only as one structured START for the reference solver."""
        logits = (_safe_log(self.pi_ref)
                  + (w * self.R_hat[0] + (1.0 - w) * self.R_hat[1]) / self.tau)
        z = logits - logits.max()
        return (z[: self.d] - z[self.d]).astype(float)

    # ---------------- reference solver --------------------------------
    def _start_stack(self, w: float, rng: np.random.RandomState) -> np.ndarray:
        """Structured + random starts for one weight w."""
        starts = [np.zeros(self.d), self._theta_star_gamma0(w)]
        for beta in (4.0, 8.0):
            for a in range(self.A):
                z = np.zeros(self.A)
                z[a] = beta
                starts.append(z[: self.d] - z[self.d])
        starts.extend(rng.normal(0.0, 3.0,
                                 size=(self.ref_cfg["n_random_starts"],
                                       self.d)))
        return np.asarray(starts, dtype=float)

    def _vec_value_grad(self, TH: np.ndarray, W: np.ndarray,
                        use_true: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised (values, grads) of F_w over rows of TH (M, d)."""
        R = self.R_true if use_true else self.R_hat
        S = self.S_true if use_true else self.S_hat
        M = TH.shape[0]
        Z = np.concatenate([TH, np.zeros((M, 1))], axis=1)      # (M, A)
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        kl = np.sum(P * (_safe_log(P) - _safe_log(self.pi_ref)[None, :]),
                    axis=1)
        val = self.tau * kl
        vcoef = self.tau * Z.copy()
        for k in range(2):
            lk = W if k == 0 else (1.0 - W)                     # (M,)
            u = P @ R[k]
            s = P @ S[k]
            val += lk * (-u + self.gamma * (s - u * u))
            vcoef += lk[:, None] * (-R[k][None, :]
                                    + self.gamma * (S[k][None, :]
                                                    - 2.0 * u[:, None]
                                                    * R[k][None, :]))
        inner = np.sum(P * vcoef, axis=1, keepdims=True)
        G_full = P * vcoef - P * inner
        return val, G_full[:, : self.d]

    def _polish(self, w: float, th0: np.ndarray, use_true: bool,
                maxiter: int = 300) -> Tuple[float, np.ndarray]:
        res = minimize(lambda th: self.scalarized_value_grad(th, w, use_true),
                       np.asarray(th0, dtype=float), jac=True,
                       method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-18,
                                "gtol": 1e-12})
        return float(res.fun), np.asarray(res.x, dtype=float)

    def _build_table(self, use_true: bool) -> Dict:
        cfg = self.ref_cfg
        n = cfg["n_dense"]
        w_grid = np.linspace(0.0, 1.0, n)
        rng = np.random.RandomState(cfg["seed"] + (1 if use_true else 0))

        # 1. vectorised Adam over (every w) x (all starts).
        starts0 = self._start_stack(w_grid[0], rng)
        n_st = starts0.shape[0]
        TH = np.empty((n * n_st, self.d))
        W = np.repeat(w_grid, n_st)
        TH[:n_st] = starts0
        for i in range(1, n):
            TH[i * n_st:(i + 1) * n_st] = self._start_stack(w_grid[i], rng)
        m1 = np.zeros_like(TH)
        m2 = np.zeros_like(TH)
        lr, b1, b2, eps_ = 0.1, 0.9, 0.999, 1e-8
        for t in range(1, cfg["adam_steps"] + 1):
            _, G = self._vec_value_grad(TH, W, use_true)
            m1 = b1 * m1 + (1 - b1) * G
            m2 = b2 * m2 + (1 - b2) * G * G
            mh = m1 / (1 - b1 ** t)
            vh = m2 / (1 - b2 ** t)
            TH -= lr * mh / (np.sqrt(vh) + eps_)
        vals, _ = self._vec_value_grad(TH, W, use_true)
        vals = vals.reshape(n, n_st)
        THr = TH.reshape(n, n_st, self.d)
        pick = np.argmin(vals, axis=1)
        theta_tab = THr[np.arange(n), pick]

        # basin statistics from the multistart pool (nonconvexity readout)
        n_basins = np.zeros(n, dtype=int)
        for i in range(n):
            v = vals[i]
            th = THr[i]
            near = np.flatnonzero(v <= v.min() + 0.05)
            reps: List[np.ndarray] = []
            for j in near:
                if all(np.max(np.abs(th[j] - r)) > 0.1 for r in reps):
                    reps.append(th[j])
            n_basins[i] = len(reps)

        # 2. polish every winner.
        f_tab = np.empty(n)
        for i in range(n):
            f_tab[i], theta_tab[i] = self._polish(w_grid[i], theta_tab[i],
                                                  use_true)

        # 3. relax sweeps until no improvement.
        sweeps = 0
        while sweeps < 6:
            improved = 0
            for order in (range(1, n), range(n - 2, -1, -1)):
                for i in order:
                    j = i - 1 if order.step == 1 else i + 1  # type: ignore
                    fc, thc = self._polish(w_grid[i], theta_tab[j], use_true,
                                           maxiter=120)
                    if fc < f_tab[i] - 1e-12:
                        f_tab[i], theta_tab[i] = fc, thc
                        improved += 1
            sweeps += 1
            if improved == 0:
                break

        # objective vectors along the table (for the front / arc CDF)
        fvec_tab = self._fvec_rows(theta_tab, use_true)
        return {"w_grid": w_grid, "theta": theta_tab, "fscal": f_tab,
                "fvec": fvec_tab, "n_basins": n_basins,
                "relax_sweeps": sweeps, "use_true": bool(use_true)}

    def _fvec_rows(self, TH: np.ndarray, use_true: bool) -> np.ndarray:
        R = self.R_true if use_true else self.R_hat
        S = self.S_true if use_true else self.S_hat
        M = TH.shape[0]
        Z = np.concatenate([TH, np.zeros((M, 1))], axis=1)
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        kl = np.sum(P * (_safe_log(P) - _safe_log(self.pi_ref)[None, :]),
                    axis=1)
        out = np.empty((M, 2))
        for k in range(2):
            u = P @ R[k]
            s = P @ S[k]
            out[:, k] = self.tau * kl - u + self.gamma * (s - u * u)
        return out

    # ---------------- cache ------------------------------------------
    def _cache_meta(self) -> Dict:
        return {"gamma": self.gamma, "T": self.T, "A": self.A,
                "tau": self.tau, "alpha": self.alpha,
                "noise_std": self.noise_std, "data_seed": self.data_seed,
                **self.ref_cfg}

    def ensure_reference(self, cache_path: Optional[str] = None,
                         verbose: bool = True) -> Dict:
        """Build or load the plug-in + true reference tables."""
        if self._ref is not None:
            return self._provenance()
        if cache_path and os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=False)
            meta = json.loads(str(z["meta_json"]))
            if meta == self._cache_meta():
                self._ref = {k: z[f"plug_{k}"] for k in
                             ("w_grid", "theta", "fscal", "fvec", "n_basins")}
                self._ref["relax_sweeps"] = int(z["plug_relax_sweeps"])
                self._ref_true = {k: z[f"true_{k}"] for k in
                                  ("w_grid", "theta", "fscal", "fvec",
                                   "n_basins")}
                self._ref_true["relax_sweeps"] = int(z["true_relax_sweeps"])
                self._cache_path = cache_path
                if verbose:
                    print(f"   reference loaded from cache: {cache_path}",
                          flush=True)
                return self._provenance()
            if verbose:
                print("   reference cache META MISMATCH — rebuilding",
                      flush=True)
        self._ref = self._build_table(use_true=False)
        self._ref_true = self._build_table(use_true=True)
        self._cache_path = cache_path
        if cache_path:
            self._save_cache()
            if verbose:
                print(f"   reference built + cached: {cache_path}", flush=True)
        return self._provenance()

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        payload = {"meta_json": json.dumps(self._cache_meta(),
                                           sort_keys=True)}
        for tag, tab in (("plug", self._ref), ("true", self._ref_true)):
            for k in ("w_grid", "theta", "fscal", "fvec", "n_basins"):
                payload[f"{tag}_{k}"] = tab[k]
            payload[f"{tag}_relax_sweeps"] = tab["relax_sweeps"]
        np.savez_compressed(self._cache_path, **payload)

    def _provenance(self) -> Dict:
        out = {}
        for tag, tab in (("plugin", self._ref), ("true", self._ref_true)):
            nb = np.asarray(tab["n_basins"])
            dj = np.linalg.norm(np.diff(tab["fvec"], axis=0), axis=1)
            out[tag] = {
                "n_dense": int(len(tab["w_grid"])),
                "relax_sweeps": int(tab["relax_sweeps"]),
                "bimodal_fraction": float(np.mean(nb >= 2)),
                "max_front_jump": float(dj.max()),
                "front_jump_at_w": float(tab["w_grid"][int(dj.argmax())]),
            }
        return out

    # ---------------- query API (drop-in for the driver) --------------
    def _table(self, use_true: bool) -> Dict:
        tab = self._ref_true if use_true else self._ref
        if tab is None:
            raise RuntimeError("call ensure_reference() first")
        return tab

    def _query(self, w: float, use_true: bool = False
               ) -> Tuple[float, np.ndarray, np.ndarray]:
        key = (round(float(w), 12), bool(use_true))
        hit = self._query_memo.get(key)
        if hit is not None:
            return hit
        tab = self._table(use_true)
        i = int(np.clip(round(float(w) * (len(tab["w_grid"]) - 1)),
                        0, len(tab["w_grid"]) - 1))
        f, th = self._polish(float(w), tab["theta"][i], use_true, maxiter=120)
        # keep the table monotone-best when the query lands on a grid w
        if abs(tab["w_grid"][i] - float(w)) < 1e-15 and f < tab["fscal"][i]:
            tab["fscal"][i] = f
            tab["theta"][i] = th
            tab["fvec"][i] = self._fvec_rows(th[None, :], use_true)[0]
        fvec = self._fvec_rows(th[None, :], use_true)[0]
        self._query_memo[key] = (f, fvec, th)
        return f, fvec, th

    def pi_star(self, w: float, use_true: bool = False) -> np.ndarray:
        _, _, th = self._query(w, use_true)
        return self.pi_of_theta(th)

    def theta_star(self, w: float, use_true: bool = False) -> np.ndarray:
        return self._query(w, use_true)[2]

    def f_pf(self, w: float) -> np.ndarray:
        return self._query(w, False)[1]

    def scalarized_opt(self, w: float) -> float:
        return self._query(w, False)[0]

    def speed(self, w: float, use_true: bool = False) -> float:
        """Numeric chordal speed ||d f_pf / d w|| from the table (the
        SURF Eq.-9 closed form died with the softmax oracle)."""
        tab = self._table(use_true)
        wg, fv = tab["w_grid"], tab["fvec"]
        i = int(np.clip(round(float(w) * (len(wg) - 1)), 1, len(wg) - 2))
        return float(np.linalg.norm(fv[i + 1] - fv[i - 1])
                     / (wg[i + 1] - wg[i - 1]))

    def arc_cdf(self, n_dense: int = 5000,
                use_true: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        tab = self._table(use_true)
        seg = np.linalg.norm(np.diff(tab["fvec"], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] <= 0.0:
            return tab["w_grid"], tab["w_grid"].copy()
        return tab["w_grid"], s / s[-1]

    def cdf_sup_gap(self, n_dense: int = 5000) -> float:
        _, c_hat = self.arc_cdf(use_true=False)
        _, c_true = self.arc_cdf(use_true=True)
        return float(np.max(np.abs(c_hat - c_true)))

    def second_moment_sup_gap(self) -> float:
        return float(np.max(np.abs(self.S_hat - self.S_true)))

    # ---------------- method-seeded refresh ---------------------------
    def refresh_with_points(self, points: np.ndarray, fvals: np.ndarray,
                            extra_w_grids: List[np.ndarray],
                            tol: float = 1e-9) -> Dict:
        """Polish from the best method point at every table/extra weight;
        accept improvements > tol into the plug-in table (best-known
        semantics).  Returns {n_improved, max_improvement}; re-saves the
        cache and clears the query memo when anything improved."""
        tab = self._table(False)
        points = np.asarray(points, dtype=float)
        fvals = np.asarray(fvals, dtype=float)
        n_improved, max_imp = 0, 0.0
        w_all = np.unique(np.concatenate([tab["w_grid"]]
                                         + [np.asarray(g, dtype=float)
                                            for g in extra_w_grids]))
        scal = fvals @ np.stack([w_all, 1.0 - w_all])       # (m, n_w)
        pick = np.argmin(scal, axis=0)
        wg = tab["w_grid"]
        for j, w in enumerate(w_all):
            f_new, th_new = self._polish(float(w), points[pick[j]], False,
                                         maxiter=200)
            i = int(np.clip(round(float(w) * (len(wg) - 1)), 0, len(wg) - 1))
            on_grid = abs(wg[i] - float(w)) < 1e-15
            f_ref = tab["fscal"][i] if on_grid else self._query(w, False)[0]
            if f_new < f_ref - tol:
                n_improved += 1
                max_imp = max(max_imp, float(f_ref - f_new))
                if on_grid:
                    tab["fscal"][i] = f_new
                    tab["theta"][i] = th_new
                    tab["fvec"][i] = self._fvec_rows(th_new[None, :],
                                                     False)[0]
                else:
                    # spread the newly found basin into the nearest grid
                    # entry, else the improvement is lost when the query
                    # memo is cleared (off-grid queries re-polish from
                    # the nearest grid theta).
                    fc2, thc2 = self._polish(wg[i], th_new, False,
                                             maxiter=200)
                    if fc2 < tab["fscal"][i] - 1e-12:
                        tab["fscal"][i] = fc2
                        tab["theta"][i] = thc2
                        tab["fvec"][i] = self._fvec_rows(thc2[None, :],
                                                         False)[0]
        if n_improved:
            # spread the improvements along the path, then re-cache
            n = len(wg)
            for order in (range(1, n), range(n - 2, -1, -1)):
                for i in order:
                    j2 = i - 1 if order.step == 1 else i + 1  # type: ignore
                    fc, thc = self._polish(wg[i], tab["theta"][j2], False,
                                           maxiter=120)
                    if fc < tab["fscal"][i] - 1e-12:
                        tab["fscal"][i], tab["theta"][i] = fc, thc
                        tab["fvec"][i] = self._fvec_rows(thc[None, :],
                                                         False)[0]
            self._query_memo.clear()
            self._save_cache()
        return {"n_improved": int(n_improved),
                "max_improvement": float(max_imp)}


def make_bandit_toy_mv(*, gamma: float, T: int = 1000, noise_std: float = 0.5,
                       data_seed: int = 7, A: int = 5, tau: float = 0.05,
                       alpha: float = 4.0, **ref_kw) -> BanditToyMVProblem:
    return BanditToyMVProblem(gamma=gamma, A=A, tau=tau, alpha=alpha, T=T,
                              noise_std=noise_std, data_seed=data_seed,
                              **ref_kw)


# =====================================================================
#  Stochastic oracle
# =====================================================================
class BanditStochOracleMV(BanditStochOracle):
    """Minibatch oracle for the mean-variance objective.

    Identical row-sampling contract to the parent (K independent index
    draws, skip-empty-component, IFO += 2*rows in grad_pair).  The
    scalarized estimator adds the EXACT variance-penalty gradient
    (arm-level statistics, zero minibatch variance) next to the exact KL
    part; only the mean-reward term is estimated from rows.  gamma = 0
    adds exact zero vectors, keeping the July-26 arithmetic bit-for-bit.
    """

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
            u_k = float(pi @ p.R_hat[k])
            c_mv = p.gamma * (p.S_hat[k] - 2.0 * u_k * p.R_hat[k])
            mv_full = pi * c_mv - pi * float(pi @ c_mv)
            g += float(lam[k]) * (kl_full[:d] + mv_full[:d] - rew_full[:d])
        return g
