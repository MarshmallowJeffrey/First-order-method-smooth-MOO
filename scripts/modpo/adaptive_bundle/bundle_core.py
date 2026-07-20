from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import warnings

import numpy as np

try:
    from scipy.optimize import minimize as scipy_minimize
except ImportError:
    scipy_minimize = None

try:
    from cyipopt import minimize_ipopt as ipopt_minimize
    HAS_IPOPT = True
    IPOPT_IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    ipopt_minimize = None
    HAS_IPOPT = False
    IPOPT_IMPORT_ERROR = exc


def ipopt_available() -> bool:
    return HAS_IPOPT


def ipopt_import_error() -> Optional[BaseException]:
    return IPOPT_IMPORT_ERROR


@dataclass
class FirstOrderBundle:
    """Bundle of points, objective values, and per-objective gradients.

    Each entry stores
        x_i,
        (F_1(x_i), ..., F_K(x_i)),
        (grad F_1(x_i), ..., grad F_K(x_i)).

    For LLM/LoRA runs `d` can be large, so points and gradients are stored in
    float32 by default. Lambda optimization still returns float64 scalars.
    """

    K: int
    d: int
    L: Sequence[float]
    dtype: np.dtype = np.float32
    points: List[np.ndarray] = field(default_factory=list)
    fvals: List[np.ndarray] = field(default_factory=list)
    grads: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError("K must be at least 1")
        if self.d < 1:
            raise ValueError("d must be at least 1")
        self.L = np.asarray(self.L, dtype=np.float64)
        if self.L.shape != (self.K,):
            raise ValueError(f"L must have shape ({self.K},), got {self.L.shape}")
        if np.any(~np.isfinite(self.L)) or np.any(self.L <= 0.0):
            raise ValueError("L must contain finite positive values")
        self.dtype = np.dtype(self.dtype)

    @property
    def m(self) -> int:
        return len(self.points)

    def add(
        self,
        x: Sequence[float],
        fvals: Sequence[float],
        grads: Sequence[Sequence[float]],
    ) -> None:
        x_arr = np.asarray(x, dtype=self.dtype)
        f_arr = np.asarray(fvals, dtype=np.float64)
        g_arr = np.asarray(grads, dtype=self.dtype)

        if x_arr.shape != (self.d,):
            raise ValueError(f"x must have shape ({self.d},), got {x_arr.shape}")
        if f_arr.shape != (self.K,):
            raise ValueError(f"fvals must have shape ({self.K},), got {f_arr.shape}")
        if g_arr.shape != (self.K, self.d):
            raise ValueError(f"grads must have shape ({self.K}, {self.d}), got {g_arr.shape}")
        if np.any(~np.isfinite(x_arr)) or np.any(~np.isfinite(f_arr)) or np.any(~np.isfinite(g_arr)):
            raise ValueError("Bundle entries must be finite")

        self.points.append(x_arr.copy())
        self.fvals.append(f_arr.copy())
        self.grads.append(g_arr.copy())

    def pop(self) -> None:
        self.points.pop()
        self.fvals.pop()
        self.grads.pop()


def project_simplex(values: Sequence[float]) -> np.ndarray:
    """Clip to the probability simplex by nonnegative renormalization."""
    arr = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(arr.shape[0], 1.0 / arr.shape[0], dtype=np.float64)
    return arr / total


def project_truncated_simplex(values: Sequence[float], lambda_min: float = 0.0) -> np.ndarray:
    """Clip to {lambda: sum lambda=1, lambda_k >= lambda_min} by renormalization."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] < 1:
        raise ValueError("lambda values must be a non-empty one-dimensional array")
    if not np.isfinite(lambda_min) or lambda_min < 0.0:
        raise ValueError("lambda_min must be finite and non-negative")
    K = arr.shape[0]
    if lambda_min * K >= 1.0:
        raise ValueError("lambda_min must be smaller than 1 / K")
    if lambda_min == 0.0:
        return project_simplex(arr)

    free_mass = 1.0 - lambda_min * K
    shifted = np.maximum(arr - lambda_min, 0.0)
    total = float(shifted.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(K, 1.0 / K, dtype=np.float64)
    return lambda_min + free_mass * shifted / total


def _bundle_grads(bundle: FirstOrderBundle) -> np.ndarray:
    return np.asarray(bundle.grads, dtype=bundle.dtype)


def _lambda_selection_grads(
    bundle: FirstOrderBundle,
    lambda_normalization: str = "none",
) -> Tuple[np.ndarray, np.ndarray]:
    if lambda_normalization not in {"none", "global_mean"}:
        raise ValueError("lambda_normalization must be either 'none' or 'global_mean'")

    Jmat = _bundle_grads(bundle)
    if lambda_normalization == "none":
        return Jmat, np.ones(bundle.K, dtype=np.float64)

    norms = np.linalg.norm(Jmat.astype(np.float64, copy=False), axis=2)
    scales = norms.mean(axis=0)
    eps = np.finfo(np.float64).eps
    scales = np.where(scales > eps, scales, 1.0)
    Jmat_scaled = Jmat.astype(np.float64, copy=False) / scales[None, :, None]
    return Jmat_scaled.astype(bundle.dtype, copy=False), scales.astype(np.float64, copy=False)


def _gn_value_batched(Jmat: np.ndarray, lam: np.ndarray) -> float:
    lam = lam.astype(Jmat.dtype, copy=False)
    weighted_grads = np.einsum("mkd,k->md", Jmat, lam, optimize=True)
    gnorms_sq = np.einsum("md,md->m", weighted_grads, weighted_grads, optimize=True)
    return float(np.min(gnorms_sq))


def gn_value_at_lambda(
    bundle: FirstOrderBundle,
    lam: Sequence[float],
    lambda_normalization: str = "none",
    lambda_min: float = 0.0,
) -> float:
    Jmat, _ = _lambda_selection_grads(bundle, lambda_normalization=lambda_normalization)
    return _gn_value_batched(Jmat, project_truncated_simplex(lam, lambda_min=lambda_min))


def bundle_gradient_diagnostics(bundle: FirstOrderBundle) -> dict:
    """Summarize per-objective gradient scales and cosines over the bundle."""
    Jmat = _bundle_grads(bundle).astype(np.float64, copy=False)
    norms = np.linalg.norm(Jmat, axis=2)
    diagnostics = {
        "grad_norm_mean": norms.mean(axis=0).tolist(),
        "grad_norm_min": norms.min(axis=0).tolist(),
        "grad_norm_max": norms.max(axis=0).tolist(),
        "grad_norm_latest": norms[-1].tolist(),
    }
    if bundle.K == 2:
        dots = np.einsum("md,md->m", Jmat[:, 0, :], Jmat[:, 1, :], optimize=True)
        denom = norms[:, 0] * norms[:, 1]
        cosines = np.divide(
            dots,
            denom,
            out=np.zeros_like(dots, dtype=np.float64),
            where=denom > 0.0,
        )
        diagnostics.update({
            "grad_cosine_mean": float(cosines.mean()),
            "grad_cosine_min": float(cosines.min()),
            "grad_cosine_max": float(cosines.max()),
            "grad_cosine_latest": float(cosines[-1]),
        })
    return diagnostics


def gn_grid_diagnostics(
    bundle: FirstOrderBundle,
    num_points: int = 21,
    lambda_normalization: str = "none",
    lambda_min: float = 0.0,
) -> list:
    """Evaluate GN(lambda; bundle) on a 2-objective helpful-weight grid."""
    if bundle.K != 2:
        return []
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not np.isfinite(lambda_min) or lambda_min < 0.0 or lambda_min >= 0.5:
        raise ValueError("For K=2, lambda_min must be finite and in [0, 0.5)")
    Jmat, scales = _lambda_selection_grads(
        bundle,
        lambda_normalization=lambda_normalization,
    )
    rows = []
    for helpful_weight in np.linspace(lambda_min, 1.0 - lambda_min, num_points):
        lam = np.array([helpful_weight, 1.0 - helpful_weight], dtype=np.float64)
        rows.append({
            "lambda_helpful": float(helpful_weight),
            "lambda_harmless": float(1.0 - helpful_weight),
            "gn": _gn_value_batched(Jmat, lam),
            "lambda_normalization": lambda_normalization,
            "lambda_normalization_scales": scales.tolist(),
            "lambda_min": float(lambda_min),
        })
    return rows


def _gn_value_and_jac_batched(Jmat: np.ndarray, lam: np.ndarray) -> Tuple[float, np.ndarray]:
    lam_for_mat = lam.astype(Jmat.dtype, copy=False)
    weighted_grads = np.einsum("mkd,k->md", Jmat, lam_for_mat, optimize=True)
    gnorms_sq = np.einsum("md,md->m", weighted_grads, weighted_grads, optimize=True)
    i_star = int(np.argmin(gnorms_sq))
    grad_lam = 2.0 * (
        Jmat[i_star].astype(np.float64, copy=False)
        @ weighted_grads[i_star].astype(np.float64, copy=False)
    )
    return float(gnorms_sq[i_star]), grad_lam


def _gn_multistart_set(
    K: int,
    prev_lam: Optional[np.ndarray],
    max_starts: int,
    lambda_min: float = 0.0,
) -> List[np.ndarray]:
    if K == 1:
        return [np.ones(1, dtype=np.float64)]
    if max_starts < 1:
        raise ValueError("max_starts must be at least 1")
    if not np.isfinite(lambda_min) or lambda_min < 0.0 or lambda_min * K >= 1.0:
        raise ValueError("lambda_min must be finite and smaller than 1 / K")

    eps = max(1e-8, float(lambda_min))
    starts: List[np.ndarray] = []

    def room() -> int:
        return max_starts - len(starts)

    if room() > 0:
        starts.append(np.full(K, 1.0 / K))
    if room() > 0:
        for k in range(min(K, room())):
            start = np.full(K, eps)
            start[k] = 1.0 - (K - 1) * eps
            starts.append(start)
    if room() > 0 and K > 1:
        for k in range(min(K, room())):
            start = np.full(K, 0.2 / (K - 1))
            start[k] = 0.8
            starts.append(start)
    if room() > 0 and prev_lam is not None:
        starts.append(project_truncated_simplex(prev_lam, lambda_min=lambda_min))
    if room() > 0:
        for a in range(K):
            for b in range(a + 1, K):
                if room() <= 0:
                    break
                start = np.full(K, eps)
                start[a] = 0.5 - (K - 2) * 0.5 * eps
                start[b] = 0.5 - (K - 2) * 0.5 * eps
                starts.append(start)
            if room() <= 0:
                break
    return starts


def maximise_gn(
    bundle: FirstOrderBundle,
    prev_lam: Optional[np.ndarray] = None,
    max_starts: int = 64,
    solver: str = "ipopt",
    require_ipopt: bool = False,
    lambda_normalization: str = "none",
    lambda_min: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """Approximate argmax_lambda min_i ||sum_k lambda_k grad F_k(x_i)||^2."""
    if solver not in {"ipopt", "slsqp"}:
        raise ValueError("solver must be either 'ipopt' or 'slsqp'")
    if lambda_normalization not in {"none", "global_mean"}:
        raise ValueError("lambda_normalization must be either 'none' or 'global_mean'")
    if not np.isfinite(lambda_min) or lambda_min < 0.0 or lambda_min * bundle.K >= 1.0:
        raise ValueError("lambda_min must be finite and smaller than 1 / K")
    if bundle.m == 0:
        raise ValueError("Cannot maximize GN for an empty bundle")
    if bundle.K == 1:
        lam = np.ones(1, dtype=np.float64)
        Jmat, _ = _lambda_selection_grads(bundle, lambda_normalization=lambda_normalization)
        return _gn_value_batched(Jmat, lam), lam

    Jmat, _ = _lambda_selection_grads(bundle, lambda_normalization=lambda_normalization)

    def neg_gn(lam: np.ndarray) -> float:
        value, _ = _gn_value_and_jac_batched(Jmat, lam)
        return -value

    def neg_gn_jac(lam: np.ndarray) -> np.ndarray:
        _, jac = _gn_value_and_jac_batched(Jmat, lam)
        return -jac

    starts = _gn_multistart_set(
        bundle.K,
        prev_lam=prev_lam,
        max_starts=max_starts,
        lambda_min=lambda_min,
    )
    best_neg_value = np.inf
    best_lam = project_truncated_simplex(starts[0], lambda_min=lambda_min)

    constraints = [{
        "type": "eq",
        "fun": lambda lam: float(np.sum(lam) - 1.0),
        "jac": lambda lam: np.ones(bundle.K, dtype=np.float64),
    }]
    bounds = [(max(1e-8, lambda_min), 1.0)] * bundle.K
    use_ipopt = solver == "ipopt" and HAS_IPOPT
    if solver == "ipopt" and not HAS_IPOPT:
        message = (
            "cyipopt/IPOPT is unavailable; GN lambda maximization would fall "
            f"back to SLSQP. Import error: {IPOPT_IMPORT_ERROR}"
        )
        if require_ipopt:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    for start in starts:
        start = project_truncated_simplex(start, lambda_min=lambda_min)
        start_neg_value = neg_gn(start)
        if start_neg_value < best_neg_value:
            best_neg_value = float(start_neg_value)
            best_lam = start

        if use_ipopt:
            try:
                result = ipopt_minimize(
                    neg_gn,
                    start,
                    jac=neg_gn_jac,
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        "print_level": 0,
                        "sb": "yes",
                        "tol": 1e-8,
                        "max_iter": 100,
                        "hessian_approximation": "limited-memory",
                    },
                )
            except Exception as exc:
                warnings.warn(f"GN IPOPT solve failed from start {start}: {exc}", RuntimeWarning)
                continue
        elif scipy_minimize is None:
            continue
        else:
            try:
                result = scipy_minimize(
                    neg_gn,
                    start,
                    jac=neg_gn_jac,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"ftol": 1e-6, "maxiter": 60},
                )
            except Exception as exc:
                warnings.warn(f"GN SLSQP solve failed from start {start}: {exc}", RuntimeWarning)
                continue

        candidate_lam = project_truncated_simplex(result.x, lambda_min=lambda_min)
        candidate_neg_value = neg_gn(candidate_lam)
        if np.isfinite(candidate_neg_value) and candidate_neg_value < best_neg_value:
            best_neg_value = float(candidate_neg_value)
            best_lam = candidate_lam

    return float(-best_neg_value), best_lam


def t_map_step(
    bundle: FirstOrderBundle,
    lam: Sequence[float],
    L_scale: float = 1.0,
) -> Tuple[np.ndarray, int, float, float, float]:
    """Compute one T-map candidate: x_new = x_i - grad F_lambda(x_i) / L_lambda."""
    if bundle.m == 0:
        raise ValueError("Cannot run a T-map step for an empty bundle")

    lam_arr = project_simplex(lam)
    Fmat = np.asarray(bundle.fvals, dtype=np.float64)
    Jmat = _bundle_grads(bundle)
    Pmat = np.asarray(bundle.points, dtype=bundle.dtype)
    L_lam = float(lam_arr @ (bundle.L * L_scale))
    if not np.isfinite(L_lam) or L_lam <= 0.0:
        raise ValueError("L_lambda must be finite and positive")

    lam_for_mat = lam_arr.astype(Jmat.dtype, copy=False)
    F_lam = Fmat @ lam_arr
    grad_lam = np.einsum("mkd,k->md", Jmat, lam_for_mat, optimize=True)
    gnorm_sq = np.einsum("md,md->m", grad_lam, grad_lam, optimize=True)
    scores = F_lam - 0.5 * gnorm_sq.astype(np.float64) / L_lam
    i_star = int(np.argmin(scores))
    x_new = Pmat[i_star].astype(np.float64, copy=False) - grad_lam[i_star].astype(np.float64, copy=False) / L_lam
    return (
        x_new.astype(bundle.dtype, copy=False),
        i_star,
        float(gnorm_sq[i_star]),
        float(scores[i_star]),
        L_lam,
    )


def prune_last_candidates(
    bundle: FirstOrderBundle,
    base_m: int,
    steps_taken: int,
    lam: Sequence[float],
) -> None:
    """Keep only the new candidate with smallest ||grad F_lambda||."""
    if steps_taken <= 1:
        return
    lam_arr = project_simplex(lam)
    cand_grads = np.asarray(bundle.grads[base_m:base_m + steps_taken], dtype=bundle.dtype)
    grad_lam = np.einsum("skd,k->sd", cand_grads, lam_arr.astype(bundle.dtype), optimize=True)
    gnorms_sq = np.einsum("sd,sd->s", grad_lam, grad_lam, optimize=True)
    best_local = int(np.argmin(gnorms_sq))
    best_idx = base_m + best_local

    keep_point = bundle.points[best_idx].copy()
    keep_fvals = bundle.fvals[best_idx].copy()
    keep_grads = bundle.grads[best_idx].copy()
    for _ in range(steps_taken):
        bundle.pop()
    bundle.points.append(keep_point)
    bundle.fvals.append(keep_fvals)
    bundle.grads.append(keep_grads)
