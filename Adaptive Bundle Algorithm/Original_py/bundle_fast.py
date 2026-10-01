"""bundle_fast.py  –  Bundle with a Gram cache + delivery-time pruning.

NEW FILE (July 15, 2026).  Part of the ``_fast`` implementation set; the
original ``bundle.py`` is untouched and remains the reference.

Two additions over ``bundle.Bundle``:

1.  **Gram cache** (``grams``): for every bundle point the K×K matrix

        M_i = J_i @ J_i.T          (J_i the K×d Jacobian at x_i)

    is computed once at ``add_point`` time (O(K² d), orders of magnitude
    cheaper than the joint-oracle call that produced J_i) and kept in
    sync by ``pop_point`` / ``__post_init__``.  It enables the exact
    identity used by the fast λ-search
    (``algorithm_fast_without_256_checkpoints._gn_value_batched_gram``):

        GN(λ) = min_i ‖J_i^T λ‖²  =  min_i λ^T M_i λ ,   ∇_λ = 2 M_{i*} λ

    turning every GN evaluation from O(m·K·d) into O(m·K²) — an exact
    algebraic rewrite, not an approximation.

2.  **Delivery-time pruning** (``prune_inactive``): the run-time policy is
    inclusive (adding a point only ever weakly improves the min-based
    criterion), so pruning happens ONLY at delivery: a point is kept iff
    it is the GN argmin ("activated") for at least one λ in a probe set
    (simplex grid ∪ caller-supplied λ's, e.g. the final strict-search
    winners).  The function asserts that GN at every probe λ is bitwise
    unchanged by the pruning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from bundle import Bundle


# ---------------------------------------------------------------------------
# Bundle subclass with Gram cache
# ---------------------------------------------------------------------------
@dataclass
class BundleFast(Bundle):
    """``bundle.Bundle`` plus a per-point Gram cache ``grams``.

    ``grams[i] = grads[i] @ grads[i].T``  (shape (K, K), float64).

    All validation and storage semantics are inherited; the cache is
    maintained on every mutation so ``gram_stack()`` is always consistent
    with ``grads``.
    """

    grams: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Rebuild unconditionally from the validated Jacobians so a caller
        # can never hand us a stale / inconsistent cache.
        self.grams = [J @ J.T for J in self.grads]

    def add_point(self, x, objectives, grad_objectives, joint_oracle=None):
        super().add_point(x, objectives, grad_objectives,
                          joint_oracle=joint_oracle)
        J = self.grads[-1]
        self.grams.append(J @ J.T)

    def pop_point(self):
        super().pop_point()
        self.grams.pop()

    def gram_stack(self) -> np.ndarray:
        """Return the stacked Gram cache, shape (m, K, K)."""
        return np.asarray(self.grams)


# ---------------------------------------------------------------------------
# Simplex grid (compositions), used by the pruning probe set
# ---------------------------------------------------------------------------
def simplex_grid(K: int, resolution: int) -> np.ndarray:
    """All λ on the uniform simplex grid {c/r : c ∈ N^K, Σc = r}.

    Returns shape (C(r+K-1, K-1), K).  K=6, r=10 → 3003 nodes; on the
    Gram path scoring all of them is a few milliseconds.
    """
    if K < 1 or resolution < 1:
        raise ValueError("K and resolution must be positive integers.")
    if K == 1:
        return np.ones((1, 1))

    out: List[np.ndarray] = []
    comp = np.zeros(K, dtype=np.int64)

    def rec(pos: int, remaining: int) -> None:
        if pos == K - 1:
            comp[pos] = remaining
            out.append(comp / float(resolution))
            return
        for c in range(remaining + 1):
            comp[pos] = c
            rec(pos + 1, remaining - c)

    rec(0, resolution)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Activation detection + delivery-time pruning
# ---------------------------------------------------------------------------
def active_indices(bundle: BundleFast,
                   lam_probes: Iterable[np.ndarray]) -> set:
    """Indices i that realise min_i λ^T M_i λ for at least one probe λ."""
    Ms = bundle.gram_stack()
    if Ms.shape[0] == 0:
        return set()
    act: set = set()
    for lam in lam_probes:
        lam = np.asarray(lam, dtype=float)
        vals = np.einsum('k,ikl,l->i', lam, Ms, lam)
        act.add(int(np.argmin(vals)))
    return act


def prune_inactive(
    bundle: BundleFast,
    *,
    grid_resolution: int = 10,
    extra_lams: Optional[Sequence[np.ndarray]] = None,
    keep_indices: Optional[Iterable[int]] = None,
) -> Dict:
    """Delivery-time pruning: keep only λ-activated points (in place).

    Probe set = simplex grid (``grid_resolution``) ∪ ``extra_lams`` (pass
    the final strict λ-search winners here so the certificate's argmin is
    guaranteed kept).  ``keep_indices`` force-keeps points regardless.

    Invariant (asserted): for every probe λ, min_i λ^T M_i λ over the kept
    set equals the value over the full set bitwise — pruning never touches
    any certificate value.

    Returns a report dict; the bundle is modified in place.
    """
    m_before = bundle.m
    grid = simplex_grid(bundle.K, grid_resolution)
    probes: List[np.ndarray] = [grid[i] for i in range(grid.shape[0])]
    if extra_lams is not None:
        probes.extend(np.asarray(l, dtype=float) for l in extra_lams)

    act = active_indices(bundle, probes)
    if keep_indices is not None:
        act |= {int(i) for i in keep_indices}
    kept = sorted(act)

    # Certificate invariance: values at the activated argmins are untouched,
    # so the min over the kept subset must equal the min over the full set.
    Ms_full = bundle.gram_stack()
    for lam in probes:
        vals = np.einsum('k,ikl,l->i', lam, Ms_full, lam)
        full_min = float(np.min(vals))
        kept_min = float(np.min(vals[kept]))
        if kept_min != full_min:
            raise AssertionError(
                "prune_inactive would change a probe GN value "
                f"({full_min!r} -> {kept_min!r}); refusing to prune."
            )

    bundle.points = [bundle.points[i] for i in kept]
    bundle.fvals = [bundle.fvals[i] for i in kept]
    bundle.grads = [bundle.grads[i] for i in kept]
    bundle.grams = [bundle.grams[i] for i in kept]

    return {
        "m_before": m_before,
        "m_after": bundle.m,
        "kept_indices": kept,
        "n_probe_lams": len(probes),
        "grid_resolution": grid_resolution,
    }
