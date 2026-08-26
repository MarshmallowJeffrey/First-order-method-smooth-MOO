"""sanity_checks_fast.py  –  Equivalence / invariance tests for the _fast set.

NEW FILE (July 15, 2026).  Run:  python sanity_checks_fast.py

Checks (all must print PASS):
 1. Gram-path GN value & λ-jacobian ≡ original einsum path (≤ 1e-12).
 2. Stochastic oracle at full batch ≡ J(θ)^T λ from the joint oracle
    (≤ 1e-10) — also proves the re-generated dataset matches the original
    instance bit-for-bit.
 3. Momentum-SVRG degeneration (β=0, p_seg=1, b=n, step_const=1, trigger off)
    reproduces the original T-map inner loop point-for-point (≤ 1e-10).
 4. prune_inactive keeps every probe-λ GN value bitwise unchanged.
 5. Strict-tier fast λ-search agrees with the original maximiser on the
    same bundle (rel ≤ 1e-4; same math, different float path).
 6. Two-tier smoke: algorithm_adaptive_fast runs end-to-end in both tier
    modes on a small instance and returns a complete result dict.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

# torch stack first (matches the repo's import-order convention)...
import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from objectives_torch_fast import make_mlp_nonconvex_fast
from bundle import Bundle
from bundle_fast import BundleFast, prune_inactive, simplex_grid
# ...then the cyipopt-touching algorithm modules.
from algorithm_without_256_checkpoints import (
    _gn_value_batched,
    _gn_value_and_jac_batched,
    _bundle_update_adaptive,
    _maximise_GN,
)
from algorithm_fast_without_256_checkpoints import (
    _gn_value_batched_gram,
    _gn_value_and_jac_batched_gram,
    _maximise_GN_fast,
    _bundle_update_msvrg,
    algorithm_adaptive_fast,
    ipopt_available,
)

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def random_simplex(K: int, rng) -> np.ndarray:
    v = rng.exponential(size=K)
    return v / v.sum()


# ---------------------------------------------------------------------
# 1. Gram ≡ einsum
# ---------------------------------------------------------------------
def test_gram_equivalence() -> None:
    rng = np.random.RandomState(0)
    m, K, d = 17, 5, 43
    Jmat = rng.randn(m, K, d)
    Ms = np.einsum('ikd,ild->ikl', Jmat, Jmat)
    worst_v, worst_j = 0.0, 0.0
    for _ in range(200):
        lam = random_simplex(K, rng)
        v_ref = _gn_value_batched(Jmat, lam)
        v_new = _gn_value_batched_gram(Ms, lam)
        worst_v = max(worst_v, abs(v_ref - v_new) / (1.0 + abs(v_ref)))
        vr, jr, ir = _gn_value_and_jac_batched(Jmat, lam)
        vg, jg, ig = _gn_value_and_jac_batched_gram(Ms, lam)
        worst_v = max(worst_v, abs(vr - vg) / (1.0 + abs(vr)))
        worst_j = max(worst_j,
                      float(np.max(np.abs(jr - jg))) / (1.0 + float(np.max(np.abs(jr)))))
        if ir != ig:
            check("gram argmin agreement", False, f"{ir} vs {ig}")
            return
    check("gram value ≡ einsum value", worst_v <= 1e-12, f"worst rel {worst_v:.2e}")
    check("gram jac ≡ einsum jac", worst_j <= 1e-12, f"worst rel {worst_j:.2e}")


# ---------------------------------------------------------------------
# Small shared MLP instance for tests 2/3/5/6
# ---------------------------------------------------------------------
def build_small():
    K, p, n, hs = 3, 4, 90, [8]
    objectives, grads, L, joint, stoch = make_mlp_nonconvex_fast(
        K=K, p=p, n=n, hidden_sizes=hs, seed=7, activation="tanh",
        n_probes=10, batch_size=n, sampler_seed=11,
    )
    rng = np.random.RandomState(3)
    d = stoch.d
    x0 = rng.randn(d) * 0.3
    return K, p, n, hs, objectives, grads, L, joint, stoch, x0


def test_stoch_full_batch(setup) -> None:
    K, p, n, hs, objectives, grads, L, joint, stoch, x0 = setup
    rng = np.random.RandomState(5)
    worst = 0.0
    for _ in range(5):
        theta = rng.randn(stoch.d) * 0.4
        lam = random_simplex(K, rng)
        fv, gv = joint(theta)
        ref = gv.T @ lam
        stoch.set_anchor(theta)
        g_y, g_a = stoch.grad_pair(theta, lam, stoch.full_batch())
        err = max(float(np.max(np.abs(g_y - ref))),
                  float(np.max(np.abs(g_a - ref))))
        worst = max(worst, err / (1.0 + float(np.max(np.abs(ref)))))
    check("stoch full-batch ≡ J^T λ (data identity)", worst <= 1e-10,
          f"worst rel {worst:.2e}")


# ---------------------------------------------------------------------
# 3. Momentum-SVRG degeneration ≡ original T-map inner loop
# ---------------------------------------------------------------------
def test_degeneration(setup) -> None:
    K, p, n, hs, objectives, grads, L, joint, stoch, x0 = setup
    L_safe = np.asarray(L) * 4.0          # keep the safeguards silent
    lam = np.array([0.5, 0.3, 0.2])
    S = 5

    b_ref = Bundle(K=K, d=stoch.d, L=L_safe)
    b_ref.add_point(x0, objectives, grads, joint_oracle=joint)
    _bundle_update_adaptive(
        b_ref, lam, objectives, grads, max_steps=S,
        eps_inner=None, prune=False, joint_oracle=joint, L_scale=1.0,
    )

    b_new = BundleFast(K=K, d=stoch.d, L=L_safe)
    b_new.add_point(x0, objectives, grads, joint_oracle=joint)
    _bundle_update_msvrg(
        b_new, lam, objectives, grads, joint, stoch,
        eps_inner=None, L_scale=1.0,
        step_const=1.0, momentum=0.0, epoch_len=1, max_segments=S,
    )

    ok = b_ref.m == b_new.m
    worst = 0.0
    if ok:
        for i in range(b_ref.m):
            worst = max(worst, float(np.max(np.abs(
                np.asarray(b_ref.points[i]) - np.asarray(b_new.points[i])))))
        ok = worst <= 1e-10
    check("MSVRG degeneration ≡ T-map inner loop",
          ok, f"m {b_ref.m}/{b_new.m}, worst |Δx| {worst:.2e}")


# ---------------------------------------------------------------------
# 4. prune_inactive invariance
# ---------------------------------------------------------------------
def test_prune() -> None:
    rng = np.random.RandomState(2)
    K, d, m = 3, 12, 25
    b = BundleFast(K=K, d=d, L=np.ones(K))
    for _ in range(m):
        J = rng.randn(K, d)
        b.points.append(rng.randn(d))
        b.fvals.append(rng.randn(K))
        b.grads.append(J)
        b.grams.append(J @ J.T)
    grid = simplex_grid(K, 6)
    Ms_before = b.gram_stack()
    before = [float(np.min(np.einsum('k,ikl,l->i', lam, Ms_before, lam)))
              for lam in grid]
    report = prune_inactive(b, grid_resolution=6)
    Ms_after = b.gram_stack()
    after = [float(np.min(np.einsum('k,ikl,l->i', lam, Ms_after, lam)))
             for lam in grid]
    ok = (report["m_after"] <= report["m_before"]
          and all(x == y for x, y in zip(before, after)))
    check("prune_inactive keeps probe GN bitwise",
          ok, f"m {report['m_before']} -> {report['m_after']}")


# ---------------------------------------------------------------------
# 5. strict fast λ-search vs original maximiser
# ---------------------------------------------------------------------
def test_lambda_search(setup) -> None:
    K, p, n, hs, objectives, grads, L, joint, stoch, x0 = setup
    b = BundleFast(K=K, d=stoch.d, L=np.asarray(L))
    rng = np.random.RandomState(9)
    b.add_point(x0, objectives, grads, joint_oracle=joint)
    for _ in range(6):
        b.add_point(x0 + rng.randn(stoch.d) * 0.2, objectives, grads,
                    joint_oracle=joint)
    v_ref, lam_ref = _maximise_GN(b, prev_lam=None, solver="ipopt",
                                  max_starts=32)
    v_new, lam_new = _maximise_GN_fast(b, prev_lam=None, tier="strict",
                                       max_starts=32)
    rel = abs(v_ref - v_new) / (1.0 + abs(v_ref))
    check("strict fast λ-search ≡ original maximiser",
          rel <= 1e-4, f"ref {v_ref:.6e} vs fast {v_new:.6e}, rel {rel:.2e}")


# ---------------------------------------------------------------------
# 6. end-to-end smoke, both tier modes
# ---------------------------------------------------------------------
def test_smoke(setup) -> None:
    K, p, n, hs, objectives, grads, L, joint, _stoch, x0 = setup
    for mode in ("strict", "two_tier"):
        # fresh sampler so runs are independent
        *_ignored, stoch2 = make_mlp_nonconvex_fast(
            K=K, p=p, n=n, hidden_sizes=hs, seed=7, activation="tanh",
            n_probes=10, batch_size=30, sampler_seed=13,
        )
        # two_tier also exercises the relative inner target; strict keeps
        # rel_target=None so the plain eps/3 path stays covered.
        rel = 0.25 if mode == "two_tier" else None
        res = algorithm_adaptive_fast(
            K=K, d=stoch2.d, objectives=objectives, grad_objectives=grads,
            L=np.asarray(L), x0=x0, stoch_oracle=stoch2,
            epsilon=1e-2, max_outer=6, lambda_max_starts=16,
            lambda_tier_mode=mode, msvrg_max_segments=4,
            msvrg_epoch_len=3, prune_grid_r=4,
            msvrg_rel_target=rel,
            joint_oracle=joint, verbose=False,
        )
        needed = {"cov_history", "cpu_times", "grad_evals_history",
                  "tier_history", "prune_report", "stop_reason",
                  "grad_equiv_total", "joint_calls", "ifo_minibatch_total",
                  "inner_target_history"}
        ok = needed.issubset(res.keys()) and len(res["cov_history"]) >= 2
        tgts = res["inner_target_history"]
        ok = ok and len(tgts) == len(res["pc_history"])
        if rel is not None and tgts:
            eps_floor = 1e-2 / 3.0
            ok = ok and all(
                t is not None and t >= eps_floor - 1e-15
                and abs(t - max(eps_floor, rel * v)) <= 1e-12 * (1.0 + v)
                for t, v in zip(tgts, res["pc_history"])
            )
        check(f"end-to-end smoke ({mode}, rel_target={rel})", ok,
              f"stop={res.get('stop_reason')}, "
              f"equiv={res.get('grad_equiv_total'):.1f}, "
              f"targets ok, pruned {res['prune_report']['m_before']}->"
              f"{res['prune_report']['m_after']}")


def main() -> int:
    print(f"IPOPT available: {ipopt_available()}", flush=True)
    test_gram_equivalence()
    setup = build_small()
    test_stoch_full_batch(setup)
    test_degeneration(setup)
    test_prune()
    test_lambda_search(setup)
    test_smoke(setup)
    print("\n" + ("ALL PASS" if not FAILURES else f"FAILURES: {FAILURES}"),
          flush=True)
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
