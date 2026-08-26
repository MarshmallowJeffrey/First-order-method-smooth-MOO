"""sanity_checks_ccp.py — correctness checks for ccp_lambda_solver.py.

NEW FILE (Aug 9, 2026).  Companion of Note/Aug_8_note.md §5.  Run:

    python sanity_checks_ccp.py

Checks
------
 1. simplex samplers: feasibility, determinism, coverage smoke
 2. phi_batch == per-point loop
 3. exact_gns_K2 == dense-grid envelope maximum
 4. game LP: highspy path == scipy fallback; warm == fresh
 5. CCP polish: monotone ascent; Lemma-2 tie invariant (informational)
 6. K = 2: multistart CCP == exact envelope value
 7. CCP vs _maximise_GN_fast (IPOPT strict tier) on real bundles
 8. adaptive N_new schedule state machine
 9. cross-round pool carry, cap bookkeeping, GNS monotone in the bundle
10. sandwich closure: m = 1 must close; K = 1 trivial path
"""

from __future__ import annotations

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from ccp_lambda_solver import (
    CCPConfig, CCPLambdaSolver, _GameLP, exact_gns_K2, highspy_available,
    phi_batch, sample_simplex_exp, sample_simplex_sobol, _phi_terms,
)
from scipy.stats import qmc

PASS = "PASS"


def _random_Q(rng, m, K, d=None, cancel=False, noise=0.05):
    """Random Gram stack.  cancel=True plants an exact cancelling weight
    per point (front-like late-stage geometry, hardest multistart case)."""
    d = d or 8 * K
    J = rng.normal(size=(m, K, d))
    if cancel:
        W = sample_simplex_exp(m, K, rng)              # w_i on the simplex
        # subtract the rank-one piece so w_i^T J_i = 0, then re-noise
        J = J - np.einsum('ik,id->ikd', np.ones((m, K)),
                          np.einsum('ik,ikd->id', W, J))
        J = J + noise * rng.normal(size=J.shape)
    return np.einsum('ikd,ild->ikl', J, J)


def check_samplers():
    rng = np.random.default_rng(0)
    for K in (2, 3, 6):
        P = sample_simplex_exp(500, K, rng)
        assert P.shape == (500, K)
        assert np.all(P >= 0) and np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
        assert np.allclose(P.mean(axis=0), 1.0 / K, atol=0.05)
        eng = qmc.Sobol(d=K - 1, scramble=True, seed=7)
        S = sample_simplex_sobol(512, K, eng)
        assert S.shape == (512, K)
        assert np.all(S >= 0) and np.allclose(S.sum(axis=1), 1.0, atol=1e-12)
        assert np.allclose(S.mean(axis=0), 1.0 / K, atol=0.05)
        eng2 = qmc.Sobol(d=K - 1, scramble=True, seed=7)
        S2 = sample_simplex_sobol(512, K, eng2)
        assert np.array_equal(S, S2), "Sobol draws must be seed-deterministic"
    print(f"{PASS} 1  samplers feasible, deterministic, centred")


def check_phi_batch():
    rng = np.random.default_rng(1)
    Q = _random_Q(rng, 20, 5)
    lams = sample_simplex_exp(50, 5, rng)
    phis, idx = phi_batch(Q, lams)
    for n in range(50):
        ref = np.array([lams[n] @ Q[i] @ lams[n] for i in range(20)])
        assert abs(phis[n] - ref.min()) <= 1e-10 * max(1.0, ref.min())
        assert abs(ref[idx[n]] - ref.min()) <= 1e-10 * max(1.0, ref.min())
    print(f"{PASS} 2  phi_batch matches the per-point loop")


def check_exact_K2():
    rng = np.random.default_rng(2)
    for trial in range(10):
        Q = _random_Q(rng, 25, 2, cancel=(trial % 2 == 0))
        gns, lam = exact_gns_K2(Q)
        s = np.linspace(0.0, 1.0, 200_001)
        L = np.c_[1.0 - s, s]
        grid_vals, _ = phi_batch(Q, L)
        gmax = float(grid_vals.max())
        scale = max(1.0, abs(gmax))
        # the grid under-shoots a kink maximiser by up to slope * h/2
        # (the envelope max sits at a crossing, so the error is FIRST
        # order in the grid step, not second)
        a, b, c = Q[:, 0, 0], Q[:, 0, 1], Q[:, 1, 1]
        alpha, beta = a - 2.0 * b + c, 2.0 * (b - a)
        slope_bound = float(np.max(2.0 * np.abs(alpha) + np.abs(beta)))
        h_grid = s[1] - s[0]
        assert gns >= gmax - 1e-9 * scale, (gns, gmax)
        assert gns <= gmax + slope_bound * h_grid + 1e-9 * scale, (gns, gmax)
        phis_at = _phi_terms(Q, lam)[1]
        assert abs(float(np.min(phis_at)) - gns) <= 1e-9 * scale
    print(f"{PASS} 3  exact_gns_K2 matches the dense-grid envelope max")


def check_game_lp():
    rng = np.random.default_rng(3)
    for m, K in ((30, 3), (100, 6)):
        M1 = rng.normal(size=(m, K)) + 1.0
        M2 = M1 + 0.05 * rng.normal(size=(m, K))
        cold = _GameLP(K, use_highspy=False)
        t1_s, _ = cold.resolve(M1)
        t2_s, _ = cold.resolve(M2)
        if highspy_available():
            warm = _GameLP(K, use_highspy=True)
            t1_h, lam1 = warm.resolve(M1)
            t2_h, lam2 = warm.resolve(M2)          # warm re-solve
            assert abs(t1_h - t1_s) <= 1e-9 * max(1.0, abs(t1_s))
            assert abs(t2_h - t2_s) <= 1e-9 * max(1.0, abs(t2_s))
            fresh = _GameLP(K, use_highspy=True)
            t2_f, _ = fresh.resolve(M2)
            assert abs(t2_h - t2_f) <= 1e-9 * max(1.0, abs(t2_f))
            # consecutive growths (bundle m -> m+2 -> m+3) must keep the
            # model consistent — regression test for the Aug-9 row-layout
            # bug (append after the equality row corrupted later rewrites)
            M3 = np.vstack([M2, rng.normal(size=(2, K)) + 1.0])
            t3_h, _ = warm.resolve(M3)
            t3_s, _ = cold.resolve(M3)
            assert abs(t3_h - t3_s) <= 1e-9 * max(1.0, abs(t3_s))
            M4 = np.vstack([M3, rng.normal(size=(1, K)) + 1.0])
            t4_h, _ = warm.resolve(M4)
            t4_s, _ = cold.resolve(M4)
            assert abs(t4_h - t4_s) <= 1e-9 * max(1.0, abs(t4_s))
            M4b = M4 + 0.01 * rng.normal(size=M4.shape)   # same-shape rewrite
            t4b_h, _ = warm.resolve(M4b)
            t4b_s, _ = cold.resolve(M4b)
            assert abs(t4b_h - t4b_s) <= 1e-9 * max(1.0, abs(t4b_s))
    tag = "highspy+scipy" if highspy_available() else "scipy only"
    print(f"{PASS} 4  game LP paths agree ({tag})")


def check_monotone_and_ties():
    rng = np.random.default_rng(4)
    solver = CCPLambdaSolver(4, CCPConfig(N0=64, r=4, seed=11))
    single_active_interior = 0
    total = 0
    for trial in range(8):
        Q = _random_Q(rng, 40, 4, cancel=(trial % 2 == 0))
        for rep in range(4):
            lam0 = sample_simplex_exp(1, 4, rng)[0]
            trace: list = []
            lam, phi, phis_at, iters, delta = solver._polish(
                Q, lam0, epsilon=None, trace=trace)
            tr = np.asarray(trace + [phi])
            assert np.all(np.diff(tr) >= -1e-9 * np.maximum(1.0, np.abs(tr[:-1]))), \
                "CCP ascent must be monotone"
            active = np.nonzero(
                phis_at <= phi + 1e-7 * max(1.0, abs(phi)))[0]
            total += 1
            if len(active) < 2 and lam.max() < 1.0 - 1e-6:
                single_active_interior += 1
    print(f"{PASS} 5  CCP ascent monotone on {total} runs "
          f"(interior single-active terminations: {single_active_interior} "
          f"— expect ~0, Lemma 2)")


def check_K2_vs_exact():
    rng = np.random.default_rng(5)
    worst = 0.0
    for trial in range(15):
        m = 5 if trial % 3 == 0 else 40
        Q = _random_Q(rng, m, 2, cancel=(trial % 2 == 0))
        ref, _ = exact_gns_K2(Q)
        solver = CCPLambdaSolver(
            2, CCPConfig(N0=256, r=8, seed=trial, seed_sampler="exp"))
        val, lam = solver.solve(Q)
        rel = (ref - val) / max(1.0, abs(ref))
        worst = max(worst, rel)
        assert rel <= 1e-7, (trial, ref, val)
        assert val <= ref + 1e-9 * max(1.0, abs(ref)), "CCP must lower-bound GNS"
    print(f"{PASS} 6  K=2 multistart CCP hits the exact GNS "
          f"(worst rel shortfall {worst:.2e})")


def _quadratic_bundle(rng, K, d, m):
    """A real BundleFast built from random convex quadratic objectives."""
    from bundle_fast import BundleFast
    mats = []
    for _ in range(K):
        C = rng.normal(size=(d, d)) / np.sqrt(d)
        mats.append(C @ C.T + 0.1 * np.eye(d))
    bvecs = [rng.normal(size=d) for _ in range(K)]
    objectives = [
        (lambda x, A=A, b=b: 0.5 * float(x @ A @ x) + float(b @ x))
        for A, b in zip(mats, bvecs)]
    grads = [(lambda x, A=A, b=b: A @ x + b) for A, b in zip(mats, bvecs)]
    L = np.array([np.linalg.eigvalsh(A)[-1] for A in mats])
    bundle = BundleFast(K=K, d=d, L=L)
    bundle.add_point(rng.normal(size=d), objectives, grads)
    for _ in range(m - 1):
        bundle.add_point(rng.normal(size=d), objectives, grads)
    return bundle


def check_vs_ipopt_fast():
    from algorithm_fast_without_256_checkpoints import _maximise_GN_fast
    rng = np.random.default_rng(6)
    diffs = []
    for trial in range(6):
        K = 3 if trial % 2 == 0 else 5
        bundle = _quadratic_bundle(rng, K, d=12, m=8 + 4 * trial)
        ref_val, _ = _maximise_GN_fast(bundle, prev_lam=None, tier="strict")
        solver = CCPLambdaSolver(
            K, CCPConfig(N0=512, r=10, seed=trial))
        val, lam = solver.solve(bundle.gram_stack())
        scale = max(1.0, abs(ref_val))
        diffs.append((val - ref_val) / scale)
        assert val >= ref_val - 1e-6 * scale, \
            f"CCP lost to IPOPT beyond tolerance: {val} vs {ref_val}"
    better = sum(1 for dd in diffs if dd > 1e-9)
    print(f"{PASS} 7  CCP >= IPOPT-strict on all bundles "
          f"(CCP strictly better on {better}/6; max rel gain "
          f"{max(diffs):.2e})")


def check_schedule():
    cfg = CCPConfig(N0=1000, r=10, adaptive_seed_schedule=True)
    s = CCPLambdaSolver(3, cfg)
    assert s.n_new == 1000
    s._update_seed_schedule(0.0, False)
    assert s.n_new == 1000                     # first zero: streak only
    s._update_seed_schedule(0.0, False)
    assert s.n_new == 500                      # two consecutive zeros: halve
    s._update_seed_schedule(0.0, False)
    assert s.n_new == 500                      # streak restarted after halving
    s._update_seed_schedule(0.1, False)
    assert s.n_new == 500                      # low band: hold
    s._update_seed_schedule(0.5, False)
    assert s.n_new == 1000                     # expansion doubles, capped at N0
    s._update_seed_schedule(0.5, False)
    assert s.n_new == 1000
    s.n_new = 150
    s._update_seed_schedule(0.0, False)
    s._update_seed_schedule(0.0, False)
    assert s.n_new == 100                      # floor 10 r
    s._update_seed_schedule(0.0, True)
    assert s.n_new == 1000                     # collapse resets to N0
    print(f"{PASS} 8  adaptive N_new schedule follows the rho rule")


def check_pool_and_monotone_bundle():
    rng = np.random.default_rng(7)
    K = 4
    Q1 = _random_Q(rng, 30, K, cancel=True)
    solver = CCPLambdaSolver(K, CCPConfig(N0=256, r=6, seed=3))
    v1, lam1 = solver.solve(Q1, epsilon=1e-3)
    st1 = solver.stats_last
    for key in ("pool_size", "pool_cap", "n_distinct_before_cap",
                "n_dropped_by_cap", "rho", "n_new_used", "ccp_iters",
                "lambda_search_wall_time", "sandwich_closed"):
        assert key in st1, key
    assert 1 <= st1["pool_size"] <= st1["pool_cap"]
    # grow the bundle by one point: GNS is nonincreasing, pool is reused
    Q2 = np.concatenate([Q1, _random_Q(rng, 1, K)], axis=0)
    v2, lam2 = solver.solve(Q2, epsilon=1e-3)
    st2 = solver.stats_last
    assert v2 <= v1 + 1e-9 * max(1.0, abs(v1)), \
        "adding a bundle point must not increase GNS"
    assert st2["m"] == 31 and st2["round"] == 1   # round index is 0-based
    print(f"{PASS} 9  pool carries across rounds; GNS monotone in the "
          f"bundle ({v1:.4e} -> {v2:.4e}); cap bookkeeping present")


def check_sandwich_and_K1():
    rng = np.random.default_rng(8)
    # m = 1: max of a single convex quadratic over the simplex sits at a
    # vertex, so the sandwich must close and return exactly max_k Q[0,k,k]
    Q = _random_Q(rng, 1, 5)
    solver = CCPLambdaSolver(5, CCPConfig(seed=1))
    val, lam = solver.solve(Q)
    assert solver.stats_last["sandwich_closed"]
    k = int(np.argmax(np.diag(Q[0])))
    assert abs(val - Q[0, k, k]) <= 1e-9 * max(1.0, Q[0, k, k])
    assert lam[k] == 1.0
    sK1 = CCPLambdaSolver(1, CCPConfig())
    v1, l1 = sK1.solve(_random_Q(rng, 7, 1))
    assert l1.shape == (1,) and l1[0] == 1.0
    print(f"{PASS} 10 sandwich closes at m=1 (vertex exact); K=1 trivial path")


if __name__ == "__main__":
    check_samplers()
    check_phi_batch()
    check_exact_K2()
    check_game_lp()
    check_monotone_and_ties()
    check_K2_vs_exact()
    check_vs_ipopt_fast()
    check_schedule()
    check_pool_and_monotone_bundle()
    check_sandwich_and_K1()
    print("\nAll ccp_lambda_solver sanity checks passed.")
