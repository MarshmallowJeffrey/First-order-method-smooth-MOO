"""sanity_checks_bandit_toy.py — pre-flight checks for the bandit toy.

NEW FILE (July 26, 2026).  Run:
    KMP_DUPLICATE_LIB_OK=TRUE python sanity_checks_bandit_toy.py
Exit code 0 iff every check passes.
"""

from __future__ import annotations

import sys

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from objectives_bandit_toy import (BanditStochOracle, calibrate_L,
                                   make_bandit_toy)

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    p = make_bandit_toy(T=1000, noise_std=0.5, data_seed=7)
    rng = np.random.RandomState(0)
    d = p.d

    # 1. analytic gradient vs central finite differences ---------------
    h = 1e-7
    worst = 0.0
    for _ in range(20):
        th = rng.normal(0.0, 3.0, size=d)
        for k in range(2):
            g = p._gradF(th, k)
            g_fd = np.empty(d)
            for j in range(d):
                e = np.zeros(d)
                e[j] = h
                g_fd[j] = (p._F(th + e, k) - p._F(th - e, k)) / (2 * h)
            worst = max(worst, float(np.max(np.abs(g - g_fd))
                                     / (1.0 + np.max(np.abs(g)))))
    check("analytic gradient == finite differences", worst < 1e-6,
          f"worst rel err {worst:.2e}")

    # 2. fused joint oracle consistency --------------------------------
    ok = True
    for _ in range(10):
        th = rng.normal(0.0, 3.0, size=d)
        fv, gv = p.joint_oracle(th)
        for k in range(2):
            ok &= abs(fv[k] - p._F(th, k)) < 1e-12
            ok &= float(np.max(np.abs(gv[k] - p._gradF(th, k)))) < 1e-12
    check("fused joint oracle == per-component callables", ok)

    # 3. closed-form solution is stationary ----------------------------
    worst = 0.0
    for w in np.linspace(0.0, 1.0, 41):
        th = p.theta_star(w)
        g = w * p._gradF(th, 0) + (1.0 - w) * p._gradF(th, 1)
        worst = max(worst, float(np.linalg.norm(g)))
    check("closed-form theta_star(w) has zero gradient", worst < 1e-10,
          f"worst ||grad|| {worst:.2e}")

    # 4. closed-form value is the minimum ------------------------------
    ok = True
    for _ in range(200):
        w = rng.rand()
        th = rng.normal(0.0, 4.0, size=d)
        fw = w * p._F(th, 0) + (1.0 - w) * p._F(th, 1)
        ok &= fw >= p.scalarized_opt(w) - 1e-12
    check("F_w(theta_star) <= F_w(random theta)", ok)

    # 5. scipy cross-check of three scalarized optima ------------------
    from scipy.optimize import minimize
    worst = 0.0
    for w in (0.1, 0.5, 0.9):
        obj = lambda th: w * p._F(th, 0) + (1.0 - w) * p._F(th, 1)
        jac = lambda th: w * p._gradF(th, 0) + (1.0 - w) * p._gradF(th, 1)
        res = minimize(obj, np.zeros(d), jac=jac, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-16, "gtol": 1e-12})
        worst = max(worst, abs(res.fun - p.scalarized_opt(w))
                    / (1.0 + abs(res.fun)))
    check("scipy minimum == closed-form value", worst < 1e-8,
          f"worst rel gap {worst:.2e}")

    # 6. SVRG estimator unbiasedness -----------------------------------
    oracle = BanditStochOracle(p, batch_size=256, seed=41)
    th = rng.normal(0.0, 2.0, size=d)
    lam = np.array([0.3, 0.7])
    g_full = lam[0] * p._gradF(th, 0) + lam[1] * p._gradF(th, 1)
    acc = np.zeros(d)
    n_draws = 20000
    for _ in range(n_draws):
        acc += oracle._scalarized_grad(th, lam, oracle.sample_batch())
    err = float(np.linalg.norm(acc / n_draws - g_full)
                / (1.0 + np.linalg.norm(g_full)))
    check("minibatch estimator is unbiased", err < 5e-4,
          f"rel err of 20k-draw mean {err:.2e}")

    # 7. full-batch degeneration is exact ------------------------------
    g_fb = oracle._scalarized_grad(th, lam, oracle.full_batch())
    err = float(np.max(np.abs(g_fb - g_full)))
    check("full-batch estimator == full gradient", err < 1e-12,
          f"max abs err {err:.2e}")

    # 8. IFO accounting -------------------------------------------------
    oracle2 = BanditStochOracle(p, batch_size=256, seed=41)
    oracle2.set_anchor(np.zeros(d))
    for _ in range(7):
        oracle2.grad_pair(th, lam, oracle2.sample_batch())
    check("IFO count == 2 * steps * b_total",
          oracle2.ifo_count == 2 * 7 * oracle2.b_total,
          f"ifo {oracle2.ifo_count}, expected {2 * 7 * oracle2.b_total}")

    # 9. Eq. (9) speed matches the numerical PF derivative -------------
    worst = 0.0
    dw = 1e-6
    for w in np.linspace(0.05, 0.95, 19):
        f_plus = p.f_pf(w + dw)
        f_minus = p.f_pf(w - dw)
        v_num = float(np.linalg.norm((f_plus - f_minus) / (2 * dw)))
        v_ana = p.speed(w)
        worst = max(worst, abs(v_num - v_ana) / (1.0 + v_ana))
    check("SURF Eq. (9) speed == numerical ||d f_PF / dw||", worst < 1e-4,
          f"worst rel err {worst:.2e}")

    # ---- informational (not pass/fail) -------------------------------
    cal = calibrate_L(p)
    print(f"[info] reward estimation gap  ||R_hat - R||_inf = "
          f"{p.reward_sup_gap():.4f}")
    print(f"[info] CDF estimation gap     ||Phi_hat - Phi||_inf = "
          f"{p.cdf_sup_gap():.4f}")
    print(f"[info] calibrated L (safety {cal['safety']}): "
          f"L1 = {cal['L'][0]:.4f}, L2 = {cal['L'][1]:.4f} "
          f"(raw max {cal['L_raw_max'][0]:.4f}, {cal['L_raw_max'][1]:.4f}; "
          f"{cal['n_hessians']} Hessians)")

    n_pass = sum(RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} checks passed")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
