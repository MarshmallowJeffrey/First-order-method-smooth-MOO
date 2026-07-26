"""sanity_checks_bandit_toy_K5.py — pre-flight checks for the K=5 bandit.

NEW FILE (July 26, 2026).  Run:
    KMP_DUPLICATE_LIB_OK=TRUE python sanity_checks_bandit_toy_K5.py
Exit code 0 iff every check passes.
"""

from __future__ import annotations

import sys

import numpy as np

from objectives_bandit_toy import (BanditStochOracle, calibrate_L,
                                   make_bandit_toy_K)

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    p = make_bandit_toy_K(K=5, T=1000, noise_std=0.5, data_seed=7)
    rng = np.random.RandomState(0)
    d, K = p.d, p.K

    # 0. reward-design properties ---------------------------------------
    diag = np.diag(p.R_true)
    argmaxes = np.argmax(p.R_true, axis=1)
    check("R design: diagonal 1, distinct argmax arms per objective",
          bool(np.allclose(diag, 1.0)
               and len(set(argmaxes.tolist())) == K),
          f"argmax arms {argmaxes.tolist()}")

    # 1. analytic gradient vs central finite differences ---------------
    h = 1e-7
    worst = 0.0
    for _ in range(20):
        th = rng.normal(0.0, 3.0, size=d)
        for k in range(K):
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
        for k in range(K):
            ok &= abs(fv[k] - p._F(th, k)) < 1e-12
            ok &= float(np.max(np.abs(gv[k] - p._gradF(th, k)))) < 1e-12
    check("fused joint oracle == per-component callables", ok)

    # 3. closed-form solution is stationary at random lambdas ----------
    worst = 0.0
    lams = list(np.eye(K)) + [np.full(K, 1.0 / K)] + \
        list(rng.dirichlet(np.ones(K), size=40))
    for lam in lams:
        th = p.theta_star_lam(lam)
        g = sum(float(lam[k]) * p._gradF(th, k) for k in range(K))
        worst = max(worst, float(np.linalg.norm(g)))
    check("closed-form theta_star(lam) has zero gradient", worst < 1e-10,
          f"worst ||grad|| {worst:.2e} over {len(lams)} lambdas")

    # 4. closed-form value is the minimum (scipy cross-check) ----------
    # Two-sided test with the two halves separated: (a) NO scipy run may
    # ever land BELOW the closed form (global-minimum property); (b) a
    # warm-started run must MATCH it.  A cold start from theta0=0 is
    # deliberately not required to match: at vertex lambdas L-BFGS-B
    # stalls in the flat softmax region ~3e-5 ABOVE the optimum — the
    # same vertex-blindness phenomenon the experiment studies.
    from scipy.optimize import minimize
    worst_below = 0.0   # how far scipy ever gets BELOW the closed form
    worst_warm = 0.0
    for lam in (np.eye(K)[0], np.eye(K)[4], np.full(K, 0.2),
                np.array([0.4, 0.1, 0.1, 0.1, 0.3])):
        obj = lambda th: sum(float(lam[k]) * p._F(th, k) for k in range(K))
        jac = lambda th: sum(float(lam[k]) * p._gradF(th, k) for k in range(K))
        cf = p.scalarized_opt_lam(lam)
        for x_init in (np.zeros(d), p.theta_star_lam(lam) + 0.01):
            res = minimize(obj, x_init, jac=jac, method="L-BFGS-B",
                           options={"maxiter": 2000, "ftol": 1e-16,
                                    "gtol": 1e-12})
            worst_below = max(worst_below, cf - res.fun)
        res_w = minimize(obj, p.theta_star_lam(lam) + 0.01, jac=jac,
                         method="L-BFGS-B",
                         options={"maxiter": 2000, "ftol": 1e-16,
                                  "gtol": 1e-12})
        worst_warm = max(worst_warm, abs(res_w.fun - cf) / (1.0 + abs(cf)))
    check("closed form is the minimum (scipy never below; warm start "
          "matches)", worst_below < 1e-10 and worst_warm < 1e-8,
          f"max below {worst_below:.2e}, warm rel gap {worst_warm:.2e}")

    # 5. oracle_batch vectorisation == per-lambda loop -----------------
    lams = rng.dirichlet(np.ones(K), size=200)
    fvecs, fstar = p.oracle_batch(lams)
    worst = 0.0
    for i in range(0, 200, 17):
        worst = max(worst, float(np.max(np.abs(fvecs[i] - p.f_vec_lam(lams[i])))))
        worst = max(worst, abs(fstar[i] - p.scalarized_opt_lam(lams[i])))
    check("oracle_batch == per-lambda closed form", worst < 1e-12,
          f"worst abs err {worst:.2e}")

    # 6. SVRG estimator unbiasedness -----------------------------------
    oracle = BanditStochOracle(p, batch_size=256, seed=41)
    th = rng.normal(0.0, 2.0, size=d)
    lam = np.array([0.3, 0.2, 0.1, 0.15, 0.25])
    g_full = sum(float(lam[k]) * p._gradF(th, k) for k in range(K))
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

    # 8. IFO accounting and batch split --------------------------------
    oracle2 = BanditStochOracle(p, batch_size=256, seed=41)
    oracle2.set_anchor(np.zeros(d))
    for _ in range(7):
        oracle2.grad_pair(th, lam, oracle2.sample_batch())
    check("b_total == 256 and IFO == 2 * steps * b_total",
          oracle2.b_total == 256
          and oracle2.ifo_count == 2 * 7 * oracle2.b_total,
          f"b_k {oracle2.b_k.tolist()}, ifo {oracle2.ifo_count}")

    # ---- informational (not pass/fail) -------------------------------
    cal = calibrate_L(p)
    print(f"[info] reward estimation gap  ||R_hat - R||_inf = "
          f"{p.reward_sup_gap():.4f}")
    Ls = ", ".join(f"{v:.4f}" for v in cal["L"])
    Lr = ", ".join(f"{v:.4f}" for v in cal["L_raw_max"])
    print(f"[info] calibrated L (safety {cal['safety']}): [{Ls}] "
          f"(raw max [{Lr}]; {cal['n_hessians']} Hessians)")

    n_pass = sum(RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} checks passed")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
