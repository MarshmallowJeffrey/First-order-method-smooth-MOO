"""sanity_checks_bandit_toy_mv.py — pre-flight checks for the
mean-variance bandit toy (objectives_bandit_toy_mv).

NEW FILE (July 31, 2026).  Run:
    KMP_DUPLICATE_LIB_OK=TRUE python sanity_checks_bandit_toy_mv.py
Exit code 0 iff every check passes.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from objectives_bandit_toy import BanditStochOracle, make_bandit_toy
from objectives_bandit_toy_mv import (BanditStochOracleMV,
                                      BanditToyMVProblem, make_bandit_toy_mv)

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))


def main() -> int:
    GAMMA = 1.5
    p0 = make_bandit_toy(T=1000, noise_std=0.5, data_seed=7)
    pmv0 = make_bandit_toy_mv(gamma=0.0, T=1000, noise_std=0.5, data_seed=7)
    pmv = make_bandit_toy_mv(gamma=GAMMA, T=1000, noise_std=0.5, data_seed=7)
    rng = np.random.RandomState(0)
    d = pmv.d

    # 1. gamma = 0 degenerates to the July-26 objective BIT-FOR-BIT ----
    ok = True
    for _ in range(200):
        th = rng.normal(0.0, 4.0, size=d)
        for k in range(2):
            ok &= pmv0._F(th, k) == p0._F(th, k)
            ok &= bool(np.all(pmv0._gradF(th, k) == p0._gradF(th, k)))
        f_a, g_a = pmv0.joint_oracle(th)
        f_b, g_b = p0.joint_oracle(th)
        ok &= bool(np.all(f_a == f_b) and np.all(g_a == g_b))
    check("gamma=0 objective/gradients/joint == July-26 (bitwise)", ok)

    # 2. gamma = 0 stochastic oracle degenerates bitwise ---------------
    o_old = BanditStochOracle(p0, batch_size=256, seed=41)
    o_new = BanditStochOracleMV(pmv0, batch_size=256, seed=41)
    ok = True
    th_a = rng.normal(0.0, 2.0, size=d)
    o_old.set_anchor(th_a)
    o_new.set_anchor(th_a)
    for _ in range(50):
        lam_w = rng.rand()
        lam = np.array([lam_w, 1.0 - lam_w])
        th = rng.normal(0.0, 2.0, size=d)
        b_old = o_old.sample_batch()
        b_new = o_new.sample_batch()
        ok &= all(np.all(x == y) for x, y in zip(b_old, b_new))
        g1, a1 = o_old.grad_pair(th, lam, b_old)
        g2, a2 = o_new.grad_pair(th, lam, b_new)
        ok &= bool(np.all(g1 == g2) and np.all(a1 == a2))
    ok &= o_old.ifo_count == o_new.ifo_count
    check("gamma=0 stochastic oracle == July-26 oracle (bitwise)", ok)

    # 3. analytic gradient vs finite differences (gamma > 0) -----------
    h = 1e-7
    worst = 0.0
    for _ in range(30):
        th = rng.normal(0.0, 3.0, size=d)
        for k in range(2):
            g = pmv._gradF(th, k)
            g_fd = np.empty(d)
            for j in range(d):
                e = np.zeros(d)
                e[j] = h
                g_fd[j] = (pmv._F(th + e, k) - pmv._F(th - e, k)) / (2 * h)
            worst = max(worst, float(np.max(np.abs(g - g_fd))
                                     / (1.0 + np.max(np.abs(g)))))
    check("MV analytic gradient == finite differences", worst < 1e-6,
          f"worst rel err {worst:.2e}")

    # 4. fused joint oracle consistency (gamma > 0) --------------------
    ok = True
    for _ in range(10):
        th = rng.normal(0.0, 3.0, size=d)
        fv, gv = pmv.joint_oracle(th)
        for k in range(2):
            ok &= abs(fv[k] - pmv._F(th, k)) < 1e-12
            ok &= float(np.max(np.abs(gv[k] - pmv._gradF(th, k)))) < 1e-12
    check("MV fused joint oracle == per-component callables", ok)

    # 5. scalarized_value_grad consistency -----------------------------
    ok = True
    for _ in range(20):
        w = rng.rand()
        th = rng.normal(0.0, 3.0, size=d)
        v, g = pmv.scalarized_value_grad(th, w)
        v_ref = w * pmv._F(th, 0) + (1.0 - w) * pmv._F(th, 1)
        g_ref = w * pmv._gradF(th, 0) + (1.0 - w) * pmv._gradF(th, 1)
        ok &= abs(v - v_ref) < 1e-12
        ok &= float(np.max(np.abs(g - g_ref))) < 1e-12
    check("scalarized_value_grad == w*F1 + (1-w)*F2 (values+grads)", ok)

    # 6. full-batch stochastic gradient is EXACT; minibatch unbiased ---
    o = BanditStochOracleMV(pmv, batch_size=256, seed=41)
    worst = 0.0
    for _ in range(10):
        w = rng.rand()
        lam = np.array([w, 1.0 - w])
        th = rng.normal(0.0, 2.0, size=d)
        g_full = o._scalarized_grad(th, lam, o.full_batch())
        _, g_ref = pmv.scalarized_value_grad(th, w)
        worst = max(worst, float(np.max(np.abs(g_full - g_ref))))
    check("full-batch oracle gradient == analytic (exact)", worst < 1e-12,
          f"worst abs err {worst:.2e}")
    w = 0.37
    lam = np.array([w, 1.0 - w])
    th = rng.normal(0.0, 2.0, size=d)
    _, g_ref = pmv.scalarized_value_grad(th, w)
    acc = np.zeros(d)
    n_draw = 4000
    for _ in range(n_draw):
        acc += o._scalarized_grad(th, lam, o.sample_batch())
    rel = float(np.linalg.norm(acc / n_draw - g_ref)
                / (1.0 + np.linalg.norm(g_ref)))
    check("minibatch oracle unbiased (4000-draw mean)", rel < 1e-2,
          f"rel err {rel:.2e}")

    # 7. grad_pair IFO accounting --------------------------------------
    o2 = BanditStochOracleMV(pmv, batch_size=256, seed=41)
    o2.set_anchor(np.zeros(d))
    before = o2.ifo_count
    b = o2.sample_batch()
    o2.grad_pair(np.zeros(d), lam, b)
    added = o2.ifo_count - before
    rows = int(sum(len(x) for x in b))
    check("grad_pair IFO accounting += 2*rows", added == 2 * rows,
          f"added {added}, rows {rows}")

    # 8. reference solver at gamma=0 reproduces the closed form --------
    psm0 = make_bandit_toy_mv(gamma=0.0, T=1000, noise_std=0.5, data_seed=7,
                              ref_n_dense=501, ref_adam_steps=400,
                              ref_n_random_starts=8)
    psm0.ensure_reference(cache_path=None, verbose=False)
    worst_v, worst_f = 0.0, 0.0
    for i, w in enumerate(np.linspace(0.0, 1.0, 501)):
        worst_v = max(worst_v, abs(psm0.scalarized_opt(w)
                                   - p0.scalarized_opt(w)))
        worst_f = max(worst_f, float(np.max(np.abs(psm0.f_pf(w)
                                                   - p0.f_pf(w)))))
    check("gamma=0 reference == closed form (scalarized opt + front)",
          worst_v < 1e-8 and worst_f < 1e-6,
          f"worst |opt| {worst_v:.2e}, worst |f_pf| {worst_f:.2e}")

    # 9. reference stationarity at gamma > 0 ---------------------------
    psm = make_bandit_toy_mv(gamma=GAMMA, T=1000, noise_std=0.5, data_seed=7,
                             ref_n_dense=501, ref_adam_steps=600,
                             ref_n_random_starts=8)
    psm.ensure_reference(cache_path=None, verbose=False)
    worst = 0.0
    for w in np.linspace(0.0, 1.0, 101):
        th = psm.theta_star(w)
        _, g = psm.scalarized_value_grad(th, w)
        worst = max(worst, float(np.linalg.norm(g)))
    check("MV reference points are stationary (||grad|| <= 1e-8)",
          worst < 1e-8, f"worst ||grad|| {worst:.2e}")

    # 10. the designed nonconvexity is realised ------------------------
    prov = psm.ensure_reference(cache_path=None, verbose=False)["plugin"]
    check("nonconvexity realised (bimodal pool or front jump)",
          prov["bimodal_fraction"] > 0.0 or prov["max_front_jump"] > 0.02,
          f"bimodal {prov['bimodal_fraction']:.3f}, "
          f"jump {prov['max_front_jump']:.3e} @ w="
          f"{prov['front_jump_at_w']:.3f}")

    # 11. chordal arc CDF is a CDF -------------------------------------
    wg, cdf = psm.arc_cdf()
    ok = (abs(cdf[0]) < 1e-15 and abs(cdf[-1] - 1.0) < 1e-12
          and bool(np.all(np.diff(cdf) >= -1e-15)))
    check("chordal arc CDF monotone with endpoints 0/1", ok)

    # 12. cache round-trip ---------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        cp = os.path.join(td, "ref.npz")
        psm._cache_path = cp
        psm._save_cache()
        p_re = make_bandit_toy_mv(gamma=GAMMA, T=1000, noise_std=0.5,
                                  data_seed=7, ref_n_dense=501,
                                  ref_adam_steps=600, ref_n_random_starts=8)
        p_re.ensure_reference(cache_path=cp, verbose=False)
        ok = bool(np.all(p_re._ref["theta"] == psm._ref["theta"])
                  and np.all(p_re._ref["fscal"] == psm._ref["fscal"])
                  and np.all(p_re._ref_true["fvec"] == psm._ref_true["fvec"]))
    check("reference cache round-trip (bitwise)", ok)

    n_ok = sum(RESULTS)
    print(f"\n{'ALL PASS' if all(RESULTS) else 'FAILURES'} "
          f"({n_ok}/{len(RESULTS)})")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
