"""sanity_checks_k10_objective.py — gates for ``objectives_mnist_patch_k10``.

NEW FILE (Sep 10, 2026).  Checks, on a small instance (per_class = 300)
so it runs in under a minute on a laptop CPU:

  G1  mu = 0, cpu: the new factory reproduces the Aug-9 base factory
      ``objectives_mnist_patch.make_mnist_patch`` — same L, same
      per-class losses and Jacobian at random theta (per-class forward
      vs full forward + K backward: agreement to 1e-12 relative), same
      minibatch stream, same stochastic gradient pair on the same batch.
  G2  mu > 0: ridge algebra — f_mu = f + (mu/2)||theta||^2,
      J_mu = J + mu·theta, L_mu = L + mu, and the stochastic pair adds
      mu·theta_y / mu·theta_a exactly.
  G3  device (only when CUDA is present): cuda vs cpu losses/Jacobian
      agree to 1e-10 relative; the stochastic pair agrees to 1e-10; the
      same segment run twice on the device is bit-identical.
  G4  test-side helper runs and returns finite values.

Usage:
    python sanity_checks_k10_objective.py            # G1, G2, G4 (+G3 if CUDA)
    python sanity_checks_k10_objective.py --per-class 1000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import _layout  # noqa: F401
from baseline_svrg_certified_without_256_checkpoints import _support_batch  # noqa: E402
from objectives_mnist_patch import make_mnist_patch  # noqa: E402
from objectives_mnist_patch_k10 import (  # noqa: E402
    evaluate_test_stack,
    make_mnist_patch_k10,
    make_patch_initial_point,
)


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(1e-300, float(np.max(np.abs(b)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-probes", type=int, default=3)
    a = ap.parse_args()
    results = {}
    rng = np.random.RandomState(11)

    # ---- G1: mu = 0 on cpu reproduces the base factory ----------------
    t = time.time()
    base = make_mnist_patch(per_class=a.per_class, batch_size=a.batch, sampler_seed=41,
                            init_seed=8, n_probes=a.n_probes, probe_seed=7)
    new0 = make_mnist_patch_k10(per_class=a.per_class, mu=0.0, batch_size=a.batch,
                                sampler_seed=41, init_seed=8, n_probes=a.n_probes,
                                probe_seed=7, device="cpu")
    _, _, Lb, jb, sb, mb = base
    _, _, L0, j0, s0, m0 = new0
    d = mb["d"]
    x0 = make_patch_initial_point(8)
    thetas = [x0, x0 + 0.3 * rng.randn(d), x0 + 1.0 * rng.randn(d)]
    worst_f, worst_J = 0.0, 0.0
    for th in thetas:
        fb, Jb = jb(th)
        f0, J0 = j0(th)
        worst_f = max(worst_f, rel(f0, fb))
        worst_J = max(worst_J, rel(J0, Jb))
    batch_b, batch_0 = sb.sample_batch(), s0.sample_batch()
    same_stream = all(np.array_equal(u, v) for u, v in zip(batch_b, batch_0))
    lam = rng.dirichlet(np.ones(10))
    lam[3] = 0.0
    lam /= lam.sum()
    sb.set_anchor(thetas[1]); s0.set_anchor(thetas[1])
    bb = _support_batch(batch_b, lam)
    gyb, gab = sb.grad_pair(thetas[2], lam, bb)
    gy0, ga0 = s0.grad_pair(thetas[2], lam, bb)
    g1 = {"L_max_rel_diff": rel(L0, Lb), "f_max_rel_diff": worst_f, "J_max_rel_diff": worst_J,
          "same_batch_stream": same_stream, "gy_rel_diff": rel(gy0, gyb), "ga_rel_diff": rel(ga0, gab),
          "ifo_equal": sb.ifo_count == s0.ifo_count, "seconds": time.time() - t}
    g1["pass"] = bool(g1["L_max_rel_diff"] <= 1e-12 and worst_f <= 1e-12 and worst_J <= 1e-12
                      and same_stream and g1["gy_rel_diff"] <= 1e-12 and g1["ga_rel_diff"] <= 1e-12
                      and g1["ifo_equal"])
    results["G1_base_equivalence_mu0"] = g1
    print(f"[G1] {'PASS' if g1['pass'] else 'FAIL'} {g1}", flush=True)

    # ---- G2: ridge algebra -------------------------------------------
    t = time.time()
    mu = 1e-4
    newmu = make_mnist_patch_k10(per_class=a.per_class, mu=mu, batch_size=a.batch,
                                 sampler_seed=41, init_seed=8, n_probes=a.n_probes,
                                 probe_seed=7, device="cpu")
    _, _, Lm, jm, sm, mm = newmu
    th = thetas[2]
    f0, J0 = j0(th)
    fm, Jm = jm(th)
    e_f = rel(fm, f0 + 0.5 * mu * float(th @ th))
    e_J = rel(Jm, J0 + mu * th[None, :])
    e_L = rel(Lm, L0 + mu)
    sm.set_anchor(thetas[1])
    _ = sm.sample_batch()                       # advance the stream like s0 did
    gym, gam = sm.grad_pair(thetas[2], lam, bb)
    e_gy = rel(gym, gy0 + mu * thetas[2])
    e_ga = rel(gam, ga0 + mu * thetas[1])
    g2 = {"f_rel_err": e_f, "J_rel_err": e_J, "L_rel_err": e_L, "gy_rel_err": e_gy,
          "ga_rel_err": e_ga, "seconds": time.time() - t}
    g2["pass"] = bool(max(e_f, e_J, e_L, e_gy, e_ga) <= 1e-12)
    results["G2_ridge_algebra"] = g2
    print(f"[G2] {'PASS' if g2['pass'] else 'FAIL'} {g2}", flush=True)

    # ---- G3: cuda vs cpu (only if available) ---------------------------
    if torch.cuda.is_available():
        t = time.time()
        newg = make_mnist_patch_k10(per_class=a.per_class, mu=mu, batch_size=a.batch,
                                    sampler_seed=41, init_seed=8, n_probes=a.n_probes,
                                    probe_seed=7, device="cuda")
        _, _, Lg, jg, sg, mg = newg
        fg, Jg = jg(th)
        sg.set_anchor(thetas[1]); _ = sg.sample_batch()
        gyg, gag = sg.grad_pair(thetas[2], lam, bb)
        # same segment twice on the device: bit-identical?
        def seg(oracle):
            oracle.rng = np.random.RandomState(123)
            oracle.set_anchor(thetas[1]); y = thetas[1].copy()
            for _ in range(5):
                b = _support_batch(oracle.sample_batch(), lam)
                gy, ga = oracle.grad_pair(y, lam, b)
                y = y - 1e-3 * (gy - ga)
            return y
        y1, y2 = seg(sg), seg(sg)
        g3 = {"device": mg["device_description"], "f_rel_diff": rel(fg, fm), "J_rel_diff": rel(Jg, Jm),
              "L_rel_diff": rel(Lg, Lm), "gy_rel_diff": rel(gyg, gym), "ga_rel_diff": rel(gag, gam),
              "segment_bit_identical": bool(np.array_equal(y1, y2)), "seconds": time.time() - t}
        g3["pass"] = bool(max(g3["f_rel_diff"], g3["J_rel_diff"], g3["gy_rel_diff"], g3["ga_rel_diff"]) <= 1e-10
                          and g3["segment_bit_identical"])
        results["G3_cuda_vs_cpu"] = g3
        print(f"[G3] {'PASS' if g3['pass'] else 'FAIL'} {g3}", flush=True)
    else:
        results["G3_cuda_vs_cpu"] = {"skipped": "no CUDA device on this machine"}
        print("[G3] skipped (no CUDA)", flush=True)

    # ---- G4: test helper -----------------------------------------------
    t = time.time()
    ce, err = evaluate_test_stack([x0, thetas[1]], device="cpu")
    g4 = {"shape": list(ce.shape), "finite": bool(np.isfinite(ce).all() and np.isfinite(err).all()),
          "ce_x0_mean": float(ce[0].mean()), "seconds": time.time() - t}
    g4["pass"] = bool(g4["finite"] and ce.shape == (2, 10))
    results["G4_test_helper"] = g4
    print(f"[G4] {'PASS' if g4['pass'] else 'FAIL'} {g4}", flush=True)

    ok = all(v.get("pass", True) for v in results.values())
    print(f"[k10-objective gates] {'ALL PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
