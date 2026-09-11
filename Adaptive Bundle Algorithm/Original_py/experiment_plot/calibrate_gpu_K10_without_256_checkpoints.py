"""calibrate_gpu_K10_without_256_checkpoints.py — two-minute calibration of
a (GPU) machine before the K = 10 campaign.  Writes
output/CCP/K10_mnist_without_256_checkpoints/calibration_K10.json with
``"pass": true/false``; ``run_dots_K10_without_256_checkpoints.py`` refuses
to train on CUDA without a passed calibration.

Checks (full-size instance, per_class = 5,421, n = 54,210; ~1-3 min):

  C1  numerics: per-class losses and the 10 x d Jacobian at a random
      theta agree between the device and the CPU (relative 1e-10), and
      so does one stochastic gradient pair on the same minibatch;
  C2  determinism: one full SVRG segment (53 minibatch steps + joint
      gradient) run twice on the device from the same state and batch
      stream is bit-identical (if not: rerun with
      ``--deterministic``, which switches on torch's deterministic
      algorithms, and report which mode passed);
  C3  timing: joint gradient, full-support segment, single-class
      segment on the device (and on the CPU for reference); projected
      campaign wall clock at the requested budget;
  C4  one CCP decision on a synthetic 5,000-point K = 10 Gram stack with
      the block-transport solver (CPU side of the campaign).

Usage:
    python calibrate_gpu_K10_without_256_checkpoints.py --device cuda
    python calibrate_gpu_K10_without_256_checkpoints.py --device cpu     # laptop dry run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import _layout  # noqa: F401
from baseline_svrg_certified_without_256_checkpoints import _support_batch  # noqa: E402
from ccp_lambda_solver import CCPConfig  # noqa: E402
from ccp_lambda_solver_bulk import CCPLambdaSolverBulk  # noqa: E402
from objectives_mnist_patch_k10 import (  # noqa: E402
    BALANCED_MAX_PER_CLASS,
    device_description,
    make_mnist_patch_k10,
    make_patch_initial_point,
    resolve_device,
)
from stepper_core import make_stepper  # noqa: E402

K = 10
HERE = Path(__file__).resolve().parent
K10_ROOT = HERE.parent.parent / "output" / "CCP" / "K10_mnist_without_256_checkpoints"
STEPPER_CFG = {"msvrg_step_const": 0.1, "msvrg_momentum": 0.5, "adam_alpha": 1e-3, "adam_beta2": 0.9}
# segment counts of the B = 500,000 campaign (support-aware costs; Sep 10 design)
SEGMENTS_500K = {"adaptive": 21_700, 3: 33_300, 4: 30_900, 5: 29_200, 6: 27_800, 7: 26_700}


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(1e-300, float(np.max(np.abs(b)))))


def _segment(stoch, joint, stepper, x, lam, epoch_len, g_full, rng_seed):
    stoch.rng = np.random.RandomState(rng_seed)
    stepper.on_lambda_change(lam, 1.0, 1.0)
    stepper.start_segment(x, g_full, 1.0, 1.0, epoch_len)
    stoch.set_anchor(x)
    y = x.copy()
    t = time.time()
    for _ in range(epoch_len):
        batch = _support_batch(stoch.sample_batch(), lam)
        gy, ga = stoch.grad_pair(y, lam, batch)
        y = stepper.step(y, gy - ga + g_full)
    t_mb = time.time() - t
    t = time.time()
    f, J = joint(y)
    t_j = time.time() - t
    return y, f, J, t_mb, t_j


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--per-class", type=int, default=BALANCED_MAX_PER_CLASS)
    ap.add_argument("--budget", type=float, default=500_000.0)
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(True) before the checks")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    dev = resolve_device(a.device)
    report = {"device": device_description(dev), "torch": torch.__version__,
              "per_class": a.per_class, "deterministic_flag": a.deterministic,
              "checks": {}}
    print(f"[calib] device = {report['device']}", flush=True)

    t = time.time()
    _, _, L_d, joint_d, stoch_d, meta = make_mnist_patch_k10(
        per_class=a.per_class, mu=1e-4, batch_size=1024, sampler_seed=41, init_seed=8,
        n_probes=2, probe_seed=7, device=str(dev))
    n, d = meta["n"], meta["d"]
    epoch_len = int(np.ceil(n / 1024))
    print(f"[calib] instance built in {time.time() - t:.1f}s (n={n}, d={d}, epoch_len={epoch_len})",
          flush=True)
    rng = np.random.RandomState(5)
    x0 = make_patch_initial_point(8)
    th = x0 + 0.3 * rng.randn(d)
    lam_full = np.ones(K) / K
    lam_vertex = np.eye(K)[0]

    # ---- C1 numerics vs CPU ---------------------------------------------
    c1 = {"skipped": "device is the cpu"}
    if dev.type != "cpu":
        t = time.time()
        _, _, L_c, joint_c, stoch_c, _ = make_mnist_patch_k10(
            per_class=a.per_class, mu=1e-4, batch_size=1024, sampler_seed=41, init_seed=8,
            n_probes=2, probe_seed=7, device="cpu")
        f_d, J_d = joint_d(th)
        f_c, J_c = joint_c(th)
        stoch_d.rng = np.random.RandomState(9); stoch_c.rng = np.random.RandomState(9)
        b = _support_batch(stoch_d.sample_batch(), lam_full); stoch_c.sample_batch()
        stoch_d.set_anchor(x0); stoch_c.set_anchor(x0)
        gy_d, ga_d = stoch_d.grad_pair(th, lam_full, b)
        gy_c, ga_c = stoch_c.grad_pair(th, lam_full, b)
        c1 = {"f_rel_diff": rel(f_d, f_c), "J_rel_diff": rel(J_d, J_c), "L_rel_diff": rel(L_d, L_c),
              "gy_rel_diff": rel(gy_d, gy_c), "ga_rel_diff": rel(ga_d, ga_c), "seconds": time.time() - t}
        c1["pass"] = bool(max(c1["f_rel_diff"], c1["J_rel_diff"], c1["gy_rel_diff"], c1["ga_rel_diff"]) <= 1e-10)
        print(f"[C1] {'PASS' if c1['pass'] else 'FAIL'} {c1}", flush=True)
    report["checks"]["C1_numerics_vs_cpu"] = c1

    # ---- C2 determinism + C3 timing on the device ------------------------
    stepper = make_stepper("adam", d, dict(STEPPER_CFG))
    f_a, J_a = joint_d(x0)
    g_full = J_a.T @ lam_full
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time(); joint_d(x0); t_joint = time.time() - t
    if dev.type == "cuda":
        torch.cuda.synchronize()
    y1, f1, J1, t_mb1, t_j1 = _segment(stoch_d, joint_d, stepper, x0, lam_full, epoch_len, g_full, 77)
    y2, f2, J2, t_mb2, t_j2 = _segment(stoch_d, joint_d, stepper, x0, lam_full, epoch_len, g_full, 77)
    c2 = {"theta_bit_identical": bool(np.array_equal(y1, y2)),
          "f_bit_identical": bool(np.array_equal(f1, f2)), "J_bit_identical": bool(np.array_equal(J1, J2)),
          "max_abs_theta_diff": float(np.max(np.abs(y1 - y2)))}
    c2["pass"] = bool(c2["theta_bit_identical"] and c2["J_bit_identical"])
    print(f"[C2] {'PASS' if c2['pass'] else 'FAIL'} {c2}", flush=True)
    report["checks"]["C2_determinism"] = c2

    g_full_v = J_a.T @ lam_vertex
    _, _, _, t_mbv, t_jv = _segment(stoch_d, joint_d, stepper, x0, lam_vertex, epoch_len, g_full_v, 78)
    seg_full = t_mb2 + t_j2
    seg_vertex = t_mbv + t_jv
    units_full = K + epoch_len * 2.0 * stoch_d.b_total * K / n
    units_vertex = K + epoch_len * 2.0 * float(stoch_d.b_k[0]) * K / n
    proj = {}
    seg_mixed = 0.5 * (seg_full + seg_vertex)       # CCP visits mixed supports
    proj["adaptive_train_min"] = SEGMENTS_500K["adaptive"] * seg_mixed / 60
    for r in (3, 4, 5, 6, 7):
        proj[f"uniform_r{r}_train_min"] = SEGMENTS_500K[r] * (0.75 * seg_vertex + 0.25 * seg_full) / 60
    proj["all_legs_train_min"] = float(sum(proj.values()))
    scale = a.budget / 500_000.0
    proj = {k: float(v * scale) for k, v in proj.items()}
    c3 = {"joint_gradient_s": t_joint, "segment_full_support_s": seg_full,
          "segment_single_class_s": seg_vertex, "ms_per_unit_full": 1e3 * seg_full / units_full,
          "ms_per_unit_vertex": 1e3 * seg_vertex / units_vertex, "epoch_len": epoch_len,
          "projection_B": a.budget, "projected_minutes": proj}
    print(f"[C3] joint {t_joint:.2f}s | segment full {seg_full:.2f}s ({c3['ms_per_unit_full']:.1f} ms/unit) "
          f"| single-class {seg_vertex:.2f}s ({c3['ms_per_unit_vertex']:.1f} ms/unit) | projected training "
          f"for all legs at B={a.budget:g}: {proj['all_legs_train_min']:.0f} min", flush=True)
    report["checks"]["C3_timing"] = c3

    # ---- C4 one CCP decision on a synthetic stack --------------------------
    t = time.time()
    rg = np.random.default_rng(0)
    Jr = rg.standard_normal((5000, K, 40)) * 0.05
    Q = Jr @ Jr.transpose(0, 2, 1)
    solver = CCPLambdaSolverBulk(K, CCPConfig(N0=2000, r=10, seed=0, seed_sampler="exp",
                                              adaptive_seed_schedule=False))
    v, lam = solver.solve(Q[:4995]); t1 = time.time() - t
    t = time.time(); v, lam = solver.solve(Q); t2 = time.time() - t
    c4 = {"decision_cold_s": t1, "decision_warm_s": t2, "backend": solver.stats_last["backend"],
          "m": 5000}
    c4["pass"] = solver.stats_last["backend"] == "highspy-bulk"
    print(f"[C4] {'PASS' if c4['pass'] else 'FAIL (highspy missing: scipy fallback would be very slow)'} {c4}",
          flush=True)
    report["checks"]["C4_ccp_decision"] = c4

    ok = all(v.get("pass", True) for v in report["checks"].values())
    report["pass"] = bool(ok)
    report["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(a.out) if a.out else K10_ROOT / "calibration_K10.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[calib] {'PASS' if ok else 'FAIL'} -> {out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
