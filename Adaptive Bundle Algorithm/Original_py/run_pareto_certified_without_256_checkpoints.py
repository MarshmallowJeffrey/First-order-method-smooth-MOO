"""Certified-mode Pareto-front comparison at K=2 (without-256-checkpoints track).

What this does
--------------
Runs ONE baseline and TWO adaptive certified runs on the identical 96x96
problem instance, then draws the Pareto-front comparison figures:

    x-axis F1(x*_lambda), y-axis F2(x*_lambda), one curve per method,
    the points along a curve being the method's delivered point for each
    lambda = (t, 1-t) as t sweeps [0, 1].

Certification setup (user-specified):
  combo 1:  baseline eps 0.01  vs adaptive eps 0.01
  combo 2:  baseline eps 0.01  vs adaptive eps 0.001
The baseline run is shared by both combos (same eps).

How the baseline's node_tol is derived from its eps (K=2, r fixed)
------------------------------------------------------------------
eps bounds the SQUARED gradient norm, so work with norms: sqrt(0.01)=0.1.
The baseline only controls its grid NODES via node_tol; for a weight
lambda=(t,1-t) between nodes, the nearest node t_i is at most
h = 1/(2r) away in t, and for K=2 exactly

    grad F_lambda(x) = t*g1(x) + (1-t)*g2(x)
    d/dt grad F_lambda(x) = g1(x) - g2(x)
 => ||grad F_lambda(x_i)|| <= ||grad F_{lambda_i}(x_i)|| + h*||g1(x_i)-g2(x_i)||

So the level the baseline can certify over ALL lambda is
    ( sqrt(node_tol) + h * D )^2,   D = max_i ||g1(x_i) - g2(x_i)||.
D is not known before the run, so node_tol is chosen by an equal split of
the norm budget: sqrt(node_tol) = sqrt(eps)/2  =>  node_tol = eps/4,
leaving the other half (h*D <= sqrt(eps)/2, i.e. D <= r*sqrt(eps)) as the
grid's share.  After the run this file MEASURES D at the delivered nodes
and reports the exact certified level; if (h*D)^2 alone exceeds eps, no
node_tol can reach eps at this r — that is the grid floor stated in
certified-mode language.

Measurement track
-----------------
Runs use the *_without_256_checkpoints variants: checkpoints record each
method's own worst-case value (no external 256-start solves), which keeps
wall time down.  The certification stop rules are unaffected: the adaptive
method stops on its own lambda-search value (<= 2*eps/3), the baseline on
per-node entry checks (<= node_tol), budget 2,000 kept as the fuse.

Pareto extraction (no re-optimisation anywhere)
-----------------------------------------------
- Adaptive: bundle.points / bundle.fvals / bundle.grads are all stored by
  the run; for each t the delivered point is argmin_i ||t*g1_i+(1-t)*g2_i||^2
  (pure algebra on stored gradients).
- Baseline: final_solutions stores the points; ONE joint-oracle call per
  node (presentation work, after the clock) supplies (F1, F2) and the two
  gradients per node, reused for both the front and the certificate audit.
  The same argmin selection rule is applied over its node set, so both
  curves have identical semantics: "the point this method would hand you
  for weight lambda".

Usage:
    python run_pareto_certified_without_256_checkpoints.py [--smoke]

Outputs (originals untouched):
    output/pareto_certified_without_256_checkpoints[_smoke]/
      README.md, pareto_data.json,
      pareto_front_combo1_bl0.01_a0.01.png,
      pareto_front_combo2_bl0.01_a0.001.png,
      baseline_eps0.01/summary.json,
      adaptive_eps0.01/summary.json, adaptive_eps0.001/summary.json
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

# experiments must be imported before the algorithm modules: it loads
# objectives_torch (and torch's OpenMP runtime) before any cyipopt import.
from experiments import _make_mlp_problem, make_mlp_initial_point  # noqa: E402
from bundle import prefer_fused_joint_oracle, validate_oracle_output  # noqa: E402
from baseline_without_256_checkpoints import uniform_discretisation  # noqa: E402
from algorithm_without_256_checkpoints import algorithm_adaptive  # noqa: E402
from run_experiments import _result_curves, _json_ready  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent / "output"

EPS_BASELINE = 0.01
EPS_ADAPTIVE = (0.01, 0.001)
NODE_TOL = EPS_BASELINE / 4.0  # equal norm-budget split; see module docstring.
N_LAMBDA_SWEEP = 1001


def problem_config(smoke: bool) -> dict:
    if smoke:
        return dict(K=2, p=6, n=200, hidden_sizes=[8], activation="tanh",
                    coarse_resolution=10, n_passes=100_000,
                    steps_per_point_per_pass=5, max_grad_evals=600,
                    eval_every_n_grads=60, max_outer=1_000_000, max_inner=5,
                    lambda_max_starts=8, prune_inner=True)
    return dict(K=2, p=20, n=50_000, hidden_sizes=[96, 96], activation="tanh",
                coarse_resolution=10, n_passes=100_000,
                steps_per_point_per_pass=5, max_grad_evals=2_000,
                eval_every_n_grads=153, max_outer=1_000_000, max_inner=5,
                lambda_max_starts=8, prune_inner=True)


def build_problem(cfg: dict):
    objectives, gradients, L, joint = _make_mlp_problem(
        K=cfg["K"], p=cfg["p"], n=cfg["n"], hidden_sizes=cfg["hidden_sizes"],
        seed=7, activation=cfg["activation"], w_true_scale=1.0,
    )
    joint_oracle = prefer_fused_joint_oracle(joint)
    x0 = make_mlp_initial_point(K=cfg["K"], p=cfg["p"],
                                hidden_sizes=cfg["hidden_sizes"], seed=8)
    return objectives, gradients, L, joint_oracle, x0


def run_baseline(cfg, objectives, gradients, L, joint_oracle, x0):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = uniform_discretisation(
            K=cfg["K"], objectives=objectives, grad_objectives=gradients,
            L=L, x0=x0, resolution=cfg["coarse_resolution"],
            n_passes=cfg["n_passes"],
            steps_per_point_per_pass=cfg["steps_per_point_per_pass"],
            eval_every_n_grads=cfg["eval_every_n_grads"],
            max_grad_evals=cfg["max_grad_evals"],
            node_tol=NODE_TOL,
            evaluate_coverage=True, joint_oracle=joint_oracle, verbose=True,
        )
    res["_warnings"] = sorted({str(w.message) for w in caught})
    return res


def run_adaptive(cfg, epsilon, objectives, gradients, L, joint_oracle, x0):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = algorithm_adaptive(
            K=cfg["K"], d=int(x0.size), objectives=objectives,
            grad_objectives=gradients, L=L, x0=x0,
            max_outer=cfg["max_outer"], max_inner=cfg["max_inner"],
            epsilon=epsilon, eval_every_n_grads=cfg["eval_every_n_grads"],
            target_cov=None, lambda_solver="ipopt", require_ipopt=True,
            lambda_max_starts=cfg["lambda_max_starts"],
            prune_inner=cfg["prune_inner"],
            max_grad_evals=cfg["max_grad_evals"],
            joint_oracle=joint_oracle, verbose=True,
        )
    res["_warnings"] = sorted({str(w.message) for w in caught})
    return res


# ---------------------------------------------------------------------------
#  Pareto extraction
# ---------------------------------------------------------------------------
def evaluate_points(points, K, d, joint_oracle):
    """One joint-oracle call per point -> (fvals (m,K), grads (m,K,d)).

    Presentation/audit work AFTER the runs; never counted on any axis.
    """
    fvals, grads = [], []
    for x in np.atleast_2d(points):
        fv, gv = validate_oracle_output(*joint_oracle(np.asarray(x, float)),
                                        K, d)
        fvals.append(np.asarray(fv, float))
        grads.append(np.asarray(gv, float))
    return np.asarray(fvals), np.asarray(grads)


def lambda_sweep_selection(grads: np.ndarray, n_sweep: int = N_LAMBDA_SWEEP):
    """For each t in [0,1]: index of argmin_i ||t*g1_i + (1-t)*g2_i||^2.

    grads has shape (m, 2, d).  Pure algebra on stored/audited gradients:
    ||t*g1+(1-t)*g2||^2 = t^2 a + 2 t(1-t) b + (1-t)^2 c with
    a = g1.g1, b = g1.g2, c = g2.g2 per point.
    """
    g1, g2 = grads[:, 0, :], grads[:, 1, :]
    a = np.einsum("md,md->m", g1, g1)
    b = np.einsum("md,md->m", g1, g2)
    c = np.einsum("md,md->m", g2, g2)
    ts = np.linspace(0.0, 1.0, n_sweep)
    # (n_sweep, m) matrix of squared norms.
    q = (ts[:, None] ** 2) * a[None, :] \
        + 2.0 * (ts * (1.0 - ts))[:, None] * b[None, :] \
        + ((1.0 - ts)[:, None] ** 2) * c[None, :]
    sel = np.argmin(q, axis=1)
    gn_at_t = q[np.arange(len(ts)), sel]
    return ts, sel, gn_at_t


def front_segments(ts, sel, fvals):
    """Compress the sweep into segments: consecutive t with the same point."""
    segments = []
    start = 0
    for i in range(1, len(ts) + 1):
        if i == len(ts) or sel[i] != sel[start]:
            idx = int(sel[start])
            segments.append({
                "t_from": float(ts[start]),
                "t_to": float(ts[i - 1]),
                "point_index": idx,
                "F1": float(fvals[idx][0]),
                "F2": float(fvals[idx][1]),
            })
            if i < len(ts):
                start = i
    return segments


def baseline_certificate(coarse_grid, node_fvals, node_grads, resolution):
    """Exact post-run audit of the K=2 covering-bound certificate."""
    h = 1.0 / (2.0 * resolution)
    rows = []
    for i, lam in enumerate(coarse_grid):
        g1, g2 = node_grads[i, 0, :], node_grads[i, 1, :]
        g_lam = lam[0] * g1 + lam[1] * g2
        node_norm = float(np.linalg.norm(g_lam))
        D = float(np.linalg.norm(g1 - g2))
        rows.append({
            "t": float(lam[0]),
            "node_grad_sq": node_norm ** 2,
            "node_passes_tol": node_norm ** 2 <= NODE_TOL,
            "D": D,
            "certified_bound": (node_norm + h * D) ** 2,
        })
    overall = max(r["certified_bound"] for r in rows)
    D_max = max(r["D"] for r in rows)
    return {
        "h": h,
        "node_tol": NODE_TOL,
        "rows": rows,
        "D_max": D_max,
        "grid_floor_sq": (h * D_max) ** 2,
        "overall_certified_level": overall,
        "target_eps": EPS_BASELINE,
        "certifies_target": overall <= EPS_BASELINE,
    }


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------
def plot_combo(out_path, bl_pack, ad_pack, title, ad_short):
    """Log-log Pareto figure: line = delivered point as t sweeps 0..1,
    one marker per distinct delivered point coloured by the mean t it
    serves; run outcomes go into the (multi-line) title, legend stays
    short.  Log axes because the baseline's extreme nodes reach losses
    orders of magnitude above the trade-off region."""
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    cmap = plt.get_cmap("viridis")
    for pack, color, marker, name in (
        (bl_pack, "#d62728", "s", "baseline (Algorithm 1)"),
        (ad_pack, "#9467bd", "^", f"adaptive (Algorithm 2), {ad_short}"),
    ):
        ts, sel, fvals = pack["ts"], pack["sel"], pack["fvals"]
        ax.plot(fvals[sel, 0], fvals[sel, 1], color=color, lw=1.6,
                alpha=0.85, label=name, zorder=3)
        distinct = sorted(set(int(s) for s in sel))
        ax.scatter(fvals[distinct, 0], fvals[distinct, 1],
                   c=[cmap(np.mean(ts[sel == i])) for i in distinct],
                   edgecolors=color, marker=marker, s=75, zorder=5,
                   linewidths=1.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045)
    cbar.set_label(r"marker colour: mean $\lambda_1$ (=t) served by the point")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$F_1(x^*_\lambda)$  (class-1 cross-entropy, log)")
    ax.set_ylabel(r"$F_2(x^*_\lambda)$  (class-2 cross-entropy, log)")
    ax.set_title(
        f"{title}\nbaseline: {bl_pack['label']}\nadaptive: {ad_pack['label']}",
        fontsize=9.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Override the baseline grid resolution r (default: the value "
             "in problem_config). Output goes to a separate folder suffixed "
             "_r<R>, so existing results are never touched. Only the "
             "baseline uses r; the adaptive runs are re-run identically so "
             "the folder stays self-contained.")
    args = parser.parse_args()

    cfg = problem_config(args.smoke)
    if args.resolution is not None:
        if args.resolution < 1:
            raise SystemExit("--resolution must be a positive integer")
        cfg["coarse_resolution"] = args.resolution
    suffix = ("_smoke" if args.smoke else "") + (
        f"_r{args.resolution}" if args.resolution is not None else "")
    out_dir = OUTPUT_ROOT / f"pareto_certified_without_256_checkpoints{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    objectives, gradients, L, joint_oracle, x0 = build_problem(cfg)
    K, d = cfg["K"], int(x0.size)

    # ---- runs (serial) ----
    print(f"=== baseline, eps={EPS_BASELINE} -> node_tol={NODE_TOL} ===")
    bl = run_baseline(cfg, objectives, gradients, L, joint_oracle, x0)
    adaptive_runs = {}
    for eps in EPS_ADAPTIVE:
        print(f"=== adaptive, eps={eps} ===")
        adaptive_runs[eps] = run_adaptive(cfg, eps, objectives, gradients, L,
                                          joint_oracle, x0)

    # ---- persist per-run summaries ----
    def save_summary(name, res, extra):
        run_dir = out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": _json_ready(cfg),
            "data_seed": 7, "init_seed": 8, "d": d,
            "checkpoint_metric": "self_reported (without-256 track)",
            "curves": _result_curves(res),
            "runtime_warnings": res.get("_warnings", []),
        }
        summary.update(_json_ready(extra))
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    save_summary("baseline_eps0.01", bl, {
        "mode": "certified (node_tol)",
        "eps": EPS_BASELINE, "node_tol": NODE_TOL,
        "certified": bl["certified"],
        "certified_grad_evals": bl["certified_grad_evals"],
        "n_unserved": len(bl["unserved_nodes"] or []),
        "node_grad_sq_final": bl["node_grad_sq"],
    })
    for eps, res in adaptive_runs.items():
        save_summary(f"adaptive_eps{eps}", res, {
            "mode": "certified (epsilon)",
            "eps": eps,
            "certified": bool(res["pc_history"][-1] <= 2 * eps / 3),
            "stop_pc_value": float(res["pc_history"][-1]),
            "grad_evals_total": int(res["grad_evals_history"][-1]),
            "bundle_size": int(len(res["bundle"].points)),
            "L_scale_final": res["L_scale_final"],
        })

    # ---- Pareto extraction ----
    # Baseline: one audit call per node (presentation work).
    bl_points = np.asarray(bl["final_solutions"], float)
    bl_fvals, bl_grads = evaluate_points(bl_points, K, d, joint_oracle)
    cert = baseline_certificate(np.asarray(bl["coarse_grid"], float),
                                bl_fvals, bl_grads, cfg["coarse_resolution"])

    # Adaptive: everything already stored in the bundle.
    packs = {}
    ts_b, sel_b, gn_b = lambda_sweep_selection(bl_grads)
    node_part = (f"nodes certified at node_tol={NODE_TOL:g}"
                 if bl["certified"] else
                 f"node certification FAILED within "
                 f"{cfg['max_grad_evals']} grads (node_tol={NODE_TOL:g})")
    packs["baseline"] = {
        "ts": ts_b, "sel": sel_b, "fvals": bl_fvals, "gn_at_t": gn_b,
        "label": (f"{node_part}; audited all-lambda level "
                  f"{cert['overall_certified_level']:.3g} "
                  f"({'<=' if cert['certifies_target'] else '>'} "
                  f"eps {EPS_BASELINE:g})"),
    }
    for eps, res in adaptive_runs.items():
        bundle = res["bundle"]
        fv = np.asarray([np.asarray(v, float) for v in bundle.fvals])
        gv = np.asarray([np.asarray(g, float) for g in bundle.grads])
        ts_a, sel_a, gn_a = lambda_sweep_selection(gv)
        certified = bool(res["pc_history"][-1] <= 2 * eps / 3)
        used = int(res["grad_evals_history"][-1])
        packs[f"adaptive_{eps}"] = {
            "ts": ts_a, "sel": sel_a, "fvals": fv, "gn_at_t": gn_a,
            "label": (f"certified eps={eps} at {used} grads, m={len(fv)}"
                      if certified else
                      f"NOT certified within {cfg['max_grad_evals']} grads"),
        }

    # ---- figures ----
    plot_combo(out_dir / "pareto_front_combo1_bl0.01_a0.01.png",
               packs["baseline"], packs["adaptive_0.01"],
               "Pareto fronts, K=2 96x96 (certified, self-reported track) — "
               f"combo 1: baseline eps {EPS_BASELINE} vs adaptive eps 0.01",
               "eps=0.01")
    plot_combo(out_dir / "pareto_front_combo2_bl0.01_a0.001.png",
               packs["baseline"], packs["adaptive_0.001"],
               "Pareto fronts, K=2 96x96 (certified, self-reported track) — "
               f"combo 2: baseline eps {EPS_BASELINE} vs adaptive eps 0.001",
               "eps=0.001")

    # ---- data file ----
    data = {
        "config": _json_ready(cfg),
        "node_tol_derivation": {
            "eps_baseline": EPS_BASELINE,
            "rule": "equal norm split: node_tol = eps/4; grid share h*D "
                    "with h = 1/(2r); exact certificate audited post-run",
            "node_tol": NODE_TOL,
        },
        "baseline_certificate_audit": cert,
        "fronts": {
            name: front_segments(p["ts"], p["sel"], p["fvals"])
            for name, p in packs.items()
        },
        "labels": {name: p["label"] for name, p in packs.items()},
    }
    (out_dir / "pareto_data.json").write_text(
        json.dumps(_json_ready(data), indent=2), encoding="utf-8")

    # ---- README ----
    n_failed = len(bl["unserved_nodes"] or [])
    readme = f"""# Certified-mode Pareto-front comparison (K=2, without-256 track)

Everything here was produced by
`Original_py/run_pareto_certified_without_256_checkpoints.py`; the module
docstring there is the full specification. Summary of what this folder is:

- Three certified runs on ONE shared problem instance
  (K=2, p={cfg['p']}, n={cfg['n']}, hidden={cfg['hidden_sizes']}, tanh,
  seeds 7/8, r={cfg['coarse_resolution']}, budget {cfg['max_grad_evals']}
  kept as the fuse; checkpoints are self-reported — no 256-start solves):
  - `baseline_eps0.01/` — Algorithm 1 in certification mode.
    eps 0.01 was split in norm space: node_tol = eps/4 = {NODE_TOL:g}
    for the grid nodes, the other half reserved for the r-grid's
    between-node degradation (h*D with h = 1/(2r) = {cert['h']:g}).
    Outcome: {node_part}.
  - `adaptive_eps0.01/`, `adaptive_eps0.001/` — Algorithm 2 with epsilon
    set; stops when its own lambda-search value <= 2*eps/3.
    Outcomes: {packs['adaptive_0.01']['label']} /
    {packs['adaptive_0.001']['label']}.
- Post-run audit of the baseline's all-lambda certificate (exact for K=2):
  for every lambda the nearest node's point satisfies
  ||grad F_lambda|| <= ||grad F_lambda_i|| + h*||g1-g2||, so the certified
  level is max_i (sqrt(node value) + h*D_i)^2.
  Measured: D_max = {cert['D_max']:.3f}, grid share alone (h*D_max)^2 =
  {cert['grid_floor_sq']:.4g}, overall audited level =
  {cert['overall_certified_level']:.4g}
  ({'certifies' if cert['certifies_target'] else 'does NOT certify'} the
  eps {EPS_BASELINE:g} target). Unserved nodes at stop: {n_failed}.
  If (h*D_max)^2 alone exceeds eps, NO node_tol can reach eps at this r —
  that is the grid floor in certified-mode language; the remedy is a finer
  grid (larger r), not a smaller node_tol.
- `pareto_front_combo1_bl0.01_a0.01.png`, `pareto_front_combo2_bl0.01_a0.001.png`
  — the requested figures. Axes: F1 and F2 (per-class cross-entropy) at
  the point each method DELIVERS for weight lambda=(t,1-t); the same
  delivery rule is used for both methods (argmin over the method's point
  set of ||grad F_lambda||^2, computed from stored/audited gradients —
  no re-optimisation). Lines trace the delivered point as t sweeps 0..1
  ({N_LAMBDA_SWEEP} values); markers are the distinct delivered points,
  coloured by the mean t they serve. Lower-left is better (both losses
  smaller); a curve that lies below-left of the other dominates it.
- `pareto_data.json` — machine-readable fronts (t-segments per delivered
  point with F1/F2), the full node-by-node certificate audit, and labels.

Caveats, stated once: the adaptive stop rule trusts its own
{cfg['lambda_max_starts']}-start search (a heuristic lower bound of an
NP-hard max), so "certified" means "its own working search could not find
a weight above 2*eps/3" — same search strength it uses to run at all.
prune_inner=True voids the epsilon-mode convergence-TIME proof condition
(a RuntimeWarning is recorded in the summaries); the stopping rule itself
is unaffected. Wall-clock on this track is not comparable to the
256-start track.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"DONE in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
