"""Pure fixed-budget protocol: NO tolerance inputs anywhere.

NEW FILE (July 27, 2026, session 12).  Design record: ``Note/Jul_27_note``
(MLP addendum, to be appended) / conversation of Jul 27.  Engine files
untouched.  User-approved design:

Both methods spend EXACTLY the same budget B in identical work units and
share one execution loop; the ONLY difference is a next-lambda POLICY:

* work unit = one SEGMENT: ``epoch_len`` Momentum-SVRG minibatch steps
  (NO trigger early-stop — that compares against a target, i.e. a
  tolerance) followed by one full joint evaluation at the endpoint
  (charged; its Gram and objective vector join the delivered set);
* every allocation decision spends ``s`` consecutive segments at the
  chosen lambda (shared s — the same knob for both methods);
* warm start is the CHAIN rule for both methods: each segment starts
  from the latest delivered endpoint (the engine's T-map anchor needs
  per-point Jacobians — memory-infeasible at bundle sizes ~B/18.8 — so
  the pure protocol makes the warm-start policy identical instead);
* stopping rule = budget exhaustion, nothing else.  No node_tol, no
  solve_target, no epsilon, no rel_target, no certificates, no shares.

Policies:
* baseline (per r): the next lambda is the next node of the snake-sorted
  uniform grid at resolution r, cycling into further passes if budget
  remains.  Decisions are free.
* adaptive: the next lambda is the current worst weight — a strict
  in-family search (``--targeting-starts``, default 24) over the
  delivered Gram stack.  Decision time is real method work and stays
  INSIDE the adaptive CPU axis.

Measurement (the only place any quality number appears): post-hoc
strict 64-start audits — final delivered-set GN* for every leg; for the
adaptive leg additionally prefix audits at checkpoints with the
monotone lower-bound envelope.  Audit time is off both cost axes.

Usage (one leg per invocation; legs skip themselves if their summary
already exists, --force re-runs):
    python run_pure_budget_K6_without_256_checkpoints.py
        --run baseline --r 10 --s 5 [--budget 80912]
    python ... --run adaptive --s 5
    python ... --figure          # collect all legs, draw + README
    python ... --smoke           # tiny end-to-end check
"""
import argparse
import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

from math import comb  # noqa: E402
from pathlib import Path  # noqa: E402

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from experiments import make_mlp_initial_point  # noqa: E402  (torch first)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from bundle import (  # noqa: E402
    prefer_fused_joint_oracle,
    validate_oracle_output,
)
from objectives_torch_fast import make_mlp_nonconvex_fast  # noqa: E402
from algorithm_fast_without_256_checkpoints import (  # noqa: E402
    _maximise_GN_fast,
    ipopt_available,
)
from baseline_svrg_certified_without_256_checkpoints import (  # noqa: E402
    _GramSet,
    _support_batch,
)
from baseline_without_256_checkpoints import (  # noqa: E402
    _sort_grid_for_warmstart,
    _uniform_simplex_grid,
)
from run_fixed_budget_K6_without_256_checkpoints import (  # noqa: E402
    _audit_prefixes,
    AUDIT_STARTS,
)
from run_experiments import DATA_SEED, INIT_SEED, _json_ready  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent.parent / "output"
V2_HOME = OUTPUT_ROOT / "pure_budget_without_256_checkpoints_SVRG_IPOPT_Baseline/baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints"
MAX_SAFEGUARD_RETRIES = 4

TRIAL_INSTANCE = dict(
    K=6, p=20, n=50_000, hidden_sizes=[96, 96], activation="tanh",
    msvrg_batch=4096, sampler_seed=41,
    msvrg_step_const=0.1, msvrg_momentum=0.5, msvrg_epoch_len=None,
)
SMOKE_INSTANCE = dict(
    K=6, p=6, n=300, hidden_sizes=[8], activation="tanh",
    msvrg_batch=60, sampler_seed=41,
    msvrg_step_const=0.1, msvrg_momentum=0.5, msvrg_epoch_len=5,
)

_ADAPT_KW = dict(color="#2ca02c", marker="^", ms=4, lw=1.8)


class _Budget:
    """Grad-equivalent meter: joint = K units, minibatch row pair = 2K/n."""

    def __init__(self, K, n, stoch, limit):
        self.K, self.n = K, n
        self.stoch = stoch
        self.ifo0 = stoch.ifo_count
        self.joint_calls = 0
        self.limit = float(limit)

    def spent(self):
        return (self.joint_calls * self.K
                + (self.stoch.ifo_count - self.ifo0) * self.K / float(self.n))

    def segment_upper_cost(self, epoch_len, b_total):
        return epoch_len * 2.0 * b_total * self.K / float(self.n) + self.K

    def allows_segment(self, epoch_len, b_total):
        return self.spent() + self.segment_upper_cost(epoch_len, b_total) \
            <= self.limit + 1e-9


def _run_leg(policy_name, next_lam, cfg, args, out_dir, extra_cfg):
    """Shared executor: identical for both methods except ``next_lam``.

    ``next_lam(gram_stack, fvals, prev_lam)`` returns the lambda for the
    next block of ``s`` segments (and may spend CPU — it is timed inside
    the leg's wall clock, i.e. ON the CPU axis).
    """
    t_build = time.time()
    objectives, gradients, L, joint, stoch = make_mlp_nonconvex_fast(
        K=cfg["K"], p=cfg["p"], n=cfg["n"],
        hidden_sizes=cfg["hidden_sizes"], seed=DATA_SEED,
        activation=cfg["activation"], w_true_scale=1.0,
        batch_size=cfg["msvrg_batch"], sampler_seed=cfg["sampler_seed"],
    )
    joint_oracle = prefer_fused_joint_oracle(joint)
    x0 = make_mlp_initial_point(K=cfg["K"], p=cfg["p"],
                                hidden_sizes=cfg["hidden_sizes"],
                                seed=INIT_SEED)
    K, d, n = cfg["K"], int(x0.size), cfg["n"]
    L_arr = np.asarray(L, dtype=float)
    epoch_len = (cfg["msvrg_epoch_len"] if cfg["msvrg_epoch_len"]
                 else max(1, int(np.ceil(n / float(cfg["msvrg_batch"])))))
    print(f"[{policy_name}] instance in {time.time() - t_build:.1f}s "
          f"(epoch_len={epoch_len})", flush=True)

    # x0 joint evaluation: checkpoint-0 setup, uncharged (track norm).
    f0, J0 = validate_oracle_output(*joint_oracle(x0), K, d)
    grams = [J0 @ J0.T]
    fvals = [np.asarray(f0, dtype=float)]
    chain_x, chain_J, chain_f = x0.copy(), J0, f0

    budget = _Budget(K, n, stoch, args.budget)
    L_scale = 1.0
    safeguard_retries = 0
    lam_history = []
    ck_grads, ck_cpu, ck_m = [0.0], [0.0], [1]
    grad_at_ck = 0.0
    t0 = time.time()
    decision_seconds = 0.0

    prev_lam = None
    while budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
        t_dec = time.time()
        lam = np.asarray(
            next_lam(grams, fvals, prev_lam), dtype=float)
        decision_seconds += time.time() - t_dec
        prev_lam = lam
        lam_history.append(lam.copy())

        retries_here = 0
        for _k in range(args.s):
            if not budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
                break
            # ---- one tolerance-free segment from the chain anchor ----
            g_a_full = chain_J.T @ lam
            F_a = float(chain_f @ lam)
            eta = cfg["msvrg_step_const"] / (float(lam @ L_arr) * L_scale)
            stoch.set_anchor(chain_x)
            y = chain_x.copy()
            u_vec = np.zeros(d)
            for _t in range(epoch_len):
                batch = _support_batch(stoch.sample_batch(), lam)
                g_y_S, g_a_S = stoch.grad_pair(y, lam, batch)
                u_vec = cfg["msvrg_momentum"] * u_vec + (g_y_S - g_a_S
                                                         + g_a_full)
                y = y - eta * u_vec
            f_y, J_y = validate_oracle_output(*joint_oracle(y), K, d)
            budget.joint_calls += 1
            grams.append(J_y @ J_y.T)
            fvals.append(np.asarray(f_y, dtype=float))
            if float(f_y @ lam) > F_a + 1e-10 * (1.0 + abs(F_a)):
                # Descent safeguard (same rule as the engines): halve the
                # step, retry from the SAME anchor; the violating point is
                # paid for and delivered (harmless under the min).
                L_scale *= 2.0
                safeguard_retries += 1
                retries_here += 1
                if retries_here > MAX_SAFEGUARD_RETRIES:
                    chain_x, chain_J, chain_f = y, J_y, f_y
                    retries_here = 0
            else:
                chain_x, chain_J, chain_f = y, J_y, f_y

            if budget.spent() - grad_at_ck >= args.eval_every:
                grad_at_ck = budget.spent()
                ck_grads.append(budget.spent())
                ck_cpu.append(time.time() - t0)
                ck_m.append(len(grams))

    wall = time.time() - t0
    ck_grads.append(budget.spent())
    ck_cpu.append(wall)
    ck_m.append(len(grams))
    Ms = np.asarray(grams, dtype=float)
    print(f"[{policy_name}] budget spent: {budget.spent():.1f} of "
          f"{args.budget} | segments={len(grams) - 1} | wall={wall:.1f}s "
          f"| decision={decision_seconds:.1f}s "
          f"| L_scale={L_scale}", flush=True)

    # ---- post-hoc measurement (off both axes) ----
    t_a = time.time()
    if policy_name == "adaptive":
        audited, audit_lams, _ = _audit_prefixes(Ms, ck_m, K)
        audited_env = [float(v) for v in
                       np.maximum.accumulate(np.asarray(audited)[::-1])[::-1]]
        final_audit, lam_star = audited[-1], audit_lams[-1]
    else:
        gs = _GramSet(list(Ms), K)
        v, lam_star = _maximise_GN_fast(gs, prev_lam=None, tier="strict",
                                        max_starts=AUDIT_STARTS)
        audited, audited_env, audit_lams = [float(v)], None, None
        final_audit = float(v)
        lam_star = [float(t) for t in lam_star]
    audit_seconds = time.time() - t_a

    lam_arr = np.asarray(lam_history, dtype=float)
    distinct = (np.unique(lam_arr.round(12), axis=0).shape[0]
                if lam_arr.size else 0)
    summary = {
        "protocol": ("pure fixed budget: shared segment unit, shared s, "
                     "chain warm start; ONLY the next-lambda policy "
                     "differs; no tolerance parameter anywhere; stop = "
                     "budget"),
        "policy": policy_name,
        "config_instance": _json_ready(cfg),
        "budget": args.budget,
        "s": args.s,
        "eval_every": args.eval_every,
        "targeting_starts": (args.targeting_starts
                             if policy_name == "adaptive" else None),
        "extra": _json_ready(extra_cfg),
        "grad_equiv_total": float(budget.spent()),
        "joint_calls": int(budget.joint_calls),
        "wall_seconds": wall,
        "decision_seconds": decision_seconds,
        "segments_total": int(len(grams) - 1),
        "m_final": int(Ms.shape[0]),
        "n_decisions": int(len(lam_history)),
        "n_distinct_lambdas": int(distinct),
        "L_scale_final": L_scale,
        "safeguard_retries": int(safeguard_retries),
        "ck_grads": ck_grads, "ck_cpu": ck_cpu, "ck_m": ck_m,
        "final_audit": float(final_audit),
        "audited_gn_history": audited,
        "audited_gn_envelope": audited_env,
        "lambda_star": lam_star,
        "audit_seconds": audit_seconds,
        "data_seed": DATA_SEED, "init_seed": INIT_SEED,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "grams.npz", gram_stack=Ms,
                        fvals=np.asarray(fvals),
                        lam_history=lam_arr)
    print(f"[{policy_name}] final strict audit = {final_audit:.6f} "
          f"-> {out_dir}", flush=True)
    return summary


def _baseline_policy(grid):
    state = {"i": -1}

    def nxt(grams, fvals, prev_lam):
        state["i"] += 1
        return grid[state["i"] % grid.shape[0]]
    return nxt


def _adaptive_policy(K, starts):
    def nxt(grams, fvals, prev_lam):
        gs = _GramSet(list(np.asarray(grams)), K)
        _v, lam = _maximise_GN_fast(gs, prev_lam=prev_lam, tier="strict",
                                    max_starts=starts)
        return lam
    return nxt


def _leg_dir(home, policy, r, s):
    return home / (f"adaptive_s{s}" if policy == "adaptive"
                   else f"baseline_r{r:02d}_s{s}")


def _figure(home, args, cfg):
    K = cfg["K"]
    legs = []
    for p in sorted(home.glob("*/summary.json")):
        with open(p, "r", encoding="utf-8") as fh:
            legs.append(json.load(fh))
    ad = [s for s in legs if s["policy"] == "adaptive"]
    bl = [s for s in legs if s["policy"] == "baseline"]
    if not ad:
        raise SystemExit("--figure: no adaptive leg found")
    ad = ad[0]

    for axis_key, x_label, fname in (
        ("ck_grads", "total gradient evaluations, grad-equivalents",
         "pure_budget_gn_vs_grads.png"),
        ("ck_cpu", "CPU time, seconds", "pure_budget_gn_vs_cpu.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        hx = np.asarray(ad[axis_key], dtype=float)
        hy = np.maximum(np.asarray(ad["audited_gn_envelope"], dtype=float),
                        np.finfo(float).tiny)
        positive = [v for v in hx if v > 0]
        x0_pseudo = (min(positive) / 3.0) if positive else 1e-3
        hx = np.where(hx > 0, hx, x0_pseudo)
        ax.plot(hx, hy,
                label=f"adaptive policy (s={ad['s']}) — strict 64-start "
                      f"prefix audit", **_ADAPT_KW)

        # Jul 27 (user): baseline legs are FINAL POINTS only (their
        # audited trajectories stay in the summaries, unplotted);
        # labels staggered into a connector-linked column so coincident
        # points (e.g. the collapsed s=5 legs, all at the x0 plateau)
        # stay readable; legend moved off the data (lower left).
        reds = plt.get_cmap("Reds")
        rs = sorted({s["extra"]["r"] for s in bl})
        pts = []
        for sm in bl:
            r = sm["extra"]["r"]
            color = reds(0.45 + 0.5 * rs.index(r) / max(1, len(rs) - 1))
            main = sm["s"] == args.s
            x_end = float(sm["grad_equiv_total"] if axis_key == "ck_grads"
                          else sm["wall_seconds"])
            y_end = sm["final_audit"]
            if main:
                ax.scatter([x_end], [y_end], marker="x", s=52, zorder=6,
                           color=color, linewidth=1.8)
            else:
                ax.scatter([x_end], [y_end], marker="o", s=40, zorder=6,
                           facecolors="none", edgecolor=color,
                           linewidth=1.4)
            pts.append((y_end, x_end, f"r={r}: {y_end:.3g}"))
        pts.sort(key=lambda t: -t[0])
        prev_y, dy = None, 6
        for y_end, x_end, text in pts:
            close = (prev_y is not None
                     and abs(np.log10(max(y_end, 1e-300))
                             - np.log10(max(prev_y, 1e-300))) < 0.35)
            dy = (dy - 15) if close else 6
            prev_y = y_end
            ax.annotate(text, xy=(x_end, y_end), xytext=(-12, dy),
                        textcoords="offset points", ha="right",
                        fontsize=7.0, color="#8b1a1a",
                        arrowprops=dict(arrowstyle="-", color="#8b1a1a",
                                        lw=0.5, alpha=0.5))

        ax.scatter([], [], marker="x", s=52, color=reds(0.7), linewidth=1.8,
                   label=f"baseline s={args.s} — final audit")
        if any(sm["s"] != args.s for sm in bl):
            ax.scatter([], [], marker="o", s=40, facecolors="none",
                       edgecolor=reds(0.7), linewidth=1.4,
                       label="baseline s=1 sensitivity — final audit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("strict full-simplex GN* (64-start audit — one meter)")
        ax.set_title(
            f"MLP K={K}, p={cfg['p']}, n={cfg['n']}, h={cfg['hidden_sizes']},"
            f" {cfg['activation']} — PURE fixed budget, no tolerance inputs\n"
            f"B={args.budget:,.0f} grad-equivalents for every leg; shared "
            f"segment unit; only the next-lambda policy differs",
            fontsize=9.6,
        )
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(home / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    rows = "\n".join(
        f"| {sm['extra'].get('r', '—')} | {sm['s']} "
        f"| {sm['n_distinct_lambdas']:,} "
        f"| {sm['grad_equiv_total']:.0f} | {sm['wall_seconds']:.0f} "
        f"| {sm['final_audit']:.4e} |"
        for sm in bl + [ad])
    epoch_len = (cfg["msvrg_epoch_len"] if cfg["msvrg_epoch_len"]
                 else max(1, int(np.ceil(cfg["n"] / float(cfg["msvrg_batch"])))))
    readme = f"""# Pure fixed-budget comparison (no tolerance inputs; without-256 track)

Produced by `Original_py/run_pure_budget_K6_without_256_checkpoints.py`
(July 27, 2026).  Protocol (user-designed): every leg spends the SAME
budget B = {args.budget:,.0f} grad-equivalents in identical work units
(one segment = {epoch_len} minibatch Momentum-SVRG steps, no trigger,
+ 1 full joint evaluation), s = {args.s} consecutive segments per
allocation decision, chain warm start for BOTH methods.  The ONLY
difference between methods is the next-lambda policy: snake grid order
(baseline, per r) vs strict worst-lambda search (adaptive,
{ad.get('targeting_starts')} starts, decision time inside its CPU axis).
No node_tol / solve_target / epsilon / rel_target anywhere; stop =
budget.  Quality appears only in the post-hoc strict 64-start audits
(adaptive curve: prefix audits, monotone lower-bound envelope).

| r | s | distinct lambdas visited | grad-equiv | wall s | strict audit |
|---|---|--------------------------|------------|--------|--------------|
{rows}

(last row = adaptive; "distinct lambdas" for the baseline = grid nodes
actually visited = its coverage at this budget.)

Sensitivity legs (s=1, open circles) let the baseline run at its own
preferred decision granularity — its decisions are free, so s>1 is a
constraint it inherits from the shared-chunk design; disclosed.

CPU-axis caveat: adaptive decision time (strict searches over a growing
Gram stack) is method work, on-axis; audits are off-axis for everyone.
MLP torch runs are not bit-reproducible in this environment
(session-12 finding); one realization each.
"""
    (home / "README.md").write_text(readme, encoding="utf-8")
    print(f"FIGURE + README -> {home}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, choices=["baseline", "adaptive"])
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--s", type=int, default=5)
    parser.add_argument("--budget", type=float, default=80_912.0)
    parser.add_argument("--eval-every", type=float, default=2000.0)
    parser.add_argument("--targeting-starts", type=int, default=24)
    parser.add_argument("--figure", action="store_true")
    parser.add_argument("--backfill-audits", action="store_true",
                        help="compute the strict prefix-audit trajectory "
                             "for baseline legs that lack it (from their "
                             "stored grams.npz); off-axis, post-hoc")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not ipopt_available():
        raise RuntimeError("IPOPT is required but unavailable.")

    cfg = dict(SMOKE_INSTANCE if args.smoke else TRIAL_INSTANCE)
    home = V2_HOME / (f"pure_budget_B{args.budget:.0f}"
                      + ("_SMOKE" if args.smoke else ""))
    home.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.budget = 400.0
        args.eval_every = 50.0
        args.targeting_starts = 8
        for r, s in ((2, 2), (2, 1)):
            grid = _sort_grid_for_warmstart(_uniform_simplex_grid(cfg["K"], r))
            a2 = argparse.Namespace(**{**vars(args), "s": s})
            _run_leg("baseline", _baseline_policy(grid), cfg, a2,
                     _leg_dir(home, "baseline", r, s), {"r": r})
        a2 = argparse.Namespace(**{**vars(args), "s": 2})
        _run_leg("adaptive",
                 _adaptive_policy(cfg["K"], args.targeting_starts), cfg, a2,
                 _leg_dir(home, "adaptive", None, 2), {})
        a2 = argparse.Namespace(**{**vars(args), "s": 2})
        _figure(home, a2, cfg)
        for p in home.glob("*/summary.json"):
            with open(p, "r", encoding="utf-8") as fh:
                sm = json.load(fh)
            assert sm["grad_equiv_total"] <= args.budget + 1e-6
            assert np.isfinite(sm["final_audit"])
        print("SMOKE OK", flush=True)
        return

    if args.backfill_audits:
        for p in sorted(home.glob("baseline_*/summary.json")):
            with open(p, "r", encoding="utf-8") as fh:
                sm = json.load(fh)
            if sm.get("audited_gn_envelope"):
                print(f"[skip] {p.parent.name} already audited", flush=True)
                continue
            npz = np.load(p.parent / "grams.npz")
            Ms = np.asarray(npz["gram_stack"], dtype=float)
            ck_m = [int(v) for v in sm["ck_m"]]
            if ck_m[-1] != Ms.shape[0]:
                ck_m[-1] = int(Ms.shape[0])
            print(f"[backfill] {p.parent.name}: {len(ck_m)} checkpoints, "
                  f"m={Ms.shape[0]}", flush=True)
            t0 = time.time()
            audited, audit_lams, _ = _audit_prefixes(Ms, ck_m, cfg["K"])
            # Merge the leg's original single final search (another valid
            # lower bound) before the envelope: never understate.
            if float(sm["final_audit"]) > audited[-1]:
                audited[-1] = float(sm["final_audit"])
            else:
                sm["final_audit"] = float(audited[-1])
                sm["lambda_star"] = audit_lams[-1]
            env = [float(v) for v in
                   np.maximum.accumulate(np.asarray(audited)[::-1])[::-1]]
            sm["audited_gn_history"] = [float(v) for v in audited]
            sm["audited_gn_envelope"] = env
            sm["audit_seconds"] = float(sm.get("audit_seconds", 0.0)
                                        + time.time() - t0)
            p.write_text(json.dumps(_json_ready(sm), indent=2),
                         encoding="utf-8")
            print(f"[backfill] {p.parent.name}: final audit "
                  f"{sm['final_audit']:.6f}", flush=True)
        return

    if args.figure:
        _figure(home, args, cfg)
        return
    if not args.run:
        raise SystemExit("pass --run baseline|adaptive, --figure or --smoke")

    out_dir = _leg_dir(home, args.run, args.r, args.s)
    if (out_dir / "summary.json").exists() and not args.force:
        print(f"[skip] {out_dir} already has a summary (use --force)",
              flush=True)
        return
    if args.run == "baseline":
        grid = _sort_grid_for_warmstart(_uniform_simplex_grid(cfg["K"], args.r))
        _run_leg("baseline", _baseline_policy(grid), cfg, args, out_dir,
                 {"r": args.r})
    else:
        _run_leg("adaptive",
                 _adaptive_policy(cfg["K"], args.targeting_starts),
                 cfg, args, out_dir, {})


if __name__ == "__main__":
    main()
