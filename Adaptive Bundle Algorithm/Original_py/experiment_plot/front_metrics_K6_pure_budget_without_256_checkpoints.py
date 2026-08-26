"""Post-hoc epsilon-Pareto-front metrics for the K=6 pure fixed-budget legs.

NEW FILE (July 30, 2026).  Change record: ``Note/Jul_30_note.md``.
Pure post-processing of the July-27 artifacts in
``output/baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/
pure_budget_B80912/`` — reads each leg's stored ``grams.npz`` (``fvals``:
the full-batch objective vector of every delivered point, shape (m, 6))
and ``summary.json``; NO experiment is re-run and no engine module is
imported (numpy + matplotlib only, so this stays runnable while other
legs are in flight on the machine).

Metric set follows the SURF paper's presentation (its Table 1: HV and
IGD per method) adapted to this track's recorded conventions:

* front = nondominated subset of a leg's delivered cloud (minimisation);
* reference = the UNION of all legs' fronts (mutual reference — no
  oracle front exists for this family; SURF itself omits IGD when no
  dense reference front is available, their Table 2);
* IGD / max-dist = mean / max over reference points of the Euclidean
  distance to the leg's front (raw value space) — the query-free
  discovered-front semantics of session 13; reported for the RAW union
  front and for the CENTRAL reference (union front restricted to all
  six losses <= CENTRAL_BOUND, the genuine trade-off region — the K=2
  runner's specialist-tail rationale; method fronts are never clipped);
* HV = hypervolume fraction of the CENTRAL box [ideal, CENTRAL_BOUND]^6
  dominated by the leg's front, Monte-Carlo estimated with ONE common
  sample set for all legs (paired comparison; count/seed recorded).
  The raw box is deliberately NOT used for HV: its volume is dominated
  by specialist tails (losses up to ~25 vs x0 ~= 2.2).
* CV / Gap-Ratio (SURF's spacing-uniformity metrics) are OMITTED: they
  need a canonical 1-D ordering of the front, which does not exist at
  K=6 — same reason recorded for the K=5 bandit ("front_uniformity
  omitted for K=5").
* The "eps" of "epsilon-Pareto front" (stationarity sense): each leg's
  final strict 64-start delivered-set audit (AUDIT_STARTS = 64 in
  ``run_fixed_budget_K6_without_256_checkpoints.py``).  That value is a
  search LOWER bound of the true GN* — it can NEVER sign the positive
  claim "GN* <= eps" (the bandit eps1e-4 false-certificate lesson).
  The certified two-sided meter exists only at K=2 (exact 1-D
  structure); at K=6 the max over the 5-simplex of a min of quadratics
  has no exact meter, so the eps labels here are lower bounds, stated
  as such everywhere.

Outputs (into the pure-budget home, existing files untouched):
  FRONTS.md, front_metrics.json, pure_budget_K6_fronts.png,
  pure_budget_K6_fronts_F1F2.png (conventional (F1, F2) front view),
  pure_budget_K6_fronts_pairwise.png (all 15 objective pairs)

Usage:
    python front_metrics_K6_pure_budget_without_256_checkpoints.py
        [--home DIR] [--central-bound 1.0] [--mc-samples 100000]
        [--mc-seed 20260730] [--smoke]
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_HOME = (
    Path(__file__).resolve().parent.parent.parent / "output"
    / "pure_budget_without_256_checkpoints_SVRG_IPOPT_Baseline/baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints"
    / "pure_budget_B80912")
AUDIT_STARTS = 64   # the track's fixed instrument (run_fixed_budget_K6:63)
DOM_TOL = 1e-12     # dominance slack (losses are O(1e-3)..O(10))


# ---------------------------------------------------------------------------
# Nondominated filtering (minimisation), vectorised.
# ---------------------------------------------------------------------------

def nondominated_mask(F, chunk=1024):
    """Boolean mask of the nondominated rows of F (minimisation).

    Sort by row sum: any dominator of z has strictly smaller sum (up to
    DOM_TOL), so it precedes z; by transitivity checking z against the
    KEPT earlier points suffices.
    """
    F = np.asarray(F, dtype=float)
    m = F.shape[0]
    order = np.argsort(F.sum(axis=1), kind="stable")
    Fs = F[order]
    kept_rows = []          # list of survivor blocks, in processed order
    kept_mask_sorted = np.zeros(m, dtype=bool)
    for lo in range(0, m, chunk):
        C = Fs[lo:lo + chunk]
        if kept_rows:
            Kmat = np.concatenate(kept_rows, axis=0)
            ge = (Kmat[None, :, :] <= C[:, None, :] + DOM_TOL).all(axis=2)
            gt = (Kmat[None, :, :] < C[:, None, :] - DOM_TOL).any(axis=2)
            dominated = (ge & gt).any(axis=1)
        else:
            dominated = np.zeros(C.shape[0], dtype=bool)
        # within-chunk pairs (i dominates j) — both directions checked
        ge = (C[None, :, :] <= C[:, None, :] + DOM_TOL).all(axis=2)
        gt = (C[None, :, :] < C[:, None, :] - DOM_TOL).any(axis=2)
        dominated |= (ge & gt & ~dominated[None, :]).any(axis=1)
        keep_local = ~dominated
        if keep_local.any():
            kept_rows.append(C[keep_local])
        kept_mask_sorted[lo:lo + chunk] = keep_local
    mask = np.zeros(m, dtype=bool)
    mask[order] = kept_mask_sorted
    return mask


def nondominated_mask_bruteforce(F):
    """O(m^2) double-loop reference implementation (smoke only)."""
    F = np.asarray(F, dtype=float)
    m = F.shape[0]
    keep = np.ones(m, dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if ((F[j] <= F[i] + DOM_TOL).all()
                    and (F[j] < F[i] - DOM_TOL).any()):
                keep[i] = False
                break
    return keep


# ---------------------------------------------------------------------------
# Distances reference -> front (IGD / max-dist), matmul trick, chunked.
# ---------------------------------------------------------------------------

def ref_to_front_distances(R, F, chunk=512):
    """Euclidean distance from every reference row to its nearest front row."""
    R = np.asarray(R, dtype=float)
    F = np.asarray(F, dtype=float)
    f2 = (F * F).sum(axis=1)
    out = np.empty(R.shape[0])
    for lo in range(0, R.shape[0], chunk):
        Rc = R[lo:lo + chunk]
        d2 = ((Rc * Rc).sum(axis=1)[:, None] + f2[None, :]
              - 2.0 * (Rc @ F.T))
        out[lo:lo + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def front_metrics(R, F):
    """IGD (mean) and max distance from reference set R to front F."""
    if F.size == 0 or R.size == 0:
        return float("inf"), float("inf")
    d = ref_to_front_distances(R, F)
    return float(d.mean()), float(d.max())


# ---------------------------------------------------------------------------
# Monte-Carlo hypervolume in the central box (common samples, paired).
# ---------------------------------------------------------------------------

def hv_dominated_mask(front_in_box, Z, chunk=2048):
    """Per-sample bool: is Z[i] dominated by front_in_box (minimisation)."""
    out = np.zeros(Z.shape[0], dtype=bool)
    if front_in_box.size == 0:
        return out
    Fb = np.asarray(front_in_box, dtype=float)
    for lo in range(0, Z.shape[0], chunk):
        Zc = Z[lo:lo + chunk]
        dom = (Fb[None, :, :] <= Zc[:, None, :] + DOM_TOL).all(axis=2)
        out[lo:lo + chunk] = dom.any(axis=1)
    return out


def hv_fraction(front_in_box, Z, chunk=2048):
    """Fraction of samples Z dominated by front_in_box (minimisation)."""
    return float(hv_dominated_mask(front_in_box, Z, chunk).mean())


# ---------------------------------------------------------------------------
# Smoke checks (synthetic only; no repo data touched).
# ---------------------------------------------------------------------------

def run_smoke():
    rng = np.random.default_rng(0)
    # 1. filter vs brute force, random cloud + planted structure
    F = rng.uniform(0.0, 1.0, size=(300, 6))
    F[10] = F[5]                       # exact duplicate
    F[20] = F[7] + 0.01                # strictly dominated
    F[30] = F[9] * np.ones(6) * 0.0    # all-zero dominator
    fast = nondominated_mask(F, chunk=64)
    slow = nondominated_mask_bruteforce(F)
    assert (fast == slow).all(), "nondominated filter mismatch vs brute force"
    # duplicates: both copies survive under strict-somewhere dominance
    assert fast[5] == fast[10], "duplicate rows treated asymmetrically"
    assert not fast[20], "planted dominated point survived"
    assert fast[30], "all-zero point should be nondominated"
    # 2. IGD hand case (2-D)
    R = np.array([[0.0, 0.0], [1.0, 1.0]])
    Fm = np.array([[0.0, 0.0]])
    igd, mx = front_metrics(R, Fm)
    assert abs(igd - np.sqrt(2.0) / 2.0) < 1e-12 and \
        abs(mx - np.sqrt(2.0)) < 1e-12, "IGD hand case failed"
    # 3. MC hypervolume vs exact, single point in [0,1]^2 at (0.25, 0.5)
    Z = rng.uniform(0.0, 1.0, size=(200_000, 2))
    frac = hv_fraction(np.array([[0.25, 0.5]]), Z)
    exact = 0.75 * 0.5
    se = np.sqrt(exact * (1 - exact) / Z.shape[0])
    assert abs(frac - exact) < 5 * se, \
        f"MC HV {frac} vs exact {exact} beyond 5 SE"
    # 4. two-point union HV (inclusion-exclusion by hand)
    P = np.array([[0.2, 0.6], [0.6, 0.2]])
    exact2 = (0.8 * 0.4) + (0.4 * 0.8) - (0.4 * 0.4)
    frac2 = hv_fraction(P, Z)
    se2 = np.sqrt(exact2 * (1 - exact2) / Z.shape[0])
    assert abs(frac2 - exact2) < 5 * se2, "two-point MC HV failed"
    print("SMOKE OK (filter=bruteforce on 300x6; IGD hand case; "
          "MC HV within 5 SE of exact on two cases)")


# ---------------------------------------------------------------------------
# Main analysis.
# ---------------------------------------------------------------------------

LEG_ORDER_KEY = {"adaptive": 0, "baseline": 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", type=Path, default=DEFAULT_HOME)
    ap.add_argument("--central-bound", type=float, default=1.0,
                    help="all six losses <= this = genuine trade-off "
                         "region (K=2 runner's CENTRAL_BOUND)")
    ap.add_argument("--mc-samples", type=int, default=100_000)
    ap.add_argument("--mc-seed", type=int, default=20260730)
    ap.add_argument("--proj-legs", type=str,
                    default="adaptive_s5,baseline_r10_s1",
                    help="comma list of leg dirnames drawn in the "
                         "projection figures, or 'all' (default: the "
                         "two full-coverage legs — the rest render "
                         "the figures unreadable)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
        return

    home = args.home
    cb = float(args.central_bound)
    legs = []
    for sd in sorted(home.glob("*/summary.json")):
        with open(sd, "r", encoding="utf-8") as fh:
            sm = json.load(fh)
        z = np.load(sd.parent / "grams.npz")
        F = np.asarray(z["fvals"], dtype=float)
        assert F.ndim == 2 and F.shape[1] == 6, f"fvals shape off in {sd}"
        assert F.shape[0] == sm["segments_total"] + 1, \
            f"fvals rows != segments+1 in {sd}"
        assert np.isfinite(sm["final_audit"])
        legs.append(dict(name=sd.parent.name, sm=sm, F=F))
    if not legs:
        raise SystemExit(f"no legs under {home}")
    legs.sort(key=lambda L: (LEG_ORDER_KEY[L["sm"]["policy"]],
                             L["sm"]["extra"].get("r") or 0, L["sm"]["s"]))

    for L in legs:
        mask = nondominated_mask(L["F"])
        L["front"] = L["F"][mask]
        # 2-D front OF THE (F1, F2) PROJECTION — for the conventional
        # front figure; NOT the projection of the 6-D front
        m2 = nondominated_mask(L["F"][:, :2])
        L["front2d"] = L["F"][m2][:, :2]
        print(f"[front] {L['name']:17s} delivered={L['F'].shape[0]:5d} "
              f"front={L['front'].shape[0]:5d} "
              f"front_F1F2={L['front2d'].shape[0]:4d}", flush=True)

    # union front: prefilter against the two full-coverage legs' fronts
    # (cheap dominance cull), then exact self-filter of the survivors
    all_fronts = np.concatenate([L["front"] for L in legs], axis=0)
    strong_list = [L["front"] for L in legs if L["front"].shape[0]
                   and (L["F"].min(axis=0) < 0.1).all()]
    strong = (np.concatenate(strong_list, axis=0) if strong_list
              else np.empty((0, all_fronts.shape[1])))
    if strong.size:
        ge = None
        keep = np.ones(all_fronts.shape[0], dtype=bool)
        for lo in range(0, all_fronts.shape[0], 1024):
            C = all_fronts[lo:lo + 1024]
            ge = (strong[None, :, :] <= C[:, None, :] + DOM_TOL).all(axis=2)
            gt = (strong[None, :, :] < C[:, None, :] - DOM_TOL).any(axis=2)
            keep[lo:lo + 1024] = ~(ge & gt).any(axis=1)
        cand = np.concatenate([all_fronts[keep], strong], axis=0)
        cand = np.unique(cand, axis=0)
    else:
        cand = all_fronts
    union = cand[nondominated_mask(cand)]
    ideal = all_fronts.min(axis=0)
    nadir_raw = union.max(axis=0)
    union_central = union[(union <= cb + DOM_TOL).all(axis=1)]
    print(f"[union] raw={union.shape[0]} central(<= {cb:g})="
          f"{union_central.shape[0]} ideal={np.round(ideal, 4)}",
          flush=True)
    assert (ideal < cb).all(), "central box empty in some objective"
    # sanity: every leg front point is in the union front or dominated.
    # Membership threshold 1e-6, NOT ~1e-12: the matmul distance trick
    # (|R|^2 + |F|^2 - 2 R.F) cancels catastrophically at loss scale
    # ~O(10), so identical rows read as d ~ 1e-8..1e-7, never exactly 0.
    for L in legs:
        d = ref_to_front_distances(L["front"], union)
        on = d < 1e-6
        if not on.all():
            off = L["front"][~on]
            for lo in range(0, off.shape[0], 512):
                O = off[lo:lo + 512]
                ge = (union[None, :, :] <= O[:, None, :] + DOM_TOL).all(axis=2)
                gt = (union[None, :, :] < O[:, None, :] - DOM_TOL).any(axis=2)
                assert (ge & gt).any(axis=1).all(), \
                    f"{L['name']}: front point neither in union nor dominated"

    # common MC samples in the central box (paired across legs)
    rng = np.random.default_rng(args.mc_seed)
    Z = rng.uniform(ideal, cb, size=(args.mc_samples, 6))
    results = {}
    hv_union = hv_fraction(union_central, Z)
    dom_adaptive = None   # legs[0] is the adaptive leg (sort key)
    for L in legs:
        sm, F, front = L["sm"], L["F"], L["front"]
        front_central = front[(front <= cb + DOM_TOL).all(axis=1)]
        igd_raw, mx_raw = front_metrics(union, front)
        igd_c, mx_c = front_metrics(union_central, front)
        dom = hv_dominated_mask(front_central, Z)
        if dom_adaptive is None:
            dom_adaptive = dom
        hv = float(dom.mean())
        ci = 1.96 * np.sqrt(max(hv * (1 - hv), 1e-12) / args.mc_samples)
        # paired HV difference vs the adaptive leg (common samples, so
        # the per-sample difference d_i in {-1,0,1} carries the CI)
        diff = dom_adaptive.astype(np.int8) - dom.astype(np.int8)
        delta = float(diff.mean())
        delta_ci = 1.96 * float(diff.std(ddof=1)) / np.sqrt(len(diff))
        results[L["name"]] = dict(
            policy=sm["policy"], r=sm["extra"].get("r"), s=sm["s"],
            n_points=int(F.shape[0]), n_front=int(front.shape[0]),
            n_front_central=int(front_central.shape[0]),
            per_obj_min=[float(v) for v in F.min(axis=0)],
            eps_search_lb_64start=float(sm["final_audit"]),
            igd_to_union_raw=igd_raw, maxdist_to_union_raw=mx_raw,
            igd_central=igd_c, maxdist_central=mx_c,
            hv_central_frac=hv, hv_central_ci95=float(ci),
            hv_adaptive_minus_this=delta,
            hv_adaptive_minus_this_ci95=float(delta_ci),
            n_front_f1f2_projection=int(L["front2d"].shape[0]),
        )
        L["gap_central"] = ref_to_front_distances(union_central, front)
        print(f"[metrics] {L['name']:17s} IGDc={igd_c:.4f} "
              f"HVc={hv:.4f}+-{ci:.4f} front_c={front_central.shape[0]}",
              flush=True)

    # legs drawn in the projection figures + joint-front composition
    # per objective pair (the counts printed on the panels; recorded
    # here so every number on the figures is quotable from the JSON)
    if args.proj_legs.strip() == "all":
        proj_legs = list(legs)
    else:
        by_name = {L["name"]: L for L in legs}
        want = [w.strip() for w in args.proj_legs.split(",") if w.strip()]
        missing = [w for w in want if w not in by_name]
        assert not missing, f"--proj-legs unknown legs: {missing}"
        proj_legs = [by_name[w] for w in want]

    pf_cache = {}

    def _pair_front(L, xi, yi):
        key = (L["name"], xi, yi)
        if key not in pf_cache:
            P = L["F"][:, [xi, yi]]
            fr = P[nondominated_mask(P)]
            pf_cache[key] = fr[np.argsort(fr[:, 0])]
        return pf_cache[key]

    pair_comp = {}
    for yi in range(1, 6):
        for xi in range(0, yi):
            rows, owner = [], []
            for k, L in enumerate(proj_legs):
                fr = _pair_front(L, xi, yi)
                rows.append(fr)
                owner.append(np.full(fr.shape[0], k))
            R = np.concatenate(rows, axis=0)
            O = np.concatenate(owner, axis=0)
            jm = nondominated_mask(R)
            pair_comp[f"F{xi + 1}-F{yi + 1}"] = {
                L["name"]: int((O[jm] == k).sum())
                for k, L in enumerate(proj_legs)}

    payload = dict(
        produced_by=Path(__file__).name,
        date="2026-07-30",
        semantics=dict(
            front="nondominated subset of the leg's delivered cloud "
                  "(all fvals rows, minimisation)",
            reference="union of all legs' fronts (mutual; no oracle "
                      "front for this family)",
            igd="mean over reference points of Euclidean distance to "
                "the leg front (raw value space); maxdist = max",
            central=f"reference restricted to all six losses <= {cb:g}; "
                    "method fronts never clipped",
            hv=f"Monte-Carlo fraction of the box [ideal, {cb:g}]^6 "
               "dominated by the leg's front; common samples for all "
               "legs (paired)",
            eps=f"final strict {AUDIT_STARTS}-start delivered-set audit "
                "= search LOWER bound of GN*; NOT a certificate (no "
                "exact meter exists at K=6, unlike K=2)",
            cv_gap_ratio="omitted: no canonical 1-D ordering of the "
                         "front at K=6 (same reason as the K=5 bandit)",
            error_bars="none: one realization per leg (MLP torch runs "
                       "are not bit-reproducible here, session 12)",
        ),
        mc=dict(samples=args.mc_samples, seed=args.mc_seed),
        reference=dict(
            union_front_size=int(union.shape[0]),
            union_front_size_central=int(union_central.shape[0]),
            ideal=[float(v) for v in ideal],
            nadir_raw=[float(v) for v in nadir_raw],
            hv_central_union_frac=float(hv_union),
        ),
        projection_joint_front_composition=dict(
            legs=[L["name"] for L in proj_legs],
            note="per objective pair: points each drawn leg contributes "
                 "to the JOINT nondominated front of that projection",
            pairs=pair_comp,
        ),
        legs=results,
    )
    (home / "front_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    # ---- figure: coverage-gap CDF (central reference) + HV bars ----
    reds = plt.get_cmap("Reds")
    rs = sorted({L["sm"]["extra"]["r"] for L in legs
                 if L["sm"]["policy"] == "baseline"})

    def leg_style(L):
        sm = L["sm"]
        if sm["policy"] == "adaptive":
            return "#2ca02c", f"adaptive (s={sm['s']})", "-"
        color = reds(0.45 + 0.5 * rs.index(sm["extra"]["r"])
                     / max(1, len(rs) - 1))
        ls = "-" if sm["s"] == 5 else "--"
        return color, f"baseline r={sm['extra']['r']} s={sm['s']}", ls

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(12.6, 5.0),
        gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    clip = 1e-4
    for L in legs:
        color, label, ls = leg_style(L)
        d = np.sort(np.maximum(L["gap_central"], clip))
        y = np.arange(1, d.size + 1) / d.size
        m = results[L["name"]]
        axl.step(d, y, where="post", color=color, ls=ls, lw=1.6,
                 label=f"{label} — IGD {m['igd_central']:.3f}")
    axl.set_xscale("log")
    axl.set_xlabel(f"distance threshold d (log; distances < {clip:g} "
                   f"clipped)")
    axl.set_ylabel("fraction of central union front within d of leg front")
    axl.set_title("coverage-gap CDF, central reference "
                  f"(all six losses <= {args.central_bound:g}; "
                  f"{'{:,}'.format(int(payload['reference']['union_front_size_central']))} pts)",
                  fontsize=9.6)
    axl.grid(True, which="both", alpha=0.25)
    axl.legend(fontsize=7.4, loc="lower right")
    axl.set_ylim(0.0, 1.02)

    xs = np.arange(len(legs))
    for i, L in enumerate(legs):
        color, label, _ls = leg_style(L)
        m = results[L["name"]]
        open_bar = L["sm"]["policy"] == "baseline" and L["sm"]["s"] != 5
        axr.bar(i, m["hv_central_frac"], width=0.62,
                color="none" if open_bar else color,
                edgecolor=color, linewidth=1.6,
                yerr=m["hv_central_ci95"], capsize=3)
    axr.axhline(hv_union, color="0.35", ls=":", lw=1.2,
                label=f"union front = {hv_union:.3f}")
    axr.set_xticks(xs)
    axr.set_xticklabels(
        [("adaptive\ns=%d" % L["sm"]["s"]) if L["sm"]["policy"] == "adaptive"
         else "r=%d\ns=%d" % (L["sm"]["extra"]["r"], L["sm"]["s"])
         for L in legs], fontsize=7.6)
    axr.set_ylabel(f"hypervolume fraction of [ideal, "
                   f"{args.central_bound:g}]^6 (MC, 95% CI)")
    axr.set_title("central-box hypervolume (open bars = s=1 "
                  "sensitivity legs)", fontsize=9.6)
    axr.grid(True, axis="y", alpha=0.25)
    axr.legend(fontsize=7.4, loc="upper right")
    budget = legs[0]["sm"]["budget"]
    fig.suptitle(
        f"Discovered epsilon-Pareto fronts at equal budget "
        f"B={budget:,.0f} — K=6 MLP, pure fixed-budget protocol\n"
        "6-D fronts admit no direct plot (K=5 bandit precedent); "
        "eps labels are strict 64-start search LOWER bounds (no "
        "certificate exists at K=6)", fontsize=9.8)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(home / "pure_budget_K6_fronts.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- projection figures: clean Pareto attainment views ----
    # staircase = the leg's attainment boundary in the projection (the
    # exact frontier of the region its delivered set dominates there);
    # shaded = that dominated region; overlap = dominated by both.
    def _leg_marker(L):
        if L["sm"]["policy"] == "adaptive":
            return "^", None
        if L["sm"]["s"] == 5:
            return "x", None
        return "o", "none"

    def _draw_pair_panel(ax, xi, yi, lw=1.1, ms=3.2):
        allP = np.concatenate([L["F"][:, [xi, yi]] for L in proj_legs],
                              axis=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(allP[:, 0].min() * 0.7, allP[:, 0].max() * 1.7)
        ax.set_ylim(allP[:, 1].min() * 0.7, allP[:, 1].max() * 1.7)
        for L in proj_legs:
            color, _lab, ls = leg_style(L)
            marker, mfc = _leg_marker(L)
            fr = _pair_front(L, xi, yi)
            ax.plot(fr[:, 0], fr[:, 1], drawstyle="steps-post", ls=ls,
                    lw=lw, marker=marker, ms=ms, color=color,
                    mfc=color if mfc is None else mfc, zorder=5)
        ax.grid(True, which="major", alpha=0.16)

    handles2 = []
    for L in proj_legs:
        color, label, ls = leg_style(L)
        marker, mfc = _leg_marker(L)
        handles2.append(plt.Line2D(
            [], [], color=color, ls=ls, lw=1.6, marker=marker, ms=5,
            mfc=color if mfc is None else mfc, label=label))

    fig2, ax2 = plt.subplots(figsize=(7.4, 6.0))
    _draw_pair_panel(ax2, 0, 1, lw=1.6, ms=4.6)
    ax2.legend(handles=handles2, fontsize=8.4, loc="upper right")
    ax2.set_xlabel("F1 — class-1 cross-entropy, full batch (log)")
    ax2.set_ylabel("F2 — class-2 cross-entropy, full batch (log)")
    ax2.set_title(
        f"Pareto fronts in the (F1, F2) projection — K=6 MLP, "
        f"pure fixed budget B={budget:,.0f}\n"
        f"staircase = attainment boundary in this projection "
        f"(6-objective run; see FRONTS.md)", fontsize=9.4)
    ax2.grid(True, which="both", alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(home / "pure_budget_K6_fronts_F1F2.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig2)

    # ---- figure 3: pairwise Pareto-attainment matrix, all 15 pairs ----
    nK = 6
    fig3, axes = plt.subplots(nK - 1, nK - 1, figsize=(12.6, 11.8))
    for row in range(nK - 1):
        for col in range(nK - 1):
            ax = axes[row][col]
            if col > row:                  # upper triangle unused
                ax.axis("off")
                continue
            xi, yi = col, row + 1          # panel = (F_{xi+1}, F_{yi+1})
            _draw_pair_panel(ax, xi, yi)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"F{yi + 1}", fontsize=9)
            if row == nK - 2:
                ax.set_xlabel(f"F{xi + 1}", fontsize=9)
    fig3.legend(handles=handles2, loc="upper right",
                bbox_to_anchor=(0.985, 0.9), fontsize=10)
    fig3.suptitle(
        f"Pairwise Pareto fronts — all 15 objective pairs (K=6 MLP, "
        f"pure fixed budget B={budget:,.0f})\n"
        f"per-class cross-entropies, log-log; staircase = attainment "
        f"boundary per projection", fontsize=11)
    fig3.tight_layout(rect=(0, 0, 1, 0.95))
    fig3.savefig(home / "pure_budget_K6_fronts_pairwise.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig3)

    # ---- FRONTS.md ----
    def row(L):
        m = results[L["name"]]
        _c, label, _ls = leg_style(L)
        return (f"| {label} | {m['n_points']:,} | {m['n_front']:,} "
                f"| {m['n_front_central']:,} "
                f"| {m['hv_central_frac']:.4f} +- {m['hv_central_ci95']:.4f} "
                f"| {m['igd_central']:.4f} | {m['maxdist_central']:.4f} "
                f"| {m['igd_to_union_raw']:.4f} "
                f"| {m['maxdist_to_union_raw']:.4f} "
                f"| {m['eps_search_lb_64start']:.4e} |")

    rows = "\n".join(row(L) for L in legs)
    per_obj = "\n".join(
        f"| {leg_style(L)[1]} | " + " | ".join(
            f"{v:.4f}" for v in results[L["name"]]["per_obj_min"]) + " |"
        for L in legs)
    md = f"""# Discovered epsilon-Pareto fronts (K=6, pure fixed budget B={budget:,.0f})

Produced by `Original_py/front_metrics_K6_pure_budget_without_256_checkpoints.py`
(July 30, 2026; change record `Note/Jul_30_note.md`).
Pure post-processing of the July-27 legs' stored `fvals` (the
full-batch objective vector of every delivered point); nothing was
re-run.  Delivered set = every segment endpoint (+ x0); front = its
nondominated subset (minimisation).

Metric set follows the SURF paper's Table 1 (HV, IGD) with the
deviations this track's record requires:

* No front FIGURE in value space: 6-D fronts admit no direct plot —
  the K=5 bandit recorded the same ("front_uniformity omitted ... no
  canonical 1-D ordering"), and the SURF paper itself only ever plots
  M = 2 fronts.  The figure here shows the coverage-gap CDF (the
  distribution whose mean is IGD and max is max-dist) and central-box
  hypervolume bars.
* Reference front = union of all legs' fronts (mutual reference; no
  oracle front exists for this family; SURF likewise omits IGD when no
  dense reference is available, their Table 2).
* CENTRAL variant: reference restricted to all six losses <=
  {cb:g} (genuine trade-off region; x0 sits at ~2.2 per objective).
  The raw union front is dominated by specialist tails (losses up to
  ~25), exactly the K=2 runner's rationale; method fronts are never
  clipped.
* HV is Monte-Carlo estimated ({args.mc_samples:,} common samples,
  seed {args.mc_seed}, paired across legs) as the dominated fraction
  of the box [ideal, {cb:g}]^6, ideal = per-objective minimum over all
  fronts.  A raw-box HV is deliberately not reported (tail-volume
  dominated).
* CV / Gap Ratio (SURF's spacing metrics) omitted — 1-D ordering
  does not exist at K=6.
* One realization per leg, no error bars across runs (MLP torch runs
  are not bit-reproducible in this environment; session-12 finding).

**eps labels are search LOWER bounds.**  Each leg's eps is its final
strict {AUDIT_STARTS}-start delivered-set audit: the best FOUND value
of GN* = max over lambda in Delta_6 of min over delivered points of
lambda' M lambda.  A search value can never sign the positive claim
"GN* <= eps" (the bandit eps1e-4 false-certificate lesson); the
certified two-sided meter exists only at K=2, where the 1-D structure
admits exact evaluation.  Quote these labels only as lower bounds.

| leg | delivered pts | front pts | front pts central | HV central (95% CI) | IGD central | max-dist central | IGD raw | max-dist raw | eps (search LB) |
|-----|---------------|-----------|-------------------|---------------------|-------------|------------------|---------|--------------|-----------------|
{rows}

Reading the RAW columns: the raw reference contains every leg's
nondominated cloud, and most of it lies in loss regions ABOVE the
shared initialization (the collapsed legs' wandering; specialist tails
up to ~25) — nondominated by construction, not genuine trade-offs.
The adaptive leg never visits those regions (GN-steered away once a
region is near-stationary), so its raw IGD/max-dist are the LARGEST
in the table by construction; distance to above-x0 clouds is not
front quality.  The central columns are the decision-relevant ones;
raw is kept only for completeness of the mutual-reference convention.

Union front: {union.shape[0]:,} pts raw, {union_central.shape[0]:,}
central; union central-box HV = {hv_union:.4f} (the attainable
envelope under this mutual reference).  Because all legs share ONE
sample set, HV differences are paired: `hv_adaptive_minus_this` in
`front_metrics.json` carries each leg's paired delta vs the adaptive
leg with its own 95% CI (tighter than the per-leg CIs suggest).

## Per-objective minimum achieved (coverage holes)

A leg that never trains an objective cannot cover that end of the
front, whatever its GN audit says.  x0 ~= 2.2 on every objective.

| leg | F1 min | F2 min | F3 min | F4 min | F5 min | F6 min |
|-----|--------|--------|--------|--------|--------|--------|
{per_obj}

## The projection figures ((F1, F2) single + all 15 pairs)

`pure_budget_K6_fronts_F1F2.png` and
`pure_budget_K6_fronts_pairwise.png` draw the two FULL-COVERAGE legs
only — adaptive and r10 s1 (the `--proj-legs` default; the other five
legs render the panels unreadable, and `--proj-legs all` restores
them).  Presentation (deliberately minimal, user request): ONLY the
two legs' fronts, each drawn as its ATTAINMENT STAIRCASE — the exact
boundary of the region its delivered set dominates in that plane
(steps-post; a straight point-to-point line would overstate what is
attained between front points).  No other overlays.  The joint-front
composition per pair (how many points of the two legs' joint
nondominated front each contributes) is NOT printed on the figures
but stays recorded in front_metrics.json under
`projection_joint_front_composition` (order adaptive / r10 s1).

Reading rules and measured context (unchanged from the 7-leg view):
fronts and dominated regions are PER PROJECTION — the four objectives
off a panel's axes are unconstrained there, and the projection front
is not the projection of the 6-D front.  The omitted legs' facts
stay on record: r10 s5 clips the (F1, F2) joint corner with one
point (0.94, 0.21); r12/r15/r20 s5's leftmost (F1, F2) front point
IS x0 (class 1 never trained); as single-class specialists they do
reach low corners in F3..F6 pairs.  The projection also compresses
the coverage story — in (F1, F2) the two drawn legs hold 8 vs 9
corner points while the 6-D record shows 215 vs 35 central front
points (IGD 0.042 vs 0.330); the 6-D table above stays the
quantitative record.

Prefix-budget front cuts are NOT possible from the July-27 artifacts
(no per-segment grad ledger was stored at K=6; the K=2 runner added
`seg_grads`/`seg_lams` for exactly that) — this analysis is
final-budget only.
"""
    (home / "FRONTS.md").write_text(md, encoding="utf-8")
    print(f"FRONTS.md + front_metrics.json + pure_budget_K6_fronts.png "
          f"-> {home}", flush=True)


if __name__ == "__main__":
    main()
