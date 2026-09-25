#!/usr/bin/env python
"""The tables of Section 4.1 and Appendix C.1 as LaTeX tabulars (booktabs), from results/:

    tables/mnist_main.tex         best configuration of each baseline vs the adaptive method
    tables/mnist_k2_full.tex      {4,9}: all configurations
    tables/mnist_k3_full.tex      {4,7,9}: all configurations
    tables/step_rules_k2.tex      step-rule experiment
    tables/screening_pairs.tex    screening, top six pairs
    tables/screening_triples.tex  screening, top three triples

    python scripts/make_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm import config as C  # noqa: E402

RESULTS, TABLES = ROOT / "results", ROOT / "tables"
DAGGER = "{}^{\\dagger}"
STEP_RULE_NAME = {"const": "Constant step (incumbent)", "bb": "Barzilai--Borwein"}
for _m in (1, 3, 10):
    STEP_RULE_NAME[f"adagrad_mult{_m}"] = f"AdaGrad ($\\alpha_{{\\mathrm{{mult}}}}={_m}$)"
for _al, _als in (("0.001", "10^{-3}"), ("0.0003", "3\\times10^{-4}"), ("0.0001", "10^{-4}")):
    for _b2 in ("0.9", "0.99"):
        STEP_RULE_NAME[f"adam_alpha{_al}_beta2{_b2}"] = f"Adam ($\\alpha={_als}$, $\\beta_2={_b2}$)"


def sci(v, dagger=False):
    mant, exp = f"{v:.2e}".split("e")
    body = mant if int(exp) == 0 else f"{mant}\\times10^{{{int(exp)}}}"
    return f"${body}{DAGGER if dagger else ''}$"


def thousands(v):
    return format(int(round(v)), ",").replace(",", "{,}")


def write(name, lines):
    TABLES.mkdir(exist_ok=True)
    (TABLES / name).write_text("\n".join(lines) + "\n")
    print("saved", TABLES / name)


def main_table(res):
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Method & $\\max_{\\lambda\\in\\Delta_K}\\mathrm{GN}(\\lambda,B_t)$ & Ratio \\\\", "\\midrule"]
    for K in (2, 3):
        r = res[K]
        stats = {(s["method"], s["param"]): s for s in r["configs"]}
        ad = r["adaptive_final_geomean"]
        digits = ",".join(str(d) for d in C.DIGITS[K])
        if K == 3:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{MNIST $\\{{{digits}\\}}$, $K={K}$}}}} \\\\")
        lines.append(f"Adaptive Bundle Method & {sci(ad)} & -- \\\\")
        for fam, p in C.BEST[K].items():
            y = stats[(fam, p)]["y_geomean"]
            label = f"Unif Discrtztn ($r={p}$)" if fam == "uniform" else f"SURF ($N={p}$)"
            lines.append(f"{label} & {sci(y)} & ${y / ad:.1f}\\times$ \\\\")
    write("mnist_main.tex", lines + ["\\bottomrule", "\\end{tabular}"])


def full_k2(res):
    stats = {(s["method"], s["param"]): s for s in res["configs"]}
    params = sorted({p for _, p in stats})

    def cells(fam, p):
        if (fam, p) not in stats:
            return "-- & -- & --"
        s = stats[(fam, p)]
        drawn = p in (C.FIGURE_UNIFORM_R[2] if fam == "uniform" else C.FIGURE_SURF_N)
        return (f"{sci(s['y_geomean'], dagger=not drawn)} & {thousands(s['x_geomean'])} / {thousands(s['x_median'])} "
                f"& {s['n_plateau']}/{len(s['seeds'])}")
    lines = ["\\begin{tabular}{r ccc ccc}", "\\toprule",
             " & \\multicolumn{3}{c}{Unif Discrtztn ($r$)} & \\multicolumn{3}{c}{SURF ($N$)} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "$r$ or $N$ & $\\max\\mathrm{GN}$ & $x$ & plateau & $\\max\\mathrm{GN}$ & $x$ & plateau \\\\", "\\midrule"]
    lines += [f"{p} & {cells('uniform', p)} & {cells('surf', p)} \\\\" for p in params]
    write("mnist_k2_full.tex", lines + ["\\bottomrule", "\\end{tabular}"])


def full_k3(res):
    lines = ["\\begin{tabular}{rrccc}", "\\toprule", "$r$ & nodes & $\\max\\mathrm{GN}$ & $x$ & plateau \\\\", "\\midrule"]
    for s in sorted(res["configs"], key=lambda s: s["param"]):
        r = s["param"]
        lines.append(f"{r} & {(r + 1) * (r + 2) // 2} & {sci(s['y_geomean'], dagger=r not in C.FIGURE_UNIFORM_R[3])} & "
                     f"{thousands(s['x_geomean'])} / {thousands(s['x_median'])} & {s['n_plateau']}/{len(s['seeds'])} \\\\")
    write("mnist_k3_full.tex", lines + ["\\bottomrule", "\\end{tabular}"])


def step_rules(res):
    lines = ["\\begin{tabular}{rlcc}", "\\toprule", "Rank & Step rule & Mean & Seed range \\\\", "\\midrule"]
    for i, tag in enumerate(res["ranking"], 1):
        f = np.asarray(res["rules"][tag]["final_per_seed"]) * 1e3
        lines.append(f"{i} & {STEP_RULE_NAME[tag]} & {f.mean():.2f} & {f.min():.2f}--{f.max():.2f} \\\\")
    write("step_rules_k2.tex", lines + ["\\bottomrule", "\\end{tabular}"])


def screening(pairs, triples):
    lines = ["\\begin{tabular}{rlcc}", "\\toprule", "Rank & Pair & $C_{\\mathrm{bal}}$ & $C_{\\mathrm{mean}}$ \\\\", "\\midrule"]
    for i, r in enumerate(pairs[:6], 1):
        lines.append(f"{i} & $\\{{{r['digits'][0]},{r['digits'][1]}\\}}$ & {r['C_bal']:.3f} & {r['C_mean']:.3f} \\\\")
    write("screening_pairs.tex", lines + ["\\bottomrule", "\\end{tabular}"])
    lines = ["\\begin{tabular}{rlcc}", "\\toprule", "Rank & Triple & $C_{\\mathrm{bal}}$ & $c_j$ \\\\", "\\midrule"]
    for i, r in enumerate(triples[:3], 1):
        d = ",".join(str(v) for v in r["digits"])
        lines.append(f"{i} & $\\{{{d}\\}}$ & {r['C_bal']:.3f} & " + " / ".join(f"{c:.2f}" for c in r["c_j"]) + " \\\\")
    write("screening_triples.tex", lines + ["\\bottomrule", "\\end{tabular}"])


def main():
    res = {K: json.loads((RESULTS / f"k{K}.json").read_text()) for K in (2, 3)}
    main_table(res)
    full_k2(res[2])
    full_k3(res[3])
    step_rules(json.loads((RESULTS / "step_rules_k2.json").read_text()))
    screening(json.loads((RESULTS / "screening_k2.json").read_text()),
              json.loads((RESULTS / "screening_k3.json").read_text()))


if __name__ == "__main__":
    main()
