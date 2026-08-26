"""Corrected flowchart: Momentum-SVRG certified baseline (specific flow).

Redraws the hand-made raster figure for
``Original_py/baseline_svrg_certified_without_256_checkpoints.py``
(share_mode="gram", the default) with the arrow topology fixed to match
the code:

* entering a node starts at segment INITIALISATION (not at the inner
  exit test);
* the descent-safeguard retry returns to INITIALISATION from the same
  anchor (not to the full joint evaluation);
* "segments < max_segments" starts a NEW segment at INITIALISATION with
  g_a, F_a recomputed at the advanced anchor (g_a/F_a therefore live in
  the init box, not in the per-node box);
* retry exhaustion is drawn: the violating endpoint is still accepted
  as the new anchor (code: the while-loop breaks and anchor <- y).

Bilingual: ``python momentum_svrg_baseline_flowchart.py`` renders the
Chinese version, ``python momentum_svrg_baseline_flowchart.py en`` the
English one (suffix ``_en``).  ``L(zh, en)`` picks any per-language
value (string, coordinate, fontsize).

Rendering notes: a line given as a LIST mixes mathtext and plain
segments (mathtext cannot font-fall-back to CJK); segments are measured
with the Agg renderer and centred as a group.  CJK is never set bold
(no bold face in the fallback fonts) — headers use a stroke
path-effect instead.

Output: momentum_svrg_baseline_flowchart[_en].png (same folder), 2x dpi.
"""

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS",
                                   "Hiragino Sans GB"]
plt.rcParams["mathtext.fontset"] = "dejavusans"

EN = len(sys.argv) > 1 and sys.argv[1] == "en"


def L(zh, en):
    """Pick the per-language value (works for strings, numbers, ...)."""
    return en if EN else zh


BLUE = "#1848c8"
ORANGE = "#ef7d0d"
GREEN = "#157a33"
RED = "#df1c1c"
BLACK = "#111111"

# Explicit per-text family list: rcParams fallback alone does not kick
# in for CJK here, an explicit list does (glyph-level fallback).
FAMILY = ["DejaVu Sans", "Arial Unicode MS", "Hiragino Sans GB"]

W, H = 1536, 1140
fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)          # y grows downward, like the original raster
ax.axis("off")
fig.patch.set_facecolor("white")
renderer = fig.canvas.get_renderer()


def put_line(cx, y, line, fs, color=BLACK):
    """One centred line; ``line`` is a str or a list of segments."""
    segs = [line] if isinstance(line, str) else line
    ts = [ax.text(0, y, s, ha="left", va="center", fontsize=fs,
                  color=color, zorder=4, fontfamily=FAMILY)
          for s in segs]
    widths = [t.get_window_extent(renderer).width for t in ts]
    x = cx - sum(widths) / 2.0            # 1 data unit == 1 px at dpi 100
    for t, w in zip(ts, widths):
        t.set_position((x, y))
        x += w


def box(cx, cy, w, h, lines, ec, fs=12.5, lw=2.0, pad=6, fc="white"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2.0, cy - h / 2.0), w, h,
        boxstyle=f"round,pad=0,rounding_size={pad}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    n = len(lines)
    step = h / (n + 0.8) if n > 1 else 0.0
    y0 = cy - step * (n - 1) / 2.0
    for i, ln in enumerate(lines):
        put_line(cx, y0 + i * step, ln, fs)


def diamond(cx, cy, w, h, lines, ec, fs=12.5, lw=2.0):
    ax.add_patch(Polygon(
        [(cx, cy - h / 2.0), (cx + w / 2.0, cy),
         (cx, cy + h / 2.0), (cx - w / 2.0, cy)],
        closed=True, linewidth=lw, edgecolor=ec, facecolor="white",
        zorder=3))
    n = len(lines)
    step = min(24.0, h / (n + 1.2)) if n > 1 else 0.0
    y0 = cy - step * (n - 1) / 2.0
    for i, ln in enumerate(lines):
        put_line(cx, y0 + i * step, ln, fs)


def arrow(pts, color, lw=2.0):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                solid_capstyle="round", zorder=2)
    ax.add_patch(FancyArrowPatch(
        pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color=color, zorder=2, shrinkA=0, shrinkB=0))


def text(x, y, s, color, fs=12, ha="center", bold=False, boxed=False):
    t = ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color,
                zorder=5, fontfamily=FAMILY)
    if bold:
        t.set_path_effects([pe.withStroke(linewidth=0.8,
                                          foreground=color)])
    if boxed:
        t.set_bbox(dict(facecolor="white", edgecolor="none", pad=1.5))


TOP = r"^{\top}"

# ---------------------------------------------------------------- title
text(W / 2, 30, L("Momentum-SVRG Baseline：具体流程",
                  "Momentum-SVRG Baseline: Detailed Flow"),
     BLACK, 21, bold=True)
text(190, 74, L("外层 λ 网格循环", "Outer λ-grid loop"), BLUE, 15,
     bold=True)
text(770, 74, L("一个 Momentum-SVRG segment",
                "One Momentum-SVRG segment"), ORANGE, 15, bold=True)
text(1170, 74, L("精确全梯度检查与链推进",
                 "Exact full-gradient check & chain advance"), GREEN, 15,
     bold=True)

# ---------------------------------------------------------- left column
box(190, 120, 120, 42, ["Start"], BLACK, fs=14, pad=16)
box(190, 196, 264, 56, [[L("离散化 ", "Discretise "), r"$\Delta_K$",
                         L("：", ":")],
                        "uniform grid + snake order"], BLUE)
box(190, 292, 276, 72, [["Full evaluate ", "$x_0$",
                         L("（不计费）：", " (uncharged):")],
                        [rf"$f_0,\ J_0,\ G_0=J_0J_0{TOP}$", L("；", ";")],
                        r"delivered$=\{x_0\}$"], BLUE)
box(190, 400, 286, 72, [L("Gram sweep：", "Gram sweep:"),
                        [rf"$\mathrm{{best}}_i=\min(\mathrm{{best}}_i,\ "
                         rf"\lambda_i{TOP}G_0\lambda_i)$", L("；", ";")],
                        r"$\leq \mathrm{node\_tol}\Rightarrow$ served"],
    BLUE)
diamond(190, 530, 214, 96, [L("还有未服务 λ？", "Any unserved λ?")], RED)
box(190, 700, 292, 72, [L("最终 strict continuous λ-search",
                          "Final strict continuous λ-search"),
                        L("（64 starts）", "(64 starts)"),
                        "+ certificate verification"], BLUE)
box(190, 800, 120, 42, ["End"], BLACK, fs=14, pad=16)

box(455, 530, 176, 84, [L("取下一个未服务 λ；", "Take next unserved λ;"),
                        "a = chain point",
                        L("（ℓ₁ 相邻 warm start）",
                          "(ℓ₁-adjacent warm start)")], BLUE,
    fs=L(12, 11.5))

arrow([(190, 141), (190, 168)], BLACK)
arrow([(190, 224), (190, 256)], BLUE)
arrow([(190, 328), (190, 364)], BLUE)
arrow([(190, 436), (190, 482)], BLUE)
arrow([(297, 530), (367, 530)], BLUE)                     # Yes -> next node
text(318, 514, "Yes", BLUE, 12, bold=True)
arrow([(83, 530), (83, 664)], BLUE)                       # No -> final search
text(66, 592, "No", BLUE, 12, bold=True)
arrow([(190, 736), (190, 779)], BLUE)

# ------------------------------------------------- orange segment frame
ax.add_patch(FancyBboxPatch(
    (585, 118), 375, 622, boxstyle="round,pad=0,rounding_size=14",
    linewidth=2.4, edgecolor=ORANGE, facecolor="none", zorder=1))

box(772, 185, 346, 92,
    [L("初始化（每个 segment / 每次重跑）：",
       "Init (per segment / per retry):"),
     [rf"$g_a=J(a){TOP}\lambda,\ \ F_a=f(a){TOP}\lambda$",
      L("（缓存）；", " (cached);")],
     r"$y=a,\ u=0,\ \eta=\mathrm{step\_const}/((\lambda^{\top}L)"
     r"\cdot L_{\mathrm{scale}})$"],
    ORANGE, fs=11.5)
box(772, 320, 282, 56, [[L("同一分层 minibatch ",
                           "Same stratified minibatch "), "$S_t$",
                         L("；", ";")],
                        [r"$\lambda_k=0$",
                         L(" 的数据行跳过", " rows are skipped")]], ORANGE)
box(772, 410, 282, 46,
    [r"$v_t=\hat{g}(y;S_t)-\hat{g}(a;S_t)+g_a$"], ORANGE, fs=13)
box(772, 488, 250, 46,
    [r"$u\leftarrow \beta u+v_t,\ \ y\leftarrow y-\eta u$"], ORANGE,
    fs=13)
diamond(772, 612, 306, 112,
        [[r"$t=\mathrm{epoch\_len}$", L("，或", ", or")],
         r"$\|v_t\|^2\leq\rho\cdot\mathrm{solve\_target}$",
         L("连续 patience 次？", "patience consecutive times?")], RED,
        fs=11.5)

arrow([(772, 231), (772, 292)], ORANGE)
arrow([(772, 348), (772, 387)], ORANGE)
arrow([(772, 433), (772, 465)], ORANGE)
arrow([(772, 511), (772, 556)], ORANGE)
arrow([(925, 612), (943, 612), (943, 320), (913, 320)], ORANGE)   # No loop
text(957, 596, "No", ORANGE, 12, bold=True)

# 入口：新节点从初始化开始（修正 1）
arrow([(543, 530), (562, 530), (562, 185), (602, 185)], BLUE)

# 触发 -> full check（只提前检查，不是证书）
arrow([(772, 668), (772, 702), (985, 702), (985, 148), (1016, 148)], GREEN)
text(786, 690, "Yes", GREEN, 12, bold=True)
text(L(866, 815), 724, L("仅触发 full check；不是证书",
                         "only triggers full check; not a certificate"),
     RED, L(11.5, 10))

# ---------------------------------------------------------- right column
box(1170, 148, 308, 58, [L("Full joint evaluation（计费）：",
                           "Full joint evaluation (charged):"),
                         rf"$f(y),\ J(y),\ G(y)=J(y)J(y){TOP}$"], GREEN)
box(1170, 258, 312, 74,
    L([["ALWAYS deliver ", "$y$", "；对所有未服务 ", r"$\lambda_i$"],
       ["做 Gram sweep：",
        rf"$\lambda_i{TOP}G(y)\lambda_i\leq\mathrm{{node\_tol}}$"],
       r"$\Rightarrow$ served"],
      [["ALWAYS deliver ", "$y$", "; Gram sweep over"],
       ["all unserved ", r"$\lambda_i$", ":  ",
        rf"$\lambda_i{TOP}G(y)\lambda_i\leq\mathrm{{node\_tol}}$"],
       r"$\Rightarrow$ served"]), GREEN)
diamond(1170, 388, 290, 116,
        [L("下降违反？", "Descent violated?"),
         r"$F_\lambda(y)>F_\lambda(a)\,+$",
         r"$10^{-10}(1+|F_\lambda(a)|)$"], RED, fs=11.5)
box(1430, 388, 200, 112,
    [[r"$L_{\mathrm{scale}}\leftarrow 2L_{\mathrm{scale}}$",
      L("；动量清零；", "; zero momentum;")],
     L("同一 anchor a 重跑", "re-run from same anchor a"),
     L("（≤ max_segment_retries）；", "(≤ max_segment_retries);"),
     L("失败点仍保留在 delivered set",
       "failed point stays delivered")], RED, fs=L(9.5, 9))
box(1170, 498, 314, 56,
    [L("接受 segment 终点：", "Accept segment endpoint:"),
     rf"$a\leftarrow y,\ J(a)\leftarrow J(y),\ "
     rf"F_a\leftarrow F_\lambda(y)$"], GREEN)
diamond(1170, 598, 324, 88,
        [rf"$q_\lambda(y)=\lambda{TOP}G(y)\lambda$",
         [r"$\leq\mathrm{solve\_target}$", L("？", "?")]], GREEN, fs=12)
box(1450, 598, 152, 60, ["node solved;", "chain ← a"], GREEN)
diamond(1170, 706, 264, 84,
        [r"$\mathrm{segments}<$",
         [r"$\mathrm{max\_segments}$", L("？", "?")]], GREEN, fs=12)
diamond(1170, 812, 288, 84,
        [[r"$\mathrm{best\_val}(\lambda)\leq\mathrm{node\_tol}$",
          L("？", "?")]], GREEN, fs=12)
box(920, 812, 176, 44, ["served_above_target"], GREEN, fs=12)
box(1440, 812, 116, 44, ["censored"], GREEN, fs=12)
box(1170, 900, 132, 44, ["chain ← a"], GREEN, fs=12.5)

arrow([(1170, 177), (1170, 221)], GREEN)
arrow([(1170, 295), (1170, 330)], GREEN)
arrow([(1315, 388), (1330, 388)], RED)                    # 违反 Yes
text(1322, 371, "Yes", RED, 12, bold=True)
arrow([(1170, 446), (1170, 470)], GREEN)                  # 违反 No
text(1184, 456, "No", GREEN, 12, bold=True)

# 安全保护重跑：回到初始化（修正 2）
arrow([(1430, 332), (1430, 94), (740, 94), (740, 139)], RED)
# 重试用尽：仍接受该终点（新增，修正 4）
arrow([(1430, 444), (1430, 498), (1329, 498)], RED)
text(L(1442, 1375), L(472, 486), L("重试用尽", "retries exhausted"),
     RED, 11, ha=L("left", "center"))

arrow([(1170, 526), (1170, 554)], GREEN)
arrow([(1332, 598), (1372, 598)], GREEN)                  # q Yes
text(1350, 582, "Yes", GREEN, 12, bold=True)
arrow([(1170, 642), (1170, 664)], GREEN)                  # q No
text(1184, 652, "No", GREEN, 12, bold=True)

# 新 segment：anchor 已推进，回到初始化重算 g_a, F_a（修正 3）
arrow([(1038, 706), (1002, 706), (1002, 112), (838, 112), (838, 139)],
      GREEN)
text(1020, 690, "Yes", GREEN, 12, bold=True)
text(1020, 758, L("新 segment（anchor 已推进）",
                  "new segment (anchor advanced)"), GREEN, 10.5)

arrow([(1170, 748), (1170, 770)], GREEN)
text(1184, 758, "No", GREEN, 12, bold=True)
arrow([(1026, 812), (1008, 812)], GREEN)                  # best Yes
text(1017, 796, "Yes", GREEN, 12, bold=True)
arrow([(1314, 812), (1382, 812)], GREEN)
text(1346, 796, "No", GREEN, 12, bold=True)
arrow([(920, 834), (920, 900), (1104, 900)], GREEN)
arrow([(1440, 834), (1440, 900), (1236, 900)], GREEN)

# ------------------------------------------------ blue return to outer
arrow([(1450, 628), (1450, 656), (1512, 656), (1512, 960), (350, 960),
       (350, 640), (190, 640), (190, 580)], BLUE)
arrow([(1170, 922), (1170, 960)], BLUE)

# ---------------------------------------------------------------- legend
ax.add_patch(FancyBboxPatch(
    (60, 1000), 1430, 108, boxstyle="round,pad=0,rounding_size=10",
    linewidth=1.6, edgecolor=BLACK, facecolor="none", zorder=1,
    linestyle=(0, (5, 4))))
box(250, 1054, 330, 72,
    [[L("随机触发：", "Stochastic trigger: "),
      r"$\|v_t\|^2\leq\rho\cdot\mathrm{solve\_target}$"],
     L("（只提前 full check）", "(only advances the full check)")],
    ORANGE, fs=L(12, 11))
box(630, 1054, 320, 72,
    [L("精确内层完成：", "Exact inner completion:"),
     rf"$\lambda{TOP}G(y)\lambda\leq\mathrm{{solve\_target}}$"],
    GREEN, fs=12)
box(1010, 1054, 330, 72,
    [L("精确服务证书：", "Exact service certificate:"),
     rf"$\min_{{\mathrm{{delivered}}}}\ \lambda{TOP}G\lambda\leq"
     r"\mathrm{node\_tol}$"], BLUE, fs=12)
box(1355, 1054, 240, 72,
    [L("默认", "Default"),
     r"$\mathrm{solve\_target}=\mathrm{node\_tol}/4$"], BLACK, fs=12)

out = __file__.replace(".py", "_en.png" if EN else ".png")
fig.savefig(out, dpi=200, facecolor="white")
print("saved", out)
