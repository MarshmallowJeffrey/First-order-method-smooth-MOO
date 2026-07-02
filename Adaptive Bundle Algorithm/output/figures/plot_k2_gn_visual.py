from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("output/figures/k2_gn_lower_envelope.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# A one-dimensional decision-space illustration.  At bundle point i, choose
# grad F_1(x_i) = 1-c_i and grad F_2(x_i) = -c_i.  Then
# alpha grad F_1 + (1-alpha) grad F_2 = alpha-c_i, so q_i=(alpha-c_i)^2.
centers = np.array([0.05, 0.50, 0.90])
alpha = np.linspace(0.0, 1.0, 4001)
q = np.array([(alpha - c) ** 2 for c in centers])
gn_before = q.min(axis=0)

idx_before = int(np.argmax(gn_before))
alpha_star = float(alpha[idx_before])
gn_star = float(gn_before[idx_before])

# Illustrate an inner solve that adds a point well suited to alpha_star.
q_new = (alpha - alpha_star) ** 2
gn_after = np.minimum(gn_before, q_new)
idx_after = int(np.argmax(gn_after))

epsilon = 0.06

# Reproduce the random-sample idea for K=2: Dirichlet(1,1) means alpha is
# uniform on [0,1].  Keep the sample count small enough to remain visible.
rng = np.random.RandomState(7)
sample_alpha = rng.dirichlet(np.ones(2), size=30)[:, 0]
structured = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
sample_alpha = np.unique(np.concatenate([structured, sample_alpha]))
sample_values = np.interp(sample_alpha, alpha, gn_before)
sample_best = int(np.argmax(sample_values))

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)

# Panel 1: individual quadratics and their lower envelope.
ax = axes[0]
colors = ["#4C78A8", "#F58518", "#54A24B"]
for i, (curve, c, color) in enumerate(zip(q, centers, colors), start=1):
    ax.plot(alpha, curve, color=color, lw=1.5, alpha=0.7,
            label=rf"$q_{i}(\alpha)=(\alpha-{c:.2f})^2$")
ax.plot(alpha, gn_before, color="black", lw=3.0,
        label=r"$GN(\alpha;\mathcal{B})=\min_i q_i(\alpha)$")
ax.scatter([alpha_star], [gn_star], marker="*", s=190, color="#E45756",
           edgecolor="black", linewidth=0.6, zorder=6)
ax.annotate(rf"worst covered $\alpha_t={alpha_star:.3f}$" + "\n" +
            rf"$GN^*={gn_star:.4f}$",
            xy=(alpha_star, gn_star), xytext=(0.34, 0.073),
            arrowprops=dict(arrowstyle="->", color="#E45756", lw=1.4),
            fontsize=10)
ax.set_title("1. Quadratics and the lower envelope")
ax.set_xlabel(r"$\alpha$ in $\lambda=(\alpha,1-\alpha)$")
ax.set_ylabel("squared gradient norm")
ax.set_xlim(0, 1)
ax.set_ylim(-0.003, 0.105)
ax.grid(alpha=0.2)
ax.legend(loc="upper center", fontsize=8.5, frameon=True)

# Panel 2: how Sample and multi-start IPOPT search the same envelope.
ax = axes[1]
ax.plot(alpha, gn_before, color="black", lw=2.8, label="GN lower envelope")
ax.scatter(sample_alpha, sample_values, s=26, color="#4C78A8", alpha=0.8,
           label="Sample candidates")
ax.scatter([sample_alpha[sample_best]], [sample_values[sample_best]],
           marker="D", s=90, color="#E45756", edgecolor="black", linewidth=0.5,
           zorder=5, label="best sampled candidate")
ax.scatter([alpha_star], [gn_star], marker="*", s=180, color="#B279A2",
           edgecolor="black", linewidth=0.5, zorder=6,
           label="best multi-start local optimum")

# Schematic local-ascent basins for IPOPT starts.
starts_and_targets = [(0.10, 0.275), (0.36, 0.275), (0.61, 0.700), (0.82, 0.700)]
for start, target in starts_and_targets:
    y_start = float(np.interp(start, alpha, gn_before))
    y_target = float(np.interp(target, alpha, gn_before))
    ax.annotate("", xy=(target, y_target), xytext=(start, y_start),
                arrowprops=dict(arrowstyle="->", color="#B279A2", alpha=0.65,
                                lw=1.1, connectionstyle="arc3,rad=0.15"))
ax.text(0.52, 0.078, "IPOPT: local refinement\nfrom several starts", color="#7C4D73",
        fontsize=9.5, ha="center")
ax.set_title("2. Two approximations to the outer max")
ax.set_xlabel(r"$\alpha$")
ax.set_xlim(0, 1)
ax.set_ylim(-0.003, 0.105)
ax.grid(alpha=0.2)
ax.legend(loc="upper left", fontsize=8.2, frameon=True)

# Panel 3: an inner update depresses the selected peak; thresholds are shown.
ax = axes[2]
ax.plot(alpha, gn_before, color="black", lw=2.2, alpha=0.55,
        label="before inner update")
ax.plot(alpha, q_new, color="#F58518", lw=1.7, ls="--",
        label=r"new $q_{new}(\alpha)$")
ax.plot(alpha, gn_after, color="#4C78A8", lw=3.0,
        label=r"after: $\min(GN,q_{new})$")
ax.axvline(alpha_star, color="#E45756", lw=1.2, ls=":")
ax.scatter([alpha_star], [np.interp(alpha_star, alpha, gn_after)],
           color="#E45756", s=60, zorder=5)
ax.scatter([alpha[idx_after]], [gn_after[idx_after]], marker="*", s=150,
           color="#4C78A8", edgecolor="black", linewidth=0.5, zorder=5)

thresholds = [
    (epsilon / 3, r"$\epsilon/3=0.020$", "#54A24B"),
    (2 * epsilon / 3, r"$2\epsilon/3=0.040$", "#B279A2"),
    (epsilon, r"$\epsilon=0.060$", "#E45756"),
]
for value, label, color in thresholds:
    ax.axhline(value, color=color, lw=1.25, ls="--", alpha=0.85)
    ax.text(0.995, value + 0.001, label, color=color, fontsize=8.7,
            ha="right", va="bottom")
ax.annotate("inner target at selected weight",
            xy=(alpha_star, 0.0), xytext=(0.06, 0.010),
            arrowprops=dict(arrowstyle="->", color="#E45756"), fontsize=9)
ax.annotate(rf"next worst weight $\alpha\approx{alpha[idx_after]:.2f}$",
            xy=(alpha[idx_after], gn_after[idx_after]), xytext=(0.49, 0.071),
            arrowprops=dict(arrowstyle="->", color="#4C78A8"), fontsize=9)
ax.set_title("3. Inner update presses down one peak")
ax.set_xlabel(r"$\alpha$")
ax.set_xlim(0, 1)
ax.set_ylim(-0.003, 0.105)
ax.grid(alpha=0.2)
ax.legend(loc="upper left", fontsize=8.4, frameon=True)

fig.suptitle(
    r"Toy $K=2$ example: $\nabla F_1(x_i)=1-c_i$, "
    r"$\nabla F_2(x_i)=-c_i$, hence $q_i(\alpha)=(\alpha-c_i)^2$",
    fontsize=13,
)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(OUT.resolve())
