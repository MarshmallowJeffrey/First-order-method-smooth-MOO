"""Settings of the reported runs."""
import numpy as np

# Empirical smoothness estimates L_k of F_k (finite-difference directional curvature at points around theta_0, the
# largest value found); they enter only the starting-point rule of GRAB and Uniform.
L_ESTIMATES = {
    "fishwood": [0.240111632014359, 0.34796415276597487],
    "fruittree_d6": [0.13867360882755067, 0.1386705991183074, 0.1386719770172668,
                     0.13867074332994217, 0.13867109068783343, 0.1386744703018394],
}

# Run-to-plateau rule (mogym.plateau); SURF additionally requires its slot weights to have settled.
RULE = dict(tol=.01, window=3, confirm=3, own_ratio=.25, own_floor=1e-5)
SURF_RULE = dict(RULE, weight_tol=.005)
MAX_SWEEPS = 50000          # Uniform safety cap
MAX_ROUNDS = 1000           # SURF safety cap
STATE_TOL = .005            # an Adam state is continued while its weight moves by at most this (every coordinate)
POINT_BAND = .05            # plotted point: earliest checkpoint after which the GN stays within 5% of its value at stopping

# One budget B per task, the same for all methods: 1.05 x the Gradient Calls of the farthest Uniform point (FishWood
# r=512, Fruit Tree r=6), rounded up to a multiple of 1e3 (FishWood) or 6e3 (Fruit Tree).  Checkpoints of every
# method: after every B/600 (K=2) or B/120 (K=6) Gradient Calls up to B, 10x that afterwards.  A run is plotted if it
# reached its plateau and its point lies within B.  SURF: N = 2, 4, ..., 128 and N = 208, the largest multiple of 16
# whose point lies within B.
TASKS = {
    "fishwood": dict(
        uniform=dict(M=50, lr=.03, values=[2, 4, 8, 16, 32, 64, 128, 256, 512]),
        surf=dict(K_S=25, lr=.03, values=[2, 4, 8, 16, 32, 64, 128, 208]),
        adaptive=dict(inner_steps=1, lr=.01, checkpoint_count=600),
        budget=108000, every=180),
    "fruittree_d6": dict(
        uniform=dict(M=5, lr=.1, values=[1, 2, 3, 4, 5, 6]),
        adaptive=dict(inner_steps=5, lr=.1, checkpoint_count=120,
                      ccp=dict(nseeds=64, nstarts=1, maxiter=15, boundary_resolution=20, keep_pool=16)),
        budget=48000, every=400),
}


# Relaxed-LP variant, for review (scripts: --variant relaxed_lp; run_all_relaxed.sh): Fruit Tree GRAB with every LP of
# the CCP solved once on the constraint-generation working set (mogym.lambda_solvers) and M_A = 2, the setting chosen
# by the same selection rule with this CCP.  Uniform and the evaluator are unchanged.
VARIANTS = {
    "relaxed_lp": {"fruittree_d6": dict(adaptive=dict(
        inner_steps=2, lr=.1, checkpoint_count=120,
        ccp=dict(nseeds=64, nstarts=1, maxiter=15, boundary_resolution=20, keep_pool=16, exact_lp=False)))},
}


def apply_variant(name):
    """Replace the settings of the variant in TASKS (in place)."""
    for task, settings in VARIANTS[name].items():
        TASKS[task].update(settings)


def smoothness(name):
    return np.asarray(L_ESTIMATES[name], float)
