"""Settings of the reported runs."""
import numpy as np

# Empirical smoothness estimates L_k of F_k (finite-difference directional curvature at points around theta_0, the
# largest value found); they enter only the starting-point rule of GRAB and Uniform.
L_ESTIMATES = {
    "fishwood": [0.240111632014359, 0.34796415276597487],
    "fruittree_d6": [0.13867360882755067, 0.1386705991183074, 0.1386719770172668,
                     0.13867074332994217, 0.13867109068783343, 0.1386744703018394],
}

# Run-to-plateau rule (mogym.plateau); SURF additionally requires its slot weights to have settled.  The rule is
# checked after every block of sweeps (Uniform) or rounds (SURF) that gives each weight at least check_steps Adam
# steps, so that its window covers the same optimization progress whatever the steps per sweep or round.
RULE = dict(tol=.01, window=3, confirm=3, own_ratio=.25, own_floor=1e-5, check_steps=50)
SURF_RULE = dict(RULE, weight_tol=.005)
MAX_SWEEPS = 50000          # Uniform safety cap
MAX_ROUNDS = 1000           # SURF safety cap
STATE_TOL = .005            # an Adam state is continued while its weight moves by at most this (every coordinate)
POINT_BAND = .05            # plotted point: earliest checkpoint after which the GN stays within 5% of its value at stopping

# One budget B per task, the same for all methods: 1.05 x the Gradient Calls of the farthest Uniform point (FishWood
# r=512, Fruit Tree r=6), rounded up to a multiple of 1e3 (FishWood) or 6e3 (Fruit Tree).  Checkpoints of every
# method: after every B/600 (K=2) or B/120 (K=6) Gradient Calls up to B, 10x that afterwards.  A run is plotted if it
# reached its plateau and its point lies within B.  SURF: the doubling values N = 2, 4, ... whose point lies within B.
# Step counts and learning rates were chosen once per method from M in {5, 10, 25, 50} x lr in {1e-3, 3e-3, 0.01,
# 0.03, 0.1, 0.3} by one rule (pilot runs to a fixed budget, scored by the mean log GN over Gradient Calls and CPU
# time); for Fruit Tree the CCP options of GRAB were then compared in the same way.
TASKS = {
    "fishwood": dict(
        uniform=dict(M=5, lr=.03, values=[2, 4, 8, 16, 32, 64, 128, 256, 512]),
        surf=dict(K_S=25, lr=.03, values=[2, 4, 8, 16, 32, 64, 128, 256]),
        adaptive=dict(inner_steps=5, lr=.003, checkpoint_count=600),
        budget=157000, every=261),
    "fruittree_d6": dict(
        uniform=dict(M=5, lr=.1, values=[1, 2, 3, 4, 5, 6]),
        adaptive=dict(inner_steps=5, lr=.1, checkpoint_count=120,
                      ccp=dict(nseeds=64, nstarts=1, maxiter=3, boundary_resolution=20, keep_pool=16,
                               lazy=100, lazy_rho=.5, screen=True, presolve=False, warm_cg=True, relaxed=True)),
        budget=60000, every=500),
}

# K>2: upper bounds on GRAB's worst-case gradient norm at every checkpoint (mogym.bounds): seconds per checkpoint,
# relative gap to the lower estimate at which a checkpoint stops, worker processes, cap on stored subsimplices.
BOUNDS = dict(seconds=20, gap=.01, workers=6, max_leaves=1_500_000)


def smoothness(name):
    return np.asarray(L_ESTIMATES[name], float)
