"""Settings of the reported runs (paper Table "Selected settings").

The learning rates and step counts were chosen by the tuning procedure described in the paper appendix
(pilot budget 5e4 Gradient Calls, 1.5e5 for Fruit Tree); only the resulting values are listed here.
"""
import numpy as np

# Empirical directional smoothness estimates L_k of F_k; they enter only the adaptive warm-start score.
L_ESTIMATES = {
    "fishwood": [0.240111632014359, 0.34796415276597487],
    "dst": [0.16001718180814467, 0.16001719646012644],
    "bb": [0.07334441503294956, 0.06938799010915637, 0.06884899017331293],
    "fruittree_d6": [0.13867360882755067, 0.1386705991183074, 0.1386719770172668,
                     0.13867074332994217, 0.13867109068783343, 0.1386744703018394],
}

# Run-to-plateau rule (mogym.plateau); SURF additionally requires its slot weights to have settled.
RULE = dict(tol=.01, window=3, confirm=3, own_ratio=.25, own_floor=1e-5)
SURF_RULE = dict(RULE, weight_tol=.005)
MAX_SWEEPS = 50000          # Uniform safety cap
MAX_ROUNDS = 1000           # SURF safety cap
ADAM_KEEP_TOL = .005        # a weight counts as unchanged if it moves by at most this in every coordinate
POINT_BAND = .05            # plotted point: earliest checkpoint within 5% of the value at stopping

# Plotted resolutions.  DST SURF N=2 does not reach a plateau within the round cap, and DST SURF N=128
# reaches it only beyond the adaptive budget; both are not plotted.
TASKS = {
    "fishwood": dict(
        uniform=dict(M=50, lr=.03, values=[2, 4, 8, 16, 32, 64, 128, 256]),
        surf=dict(K_S=25, lr=.03, values=[2, 4, 8, 16, 32, 64, 128]),
        adaptive=dict(inner_steps=10, lr=.01, lambda_method="envelope", checkpoint_count=600),
        budget_round=1000, budget=81000),
    "dst": dict(
        uniform=dict(M=5, lr=.3, values=[2, 4, 8, 16, 32, 64, 128, 256, 512]),
        surf=dict(K_S=25, lr=.1, values=[4, 8, 16, 32, 64]),
        adaptive=dict(inner_steps=10, lr=.1, lambda_method="envelope", checkpoint_count=600),
        budget_round=1000, budget=96000),
    "bb": dict(
        uniform=dict(M=25, lr=.1, values=list(range(1, 25))),
        adaptive=dict(inner_steps=25, lr=.03, lambda_method="k3_special", k3_rtol=.1, k3_max_nodes=500,
                      checkpoint_count=120),
        budget_round=3000, budget=27000),
    "fruittree_d6": dict(
        uniform=dict(M=25, lr=.03, values=[1, 2, 3, 4, 5, 6]),
        adaptive=dict(inner_steps=10, lr=.1, lambda_method="periodic_strong_ccp", hybrid_period=10,
                      weak_ccp=(128, 2, 30), strong_ccp=(1024, 8, 100), boundary_ccp_seeds=True,
                      boundary_seed_resolution=20, fresh_ccp_seeds=True, lp_warm_start=True, checkpoint_count=120),
        budget_round=6000, budget=150000),
}


def smoothness(name):
    return np.asarray(L_ESTIMATES[name], float)
