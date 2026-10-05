"""Finite MDP models of the two MO-Gymnasium tasks (MO-Gymnasium 1.3.2).

build(name) returns a dict with the transition tensor P (S, A, S), the vector reward R (S, A, K), the
initial distribution rho0, the discount gamma, the KL coefficient tau and the uniform reference policy.
The tabular builders store float32 values (as the original SURF notebooks do); the arrays are then
converted to float64.

  fishwood      K=2   2 states, 2 actions        gamma 0.995, tau 0.5  (as in the SURF paper)
  fruittree_d6  K=6   Fruit Tree, depth 6        gamma 0.99,  tau 1  (64 states: 63 internal nodes + absorbing)
"""
import numpy as np

SETTINGS = {
    "fishwood": dict(gamma=0.995, tau=0.5),
    "fruittree_d6": dict(gamma=0.99, tau=1.0, depth=6),
}


def build(name):
    cfg = SETTINGS[name]
    if name == "fishwood":
        model = _fishwood(fishproba=0.1, woodproba=0.9)
    elif name == "fruittree_d6":
        model = _fruit_tree(cfg["depth"])
    else:
        raise ValueError(name)
    model = dict(model)
    model["S"], model["A"] = int(model["S"]), int(model["A"])
    model["K"] = int(np.asarray(model["R"]).shape[-1])
    for key in ("P", "R", "rho0"):
        model[key] = np.asarray(model[key], dtype=float)
    model["pi_ref"] = np.full((model["S"], model["A"]), 1. / model["A"])
    model.update(name=name, gamma=cfg["gamma"], tau=cfg["tau"], d=model["S"] * model["A"])
    assert np.max(np.abs(model["P"].sum(axis=2) - 1)) < 1e-14
    return model


# ------------------------------------------------------------------ FishWood
def _fishwood(fishproba, woodproba):
    """fishwood-v0: state 0 = fishing, 1 = woods; action a moves to state a.  R[s, a, 0] = fish rate in
    state 0, R[s, a, 1] = wood rate in state 1 (expected rewards).  Start in the woods."""
    S, A = 2, 2
    P = np.zeros((S, A, S), dtype=np.float32)
    for s in range(S):
        for a in range(A):
            P[s, a, a] = 1.0
    R = np.zeros((S, A, 2), dtype=np.float32)
    R[0, :, 0] = float(fishproba)
    R[1, :, 1] = float(woodproba)
    rho0 = np.array([0.0, 1.0], dtype=np.float32)
    return dict(S=S, A=A, P=P, R=R, rho0=rho0)


# ---------------------------------------------------------------- Fruit Tree
def _fruit_tree(depth):
    """The model of the paper: the 2^depth - 1 internal nodes (i, j), 0 <= i <= depth - 1, 0 <= j < 2^i, indexed
    2^i - 1 + j, and one absorbing state.  From (i, j) with i <= depth - 2 action a moves to (i + 1, 2j + a); from
    (depth - 1, j) action a picks the fruit of leaf 2j + a (its 6-dimensional nutrient vector is the reward) and
    moves to the absorbing state, which loops on itself with reward 0."""
    import mo_gymnasium as mo_gym
    env = mo_gym.make("fruit-tree-v0", depth=depth)
    tree = np.asarray(env.unwrapped.tree, dtype=np.float32).copy()  # full tree; the leaves are 2^depth - 1, ...
    env.close()
    K, A = 6, 2
    absorbing = 2 ** depth - 1
    S = absorbing + 1
    P = np.zeros((S, A, S))
    R = np.zeros((S, A, K))
    for row in range(depth):
        for col in range(2 ** row):
            s = 2 ** row - 1 + col
            for a in range(A):
                if row == depth - 1:
                    P[s, a, absorbing] = 1.
                    R[s, a] = tree[2 ** depth - 1 + 2 * col + a]
                else:
                    P[s, a, 2 ** (row + 1) - 1 + 2 * col + a] = 1.
    P[absorbing, :, absorbing] = 1.
    rho0 = np.zeros(S)
    rho0[0] = 1.
    return dict(S=S, A=A, P=P, R=R, rho0=rho0)
