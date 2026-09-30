"""Finite MDP models of the four MO-Gymnasium tasks (MO-Gymnasium 1.3.2).

build(name) returns a dict with the transition tensor P (S, A, S), the vector reward R (S, A, K), the
initial distribution rho0, the discount gamma, the KL coefficient tau and the uniform reference policy.
The tabular builders store float32 values (as the original SURF notebooks do); the arrays are then
converted to float64.

  fishwood      K=2   2 states, 2 actions        gamma 0.995, tau 0.5  (as in the SURF paper)
  dst           K=2   concave Deep Sea Treasure  gamma 0.999, tau 1.5  (as in the SURF paper)
  bb            K=3   Breakable Bottles          gamma 0.99,  tau 1
  fruittree_d6  K=6   Fruit Tree, depth 6        gamma 0.99,  tau 1
"""
import numpy as np

SETTINGS = {
    "fishwood": dict(gamma=0.995, tau=0.5),
    "dst": dict(gamma=0.999, tau=1.5),
    "bb": dict(gamma=0.99, tau=1.0),
    "fruittree_d6": dict(gamma=0.99, tau=1.0, depth=6),
}


def build(name):
    cfg = SETTINGS[name]
    if name == "fishwood":
        model = _fishwood(fishproba=0.1, woodproba=0.9)
    elif name == "dst":
        model = _deep_sea_treasure()
    elif name == "bb":
        model = _breakable_bottles()
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


# -------------------------------------------------------- Deep Sea Treasure
def _deep_sea_treasure():
    """deep-sea-treasure-concave-v0 as a deterministic tabular model.  States: the non-blocked cells plus
    one absorbing terminal state; treasure cells are terminal on entry.  R[s, a, 0] = treasure value on
    entering a treasure cell, R[s, a, 1] = -1 per step."""
    import mo_gymnasium as mo_gym
    env = mo_gym.make("deep-sea-treasure-concave-v0")
    uenv = env.unwrapped
    grid = next(np.array(getattr(uenv, n)) for n in ("sea_map", "dst_map", "map", "_map", "grid", "treasure_map")
                if hasattr(uenv, n) and np.array(getattr(uenv, n)).ndim == 2)
    nrow, ncol = grid.shape
    states = [(r, c) for r in range(nrow) for c in range(ncol) if grid[r, c] != -10] + [("terminal",)]
    index = {s: i for i, s in enumerate(states)}
    S, A = len(states), 4
    terminal = index[("terminal",)]
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
    P = np.zeros((S, A, S), dtype=np.float32)
    R = np.zeros((S, A, 2), dtype=np.float32)
    for i, s in enumerate(states):
        if s == ("terminal",) or grid[s] > 0:
            P[i, :, terminal] = 1.0
            continue
        r, c = s
        for a, (dr, dc) in enumerate(deltas):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= nrow or nc < 0 or nc >= ncol or grid[nr, nc] == -10:
                nr, nc = r, c
            P[i, a, index[(nr, nc)]] = 1.0
            R[i, a, 0] = float(grid[nr, nc]) if grid[nr, nc] > 0 else 0.0
            R[i, a, 1] = -1.0
    obs, _ = env.reset()
    env.close()
    rho0 = np.zeros(S, dtype=np.float32)
    rho0[index[tuple(obs.tolist()) if hasattr(obs, "tolist") else tuple(obs)]] = 1.0
    return dict(S=S, A=A, P=P, R=R, rho0=rho0)


# ---------------------------------------------------------------- Fruit Tree
def _fruit_tree(depth):
    """Full binary tree; node (row, col) has children (row+1, 2col+a).  The 6-dimensional fruit vector of
    a leaf is received on the transition into the leaf; leaves move to an absorbing terminal state."""
    import mo_gymnasium as mo_gym
    env = mo_gym.make("fruit-tree-v0", depth=depth)
    tree = np.asarray(env.unwrapped.tree, dtype=np.float32).copy()
    env.close()
    K, A = 6, 2
    terminal = 2 ** (depth + 1) - 1
    S = terminal + 1
    P = np.zeros((S, A, S))
    R = np.zeros((S, A, K))
    for row in range(depth + 1):
        for col in range(2 ** row):
            s = 2 ** row - 1 + col
            for a in range(A):
                child = terminal if row == depth else 2 ** (row + 1) - 1 + 2 * col + a
                P[s, a, child] = 1.
                if row == depth - 1:
                    R[s, a] = tree[child]
    P[terminal, :, terminal] = 1.
    rho0 = np.zeros(S)
    rho0[0] = 1.
    return dict(S=S, A=A, P=P, R=R, rho0=rho0)


# --------------------------------------------------------- Breakable Bottles
# State = (location 0..4, bottles carried 0..2, bottles delivered 0..2, 3-bit mask of dropped bottles at
# locations 1..3); index 360 is an absorbing terminal.  Rules of mo_gymnasium breakable-bottles-v0 (default
# registration: prob_drop 0.1, time_penalty -1, bottle_reward 25, breakable bottles).
BB_SIZE = 5
BB_LEFT, BB_RIGHT, BB_PICKUP = 0, 1, 2
BB_PROB_DROP = 0.1
BB_N_CARRY, BB_N_DELIV, BB_N_DROPPED = 3, 3, 8
BB_TERMINAL = BB_SIZE * BB_N_CARRY * BB_N_DELIV * BB_N_DROPPED   # 360
BB_N_STATES = BB_TERMINAL + 1                                    # 361


def _bb_index(loc, carry, deliv, dropped_bits):
    return loc * (BB_N_CARRY * BB_N_DELIV * BB_N_DROPPED) + carry * (BB_N_DELIV * BB_N_DROPPED) \
        + deliv * BB_N_DROPPED + dropped_bits


def _bb_state(idx):
    dropped_bits = idx % BB_N_DROPPED
    idx //= BB_N_DROPPED
    deliv = idx % BB_N_DELIV
    idx //= BB_N_DELIV
    return idx // BB_N_CARRY, idx % BB_N_CARRY, deliv, dropped_bits


def _bb_potential(dropped_bits):
    """phi(s) = -1 if any bottle lies on the ground, else 0."""
    return -1 if dropped_bits != 0 else 0


def _bb_move(loc, carry, deliv, dropped_bits, action):
    """Successor state (without a drop event) and bottles delivered in this step."""
    delivered = 0
    if action == BB_LEFT and loc > 0:
        loc -= 1
    elif action == BB_RIGHT and loc < BB_SIZE - 1:
        loc += 1
        if loc == BB_SIZE - 1 and carry > 0:
            after = min(deliv + carry, 2)
            delivered = after - deliv
            deliv, carry = after, 0
    elif action == BB_PICKUP and loc == 0 and carry < 2:
        carry += 1
    return (loc, carry, deliv, dropped_bits), delivered


def _bb_drop_then_move(loc, carry, deliv, dropped_bits, action):
    """Drop branch: one of two carried bottles falls at the current tile."""
    dropped_bits |= 1 << (loc - 1)
    return _bb_move(loc, carry - 1, deliv, dropped_bits, action)


def _breakable_bottles():
    S, A, K = BB_N_STATES, 3, 3
    P, R = np.zeros((S, A, S)), np.zeros((S, A, K))
    for s in range(S):
        if s == BB_TERMINAL or _bb_state(s)[2] == 2:
            P[s, :, BB_TERMINAL] = 1.
            continue
        loc, carry, delivered, bits = _bb_state(s)
        for a in range(A):
            eligible = a in (BB_LEFT, BB_RIGHT) and carry == 2 and 1 <= loc <= 3
            branches = [(1 - BB_PROB_DROP, False), (BB_PROB_DROP, True)] if eligible else [(1., False)]
            for prob, drop in branches:
                ns, count = (_bb_drop_then_move if drop else _bb_move)(loc, carry, delivered, bits, a)
                dest = BB_TERMINAL if ns[2] == 2 else _bb_index(*ns)
                P[s, a, dest] += prob
                R[s, a] += prob * np.array([-1., 25. * count, _bb_potential(ns[3]) - _bb_potential(bits)])
    rho0 = np.zeros(S)
    rho0[_bb_index(4, 0, 0, 0)] = 1.
    return dict(S=S, A=A, P=P, R=R, rho0=rho0)
