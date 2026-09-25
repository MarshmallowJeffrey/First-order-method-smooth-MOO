"""The three methods.  All use the same segment, step rule, budget meter and bundle (every segment end point).

Adaptive bundle method: one chain of points.  Every decision searches lambda with CCP on the current bundle and
runs s segments from the last accepted point; the step rule's state (Adam moments) is reset only when lambda
changes.

Uniform discretization with resolution r: the snake-ordered grid of Delta_K, visited forward and backward; each
visit runs s segments at the node.  The first visit of a node starts from the last accepted point of the run and
initializes its own step-rule state; later visits continue from the node's own last accepted point and state.

SURF (K = 2; Jiang et al., Algorithm 1): N + 1 slots at the quantiles n/N of the arc-length distribution Phi of the
front; every round each slot runs ``segments_per_slot`` segments at its weight, then Phi is updated from the slots'
objective values (chord lengths, PCHIP interpolation, damping alpha = 0.3).  Slots start from theta_0 and keep
their own point and step-rule state across rounds.

After a rejected segment the point stays, the step rule reacts (Adam: moments cleared, alpha halved) and L_scale
doubles; the 5th rejection in a row is accepted.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from .ccp import CCPConfig, CCPSolver
from .grid import node_index, snake_grid
from .model import initial_point
from .steppers import make_stepper
from .training import MAX_RETRIES, RunRecord, run_segment

INIT_SEED = 8
SURF_ALPHA = 0.3
PHI_GRID_POINTS = 1001
EPS_ARC = 1e-12


def _epoch_len(problem):
    return max(1, int(np.ceil(problem.n / float(problem.stoch.batch_size))))


def run_adaptive(problem, step_rule, budget, schedule, s=5, ccp_config=None):
    """The adaptive bundle method; returns the RunRecord."""
    name, params = step_rule
    K, batch = problem.K, problem.stoch.batch_size
    epoch_len = _epoch_len(problem)
    x0 = initial_point(K, INIT_SEED)
    rec = RunRecord(problem, x0, budget, schedule)
    stepper = make_stepper(name, problem.d, params)
    solver = CCPSolver(K, ccp_config or CCPConfig(), transport="bulk")
    chain = (x0.copy(), rec.f0, rec.J0)
    L_scale, prev_lam = 1.0, None
    rec.start_clock()
    while rec.budget.allows_segment(epoch_len, batch):
        t_dec = time.time()
        _, lam = solver.solve(np.asarray(rec.grams, dtype=float))
        lam = np.asarray(lam, dtype=float)
        rec.decision_seconds += time.time() - t_dec
        L_lam = float(lam @ problem.L)
        if prev_lam is None or not np.array_equal(lam, prev_lam):
            stepper.on_lambda_change(lam, L_lam, L_scale)
        prev_lam = lam
        retries = 0
        for _ in range(s):
            if not rec.budget.allows_segment(epoch_len, batch):
                break
            y, f_y, J_y, accepted = run_segment(problem, stepper, chain, lam, L_lam, L_scale, epoch_len)
            rec.add(f_y, J_y, lam)
            if accepted:
                chain, retries = (y, f_y, J_y), 0
            else:
                L_scale *= 2.0
                rec.rejections += 1
                retries += 1
                if retries > MAX_RETRIES:
                    chain, retries = (y, f_y, J_y), 0
            stepper.on_segment_result(accepted, L_lam, L_scale)
            rec.checkpoint_if_due()
    rec.finish()
    return rec


def run_uniform(problem, step_rule, budget, schedule, r, s=5):
    """Uniform discretization with resolution r; returns the RunRecord."""
    name, params = step_rule
    K, batch = problem.K, problem.stoch.batch_size
    epoch_len = _epoch_len(problem)
    grid = snake_grid(K, r)
    n_nodes = int(grid.shape[0])
    x0 = initial_point(K, INIT_SEED)
    rec = RunRecord(problem, x0, budget, schedule)
    steppers = [make_stepper(name, problem.d, params) for _ in range(n_nodes)]
    chain = (x0.copy(), rec.f0, rec.J0)
    node = [None] * n_nodes                        # each node's own point (x, f, J)
    retries = [0] * n_nodes
    L_scale, visit = 1.0, 0
    rec.start_clock()
    while rec.budget.allows_segment(epoch_len, batch):
        k = node_index(visit, n_nodes)
        visit += 1
        lam = grid[k]
        L_lam = float(lam @ problem.L)
        st = steppers[k]
        if node[k] is None:                        # first visit: from the chain point
            node[k] = chain
            st.on_lambda_change(lam, L_lam, L_scale)
        retries[k] = 0
        done = 0
        for _ in range(s):
            if not rec.budget.allows_segment(epoch_len, batch):
                break
            done += 1
            y, f_y, J_y, accepted = run_segment(problem, st, node[k], lam, L_lam, L_scale, epoch_len)
            rec.add(f_y, J_y, lam)
            if accepted:
                node[k] = chain = (y, f_y, J_y)
                retries[k] = 0
            else:
                L_scale *= 2.0
                rec.rejections += 1
                retries[k] += 1
                if retries[k] > MAX_RETRIES:
                    node[k] = chain = (y, f_y, J_y)
                    retries[k] = 0
            st.on_segment_result(accepted, L_lam, L_scale)
            rec.checkpoint_if_due()
        if done < s:
            break
    rec.finish()
    return rec


def surf_place(phi, quantiles, wgrid):
    """Slot weights: the quantiles of Phi (end slots fixed at 0 and 1)."""
    w = np.interp(quantiles, phi, wgrid)
    w[0], w[-1] = 0.0, 1.0
    return w


def surf_update_phi(w_nodes, F, phi, wgrid, alpha):
    """Chord lengths between neighbouring slots' objective values, PCHIP in w, normalized, damped."""
    chords = np.linalg.norm(np.diff(np.asarray(F, dtype=float), axis=0), axis=1)
    s_vals = np.concatenate([[0.0], np.cumsum(np.maximum(chords, EPS_ARC))])
    s_interp = PchipInterpolator(w_nodes, s_vals)(wgrid)
    phi_new = alpha * (s_interp / s_interp[-1]) + (1.0 - alpha) * phi
    phi_new[0], phi_new[-1] = 0.0, 1.0
    return phi_new


def run_surf(problem, step_rule, budget, schedule, N, segments_per_slot=5, alpha=SURF_ALPHA):
    """SURF with N + 1 slots; returns the RunRecord."""
    if problem.K != 2:
        raise ValueError("SURF is run for K = 2 only")
    name, params = step_rule
    batch = problem.stoch.batch_size
    epoch_len = _epoch_len(problem)
    x0 = initial_point(2, INIT_SEED)
    rec = RunRecord(problem, x0, budget, schedule)
    steppers = [make_stepper(name, problem.d, params) for _ in range(N + 1)]
    started = [False] * (N + 1)
    slot = [(x0.copy(), np.asarray(rec.f0, dtype=float), rec.J0) for _ in range(N + 1)]
    retries = [0] * (N + 1)
    wgrid = np.linspace(0.0, 1.0, PHI_GRID_POINTS)
    phi = wgrid.copy()
    quantiles = np.arange(N + 1) / float(N)
    L_scale, rounds = 1.0, 0
    rec.start_clock()
    stopped = False
    while not stopped:
        t_dec = time.time()
        w_nodes = surf_place(phi, quantiles, wgrid)
        rec.decision_seconds += time.time() - t_dec
        full_round = True
        for j in range(N + 1):
            for _ in range(segments_per_slot):
                if not rec.budget.allows_segment(epoch_len, batch):
                    stopped, full_round = True, False
                    break
                w = float(w_nodes[j])
                lam = np.array([w, 1.0 - w])
                L_lam = float(lam @ problem.L)
                st = steppers[j]
                if not started[j]:
                    st.on_lambda_change(lam, L_lam, L_scale)
                    started[j] = True
                y, f_y, J_y, accepted = run_segment(problem, st, slot[j], lam, L_lam, L_scale, epoch_len)
                rec.add(f_y, J_y, lam)
                if accepted:
                    slot[j], retries[j] = (y, f_y, J_y), 0
                else:
                    L_scale *= 2.0
                    rec.rejections += 1
                    retries[j] += 1
                    if retries[j] > MAX_RETRIES:
                        slot[j], retries[j] = (y, f_y, J_y), 0
                st.on_segment_result(accepted, L_lam, L_scale)
                rec.checkpoint_if_due()
            if stopped:
                break
        if full_round:
            rounds += 1
            t_dec = time.time()
            phi = surf_update_phi(w_nodes, np.asarray([sl[1] for sl in slot], dtype=float), phi, wgrid, alpha)
            rec.decision_seconds += time.time() - t_dec
    rec.finish()
    rec.surf_rounds = rounds
    return rec
