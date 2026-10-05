"""Uniform discretization (paper Algorithm 6) with every InnerSolver call run to the GN plateau.

Algorithm 6 makes one InnerSolver call per grid weight lambda_i of G_r = {lambda in simplex: r lambda integral} and
adds the iterate of smallest ||grad F_lambda_i|| to the bundle.  Here the call for lambda_i is the whole Adam
trajectory of that weight, advanced in sweeps of `steps` Adam steps so that all weights progress together and the
bundle can be checked after every sweep:

  sweep 1   weights in snake order, each from the point of the current bundle (theta_0 and the incumbents of the
            weights visited so far) minimizing F_lambda - ||grad F_lambda||^2 / (2 L_lambda), L_lambda = sum_k
            lambda_k L_k, the first minimizer in that order (Algorithm 6, Step 2, the rule of GRAB).  The Adam state
            is new, unless the chosen point is the incumbent of a weight within state_tol (every coordinate): then
            the solve starts from a copy of the Adam state that incumbent was produced with.
  sweep >1  each weight continues its own trajectory: from its own last iterate, with its own Adam state.

At a checkpoint the bundle is {theta_0} U {incumbent_i}, incumbent_i = the min-grad iterate of weight i so far (within
the first sweep: the weights visited so far).  Two kinds of checkpoints are recorded (field "kind"):

  "calls"  at the end of the first grid weight past every `every` Gradient Calls up to `budget` and every 10 x `every`
           afterwards: the schedule shared with GRAB, on which the plotted point is located;
  "sweep"  at the end of every sweep: the values g_1, g_2, ... of the stopping rule (mogym.plateau) with the readiness
           condition P95_i ||grad F_lambda_i(incumbent_i)|| <= max(own_floor, own_ratio * GN).
"""
import numpy as np

from . import metrics, plateau
from .adam import Adam
from .lambda_solvers import snake_grid
from .oracle import Oracle
from .recorder import Recorder


def uniform(model, resolution, path, *, lr, steps, rule, every, budget, L, state_tol, max_sweeps, pool=None,
            save_arrays=True, run_spec=None):
    K, d = model['K'], model['d']
    grid = snake_grid(K, resolution)
    oracle = Oracle(model)
    config = dict(method='Uniform discretization', resolution=resolution, grid_size=len(grid), lr=lr,
                  steps_per_sweep=steps, state_tol=state_tol, plateau_rule=dict(rule), max_sweeps=max_sweeps,
                  checkpoint_every=every, checkpoint_budget=budget)
    rec = Recorder(oracle, config, model=model, run_spec=run_spec, save_arrays=save_arrays)
    f0, j0 = oracle(np.zeros(d)); count = K
    grams = np.empty((len(grid) + 1, K, K))
    grams[0] = metrics.gram(j0[None])[0]
    best, seen = [None] * len(grid), []  # incumbents (|grad|^2, x, F, J, Adam m, v, t)

    def metric(changed):
        for i in changed:
            grams[1 + i] = metrics.gram(best[i][3][None])[0]
        return metrics.reporting_metric_gram(grams[[0] + [1 + i for i in seen]], pool)

    def checkpoint(kind, changed):
        return rec.checkpoint(len(seen) + 1, None, None, count, lambda: metric(changed), kind)

    checkpoint('start', [])
    current, opts = [None] * len(grid), [None] * len(grid)
    gs, p95s, trigger, status = [], [], None, 'safety_cap'
    nxt, changed_calls, changed_sweep = every, [], []
    for sweep in range(1, max_sweeps + 1):
        for i, lam in enumerate(grid):
            if sweep == 1:  # Algorithm 6, Step 2
                ll = float(np.asarray(L) @ lam)
                cands = [(np.zeros(d), f0, j0, None)] + [(best[k][1], best[k][2], best[k][3], k) for k in seen]
                scores = [float(fc @ lam) - float((jc.T @ lam) @ (jc.T @ lam)) / (2 * ll) for _, fc, jc, _ in cands]
                x, f, j, owner = cands[int(np.argmin(scores))]
                opts[i] = Adam(d, lr)
                if owner is not None and float(np.max(np.abs(grid[owner] - lam))) <= state_tol:
                    opts[i].m, opts[i].v, opts[i].t = best[owner][4].copy(), best[owner][5].copy(), best[owner][6]
                seen.append(i)
            else:
                x, f, j = current[i]
            g = j.T @ lam
            old = best[i]
            for _ in range(steps):
                x = opts[i].step(x, g)
                f, j = oracle(x); count += K
                g = j.T @ lam
                gsq = float(g @ g)
                if best[i] is None or gsq < best[i][0]:
                    best[i] = (gsq, x, f, j, opts[i].m.copy(), opts[i].v.copy(), opts[i].t)
            current[i] = (x, f, j)
            if best[i] is not old:
                changed_calls.append(i); changed_sweep.append(i)
            if count >= nxt:  # Gradient-Call checkpoint
                checkpoint('calls', changed_calls); changed_calls = []
                while nxt <= count:
                    nxt += every if nxt < budget else 10 * every
        gs.append(float(checkpoint('sweep', changed_sweep))); changed_sweep = []
        p95s.append(float(np.quantile(np.sqrt([b[0] for b in best]), .95)))
        ready = p95s[-1] <= max(rule['own_floor'], rule['own_ratio'] * gs[-1])
        trigger, stop = plateau.update(gs, trigger, ready, rule)
        if stop:
            status = 'plateau'
            break
    X = np.asarray([np.zeros(d)] + [b[1] for b in best])
    F = np.asarray([f0] + [b[2] for b in best])
    J = np.asarray([j0] + [b[3] for b in best])
    end = rec.rows[-1]  # the last sweep checkpoint (at the stop if status == 'plateau')
    print(f'{model["name"]} uniform r={resolution} sweeps={len(gs)} {status} calls={end["component_gradients"]} '
          f'GN={end["gn"]:.5g}', flush=True)
    return rec.finish(path, X, F, J, dict(
        status=status, sweeps=len(gs), stop_calls=end['component_gradients'], stop_cpu=end['train_cpu'],
        stop_gn=end['gn'], own_gradient_p95_history=p95s))
