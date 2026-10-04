"""Uniform discretization (paper Algorithm 6, uniform-grid bundle method) with every InnerSolver run to the GN plateau.

Algorithm 6 makes one InnerSolver call per grid weight lambda_i of G_r = {lambda in simplex: r lambda
integral} and adds R(CandidatePoints) to the bundle, R = the candidate of smallest ||grad F_lambda_i||.
Here the call for lambda_i is the whole Adam trajectory of that weight, advanced in sweeps of `steps`
Adam steps so that all weights progress together and the bundle can be checked after every sweep:

  sweep 1   weights in snake order (consecutive weights 2/r apart in l1), each with a new Adam state, from the
            point of the current bundle (theta_0 and the incumbents of the weights visited so far) minimizing
            F_lambda - ||grad F_lambda||^2 / (2 L_lambda), L_lambda = sum_k lambda_k L_k, the first minimizer in
            that order (Algorithm 6, Step 2; the rule of GRAB).  Sweep 1 is thus Algorithm 6 with `steps`
            iterations per weight;
  sweep >1  each weight continues its own InnerSolver call: from its own last iterate, with its own Adam state.

At a checkpoint the bundle is {theta_0} U {incumbent_i}, incumbent_i = the min-grad iterate of weight i so
far (within the first sweep: the weights visited so far), i.e. what Algorithm 6 returns if the InnerSolvers
stopped there.  Two kinds of checkpoints are recorded (field "kind"):

  "calls"  at the end of the first grid weight past every `every` Gradient Calls up to `budget` and every
           10 x `every` afterwards: the schedule shared with GRAB, on which the plotted point is located;
  "sweep"  at the end of every sweep: the values g_1, g_2, ... of the stopping rule (mogym.plateau) with the
           readiness condition P95_i ||grad F_lambda_i(incumbent_i)|| <= max(own_floor, own_ratio * GN).
"""
import numpy as np

from . import metrics, plateau
from .adam import Adam
from .lambda_solvers import snake_grid
from .oracle import Oracle
from .recorder import Recorder


def uniform_plateau(model, resolution, path, *, lr, steps, rule, every, budget, pool=None, max_sweeps=50000,
                    adam_beta1=.9, adam_beta2=.999, save_arrays=True, L):
    K, d = model['K'], model['d']
    grid = snake_grid(K, resolution)
    M = int(steps)
    oracle = Oracle(model)
    config = dict(method='Uniform discretization', resolution=resolution, grid_size=len(grid), lr=lr,
                  steps_per_sweep=M, order='snake', plateau_rule=dict(rule), max_sweeps=max_sweeps,
                  adam_beta1=adam_beta1, adam_beta2=adam_beta2, checkpoint_every=every, checkpoint_budget=budget)
    rec = Recorder(oracle, config, save_arrays=save_arrays)
    f0, j0 = oracle(np.zeros(d)); count = K
    grams = np.empty((len(grid) + 1, K, K))
    grams[0] = metrics.gram(j0[None])[0]
    best, seen = [None] * len(grid), []

    def metric(changed):
        for i in changed:
            grams[1 + i] = metrics.gram(best[i][3][None])[0]
        return metrics.reporting_metric_gram(grams[[0] + [1 + i for i in seen]], pool)

    def checkpoint(kind, changed):
        size = np.empty((len(seen) + 1, 0))  # only the bundle size is passed to the recorder
        gn = rec.checkpoint(size, None, size, count, force_exact=lambda: metric(changed))
        rec.rows[-1]['kind'] = kind
        return gn

    rec.checkpoint(np.zeros((1, d)), None, np.empty(0), count, force_exact=lambda: metric([]))
    rec.rows[-1]['kind'] = 'start'
    current, opts = [None] * len(grid), [None] * len(grid)
    gs, p95s, ready_history, trigger, status, stop_a = [], [], [], None, 'safety_cap', None
    nxt, changed_calls, changed_sweep = every, [], []
    for sweep in range(1, max_sweeps + 1):
        for i, lam in enumerate(grid):
            if sweep == 1:
                ll = float(np.asarray(L) @ lam)  # Algorithm 6, Step 2
                cands = [(np.zeros(d), f0, j0)] + [best[k][1:] for k in seen]
                scores = [float(fc @ lam) - float((jc.T @ lam) @ (jc.T @ lam)) / (2 * ll) for _, fc, jc in cands]
                x, f, j = cands[int(np.argmin(scores))]
                opts[i] = Adam(d, lr, beta1=adam_beta1, beta2=adam_beta2); seen.append(i)
            else:
                x, f, j = current[i]
            g = j.T @ lam
            old = best[i]
            for _ in range(M):
                x = opts[i].step(x, g)
                f, j = oracle(x); count += K
                g = j.T @ lam
                gsq = float(g @ g)
                if best[i] is None or gsq < best[i][0]:
                    best[i] = (gsq, x, f, j)
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
        ready_history.append(bool(ready))
        trigger, stop = plateau.update(gs, trigger, ready, rule)
        if stop:
            stop_a = len(gs); status = 'plateau'
            break
    X = np.asarray([np.zeros(d)] + [b[1] for b in best])
    F = np.asarray([f0] + [b[2] for b in best])
    J = np.asarray([j0] + [b[3] for b in best])
    end = rec.rows[-1]  # the last sweep checkpoint (at the stop if status == 'plateau')
    own = np.sqrt([b[0] for b in best])
    print(f'{model["name"]} uniform r={resolution} M={M} sweeps={len(gs)} {status} '
          f'calls={end["component_gradients"]} GN={end["gn"]:.5g}', flush=True)
    return rec.finish(path, X, F, J, dict(
        status=status, sweeps=len(gs), trigger_sweep=trigger if stop_a else None, bundle_size=len(grid) + 1,
        stop_calls=end['component_gradients'], stop_cpu=end['train_cpu'], stop_gn=end['gn'],
        ready_history=ready_history, own_gradient_p95_history=p95s, grid=grid.tolist(),
        own_gradient_p95=float(np.quantile(own, .95)), own_gradient_max=float(own.max())))
