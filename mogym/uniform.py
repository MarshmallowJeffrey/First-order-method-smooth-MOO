"""Uniform discretization (Algorithm 6) with every InnerSolver call run to the GN plateau.

Algorithm 6 makes one InnerSolver call per grid weight lambda_i of G_r = {lambda in simplex: r lambda integral}; the
points it returns are appended to the bundle and kept, with the returned-point convention of Algorithm 1.  Here the
call for lambda_i is the whole Adam trajectory of that weight, advanced in sweeps of `steps` Adam steps so that all
weights progress together and the bundle can be checked after every sweep.  Every sweep of a weight returns the
iterate of smallest ||grad F_lambda_i|| among its `steps` iterates, which is appended to the bundle, as GRAB appends
the best iterate of every inner solve of M_A steps (also when it continues a trajectory):

  sweep 1   weights in snake order, each from the point of the current bundle (theta_0 and the points returned by the
            weights visited so far) minimizing F_lambda - ||grad F_lambda||^2 / (2 L_lambda), L_lambda = sum_k
            lambda_k L_k, the first minimizer in that order (Algorithm 6, Step 2, the rule of GRAB).  The Adam state
            is new, unless the chosen point was returned by a weight within state_tol (every coordinate): then the
            solve starts from a copy of the Adam state that point was produced with.
  sweep >1  each weight continues its own trajectory: from its own last iterate, with its own Adam state.

The bundle only grows: theta_0, then the returned points in the order returned.  It is kept as the points theta_i,
their objective values and their Gram matrices J_i J_i' (all the metric needs; the .npz stores them under "Q" in
place of the Jacobians).  Its metric is updated with the points appended since the last checkpoint (K=2: the exact envelope; K>2: the minima over the fixed pool, then the CCP
polishing of mogym.metrics).  incumbent_i, the smallest-gradient iterate of weight i so far, enters only the readiness
condition.  Two kinds of checkpoints are recorded (field "kind"):

  "calls"  at the end of the first grid weight past every `every` Gradient Calls up to `budget` and every 10 x `every`
           afterwards: the schedule shared with GRAB, on which the plotted point is located;
  "sweep"  at the end of every sweep.  The stopping rule (mogym.plateau) is checked at the end of every block of
           ceil(check_steps / steps) sweeps, on the GN values g_1, g_2, ... there, with the readiness condition
           P95_i ||grad F_lambda_i(incumbent_i)|| <= max(own_floor, own_ratio * GN).
"""
import numpy as np

from . import metrics, plateau
from .adam import Adam
from .lambda_solvers import Envelope, snake_grid
from .oracle import Oracle
from .recorder import Recorder


class BundleMetric:
    """The reported metric of a bundle that only grows.  add() receives the Jacobian of every appended point; each
    call turns the Jacobians received since the previous call into Gram matrices (the Jacobians are not kept) and
    updates the metric with them; called inside the checkpoints, so not part of the training time."""
    def __init__(self, K, pool, chunk=400):
        if K > 2 and pool is None:
            raise ValueError("K>2 needs the fixed weight pool")
        self.K, self.pool, self.chunk = K, pool, chunk
        self.pending, self.Q, self.n = [], np.empty((1024, K, K)), 0
        self.envelope = Envelope() if K == 2 else None
        self.values = None if K == 2 else np.full(len(pool), np.inf)  # min_i w' Q_i w for every pool weight w

    def add(self, J):
        self.pending.append(J)

    def gram(self):
        """All Gram matrices, in the order of the bundle (the pending Jacobians are converted first)."""
        if self.pending:
            new = metrics.gram(np.asarray(self.pending)); self.pending = []
            while self.n + len(new) > len(self.Q):
                self.Q = np.concatenate([self.Q, np.empty_like(self.Q)])
            self.Q[self.n:self.n + len(new)] = new; self.n += len(new)
        return self.Q[:self.n]

    def __call__(self):
        n0 = self.n
        new = self.gram()[n0:]
        if self.K == 2:
            for q in new:
                self.envelope.add(q)
            value, weight = self.envelope.solve()
            return np.sqrt(value), weight
        for s in range(0, len(self.pool) if len(new) else 0, self.chunk):
            w = self.pool[s:s + self.chunk]
            self.values[s:s + self.chunk] = np.minimum(
                self.values[s:s + self.chunk], np.einsum("wk,nkl,wl->wn", w, new, w, optimize=True).min(axis=1))
        return metrics.polish(self.gram(), self.pool, self.values)


def uniform(model, resolution, path, *, lr, steps, rule, every, budget, L, state_tol, max_sweeps, pool=None,
            save_arrays=True, run_spec=None):
    K, d = model['K'], model['d']
    grid = snake_grid(K, resolution)
    oracle = Oracle(model)
    config = dict(method='Uniform discretization', resolution=resolution, grid_size=len(grid), lr=lr,
                  steps_per_sweep=steps, state_tol=state_tol, plateau_rule=dict(rule), max_sweeps=max_sweeps,
                  checkpoint_every=every, checkpoint_budget=budget, bundle='every sweep appends its best iterate')
    rec = Recorder(oracle, config, model=model, run_spec=run_spec, save_arrays=save_arrays)
    f0, j0 = oracle(np.zeros(d)); count = K
    X, F = [np.zeros(d)], [f0]  # the bundle, in the order returned (its Gram matrices are kept by the metric)
    metric = BundleMetric(K, pool)
    metric.add(j0)
    best, seen = [None] * len(grid), []  # incumbents (|grad|^2, x, F, J, Adam m, v, t)

    def checkpoint(kind):
        return rec.checkpoint(len(X), None, None, count, metric, kind)

    checkpoint('start')
    current, opts = [None] * len(grid), [None] * len(grid)
    gs, p95s, trigger, status = [], [], None, 'safety_cap'
    nxt = every
    per_check = plateau.block(rule, steps)  # sweeps per check of the stopping rule
    for sweep in range(1, max_sweeps + 1):
        for i, lam in enumerate(grid):
            if sweep == 1:  # Algorithm 6, Step 2 (in sweep 1 the bundle is theta_0 and the incumbents of `seen`)
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
            ret = None  # the iterate this sweep returns
            for _ in range(steps):
                x = opts[i].step(x, g)
                f, j = oracle(x); count += K
                g = j.T @ lam
                gsq = float(g @ g)
                if ret is None or gsq < ret[0]:
                    ret = (gsq, x, f, j)
                if best[i] is None or gsq < best[i][0]:
                    best[i] = (gsq, x, f, j, opts[i].m.copy(), opts[i].v.copy(), opts[i].t)
            current[i] = (x, f, j)
            X.append(ret[1]); F.append(ret[2]); metric.add(ret[3])
            if count >= nxt:  # Gradient-Call checkpoint
                checkpoint('calls')
                while nxt <= count:
                    nxt += every if nxt < budget else 10 * every
        g_sweep = float(checkpoint('sweep'))
        if sweep % per_check:
            continue
        gs.append(g_sweep)
        p95s.append(float(np.quantile(np.sqrt([b[0] for b in best]), .95)))
        ready = p95s[-1] <= max(rule['own_floor'], rule['own_ratio'] * gs[-1])
        trigger, stop = plateau.update(gs, trigger, ready, rule)
        if stop:
            status = 'plateau'
            break
    end = rec.rows[-1]  # the last sweep checkpoint (at the stop if status == 'plateau')
    print(f'{model["name"]} uniform r={resolution} sweeps={sweep} {status} calls={end["component_gradients"]} '
          f'GN={end["gn"]:.5g}', flush=True)
    return rec.finish(path, np.asarray(X), np.asarray(F), None, Q=metric.gram(), extra=dict(
        status=status, sweeps=sweep, checks=len(gs), stop_calls=end['component_gradients'], stop_cpu=end['train_cpu'],
        stop_gn=end['gn'], own_gradient_p95_history=p95s))
