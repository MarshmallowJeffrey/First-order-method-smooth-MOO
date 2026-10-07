"""GRAB (Algorithm 1) with an Adam inner solver.

Outer iteration t (no tolerance epsilon; the run stops when the budget is spent):
  1. lambda_t: a preference vector with large GN(.; B_{t-1}) (K=2: the exact maximizer from the envelope; K>2: the
     CCP selection of mogym.lambda_solvers with the options of the configuration).
  2. theta_0^(t) = argmin over the bundle of F_lambda_t - ||grad F_lambda_t||^2 / (2 L_lambda_t), L_lambda = sum_k
     lambda_k L_k, the first minimizer in the stored order.
  3. inner_steps Adam steps on F_lambda_t.  Every inner solve belongs to a trajectory, which keeps its last iterate
     and its Adam state; theta_0^(t) and the point added in step 4 belong to it.  If theta_0^(t) belongs to a
     trajectory last run at a weight within state_tol of lambda_t (every coordinate), the solve resumes that
     trajectory from its last iterate and Adam state; otherwise a new trajectory starts at theta_0^(t) with a new
     Adam state.
  4. The iterate of the solve with the smallest ||grad F_lambda_t|| is added to the bundle.
Every oracle call returns all K objective values and gradients and counts K Gradient Calls; theta_0 counts too.
"""
import time

import numpy as np

from . import lambda_solvers as ls
from .adam import Adam
from .oracle import Oracle
from .recorder import Recorder


def _grow(n, *arrays):
    if n < len(arrays[0]):
        return arrays
    return tuple(np.concatenate([a, np.empty_like(a)]) for a in arrays)


def adaptive(model, L, budget, path, *, lr, inner_steps, state_tol, checkpoint_count, ccp=None, run_spec=None):
    """ccp: CCP settings for K>2 (nseeds, nstarts, maxiter, boundary_resolution, keep_pool)."""
    K, d = model['K'], model['d']
    oracle = Oracle(model)
    config = dict(method='GRAB', lr=lr, inner_steps=inner_steps, state_tol=state_tol, budget=budget,
                  L=np.asarray(L).tolist(), selection='envelope' if K == 2 else dict(method='CCP', **ccp))
    solver = ls.Envelope() if K == 2 else ls.CCP(K, **ccp)
    ls.reset_lp_state()
    rec = Recorder(oracle, config, model=model, run_spec=run_spec)
    capacity = budget // (K * inner_steps) + 3
    X = np.empty((capacity, d)); F = np.empty((capacity, K)); J = np.empty((capacity, K, d))
    G = np.empty((capacity, K, K))  # Gram matrices J_i J_i'
    TR = np.empty(capacity, dtype=int); trajs = []  # trajectory of every bundle point; their last iterates and states
    X[0] = 0.; F[0], J[0] = oracle(X[0]); TR[0] = -1; n = 1; count = K
    G[0] = J[0] @ J[0].T; solver.add(G[0])

    def exact_metric():  # K=2: the checkpoint GN is the envelope's exact value
        value, weight = solver.solve()
        return np.sqrt(value), weight

    def no_estimate():  # K>2: the reported metric is computed afterwards on the fixed pool (mogym.points)
        return float('nan'), np.full(K, np.nan)

    metric = exact_metric if K == 2 else no_estimate
    rec.checkpoint(n, F[:n], J[:n], count, metric)
    lambdas = []; continued = 0; lambda_time = 0.
    # checkpoint at the end of the first outer iteration that completes at or after every budget / checkpoint_count calls
    thresholds = np.linspace(0, budget, checkpoint_count + 1)[1:]; threshold = 0
    while count + K <= budget:
        t = time.perf_counter(); lam = solver.select(); lambda_time += time.perf_counter() - t
        lambdas.append(lam.tolist())
        ll = float(L @ lam)
        # Step 2 from the stored values and Gram matrices: ||grad F_lambda(theta_i)||^2 = lambda' Q_i lambda
        scores = F[:n] @ lam - np.einsum('nkl,k,l->n', G[:n], lam, lam) / (2 * ll)
        idx = int(np.argmin(scores)); tid = int(TR[idx])
        if tid >= 0 and np.max(np.abs(trajs[tid]['lam'] - lam)) <= state_tol:  # resume the trajectory
            x = trajs[tid]['x'].copy(); g = trajs[tid]['j'].T @ lam; opt = trajs[tid]['opt']; continued += 1
        else:  # a new trajectory at theta_0^(t)
            x = X[idx].copy(); g = J[idx].T @ lam; opt = Adam(d, lr)
            tid = len(trajs); trajs.append(dict(opt=opt)); TR[idx] = tid
        candidates = []
        for _ in range(min(inner_steps, (budget - count) // K)):
            x = opt.step(x, g)
            f, j = oracle(x); g = j.T @ lam; count += K
            candidates.append((float(g @ g), x, f, j))
        trajs[tid].update(x=x.copy(), f=f, j=j, lam=lam.copy())
        best = min(candidates, key=lambda a: a[0])
        X, F, J, G, TR = _grow(n, X, F, J, G, TR)
        X[n], F[n], J[n] = best[1:]; TR[n] = tid
        G[n] = J[n] @ J[n].T; solver.add(G[n]); n += 1
        if threshold < len(thresholds) and count >= thresholds[threshold]:
            gn = rec.checkpoint(n, F[:n], J[:n], count, metric)
            print(f'{model["name"]} GRAB calls={count} GN={gn:.5g}', flush=True)
            while threshold < len(thresholds) and count >= thresholds[threshold]:
                threshold += 1
    if rec.rows[-1]['component_gradients'] != count:
        rec.checkpoint(n, F[:n], J[:n], count, metric)
    return rec.finish(path, X[:n], F[:n], J[:n], dict(
        lambdas=lambdas, lambda_selection_wall=lambda_time, continued_trajectories=continued, trajectories=len(trajs),
        lp_count=getattr(solver, 'lp_count', 0)))
