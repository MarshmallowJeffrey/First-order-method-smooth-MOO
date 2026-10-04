"""Adaptive bundle method (paper Algorithm 1) with an Adam inner solver.

One outer iteration:
  1. lambda_t = argmax_lambda min_i ||J_i' lambda|| over the current bundle (preference-weight solver:
     K=2 exact envelope, K>2 periodic multistart CCP; mogym.lambda_solvers);
  2. warm start at argmin_{theta in B} F_lambda(theta) - ||grad F_lambda(theta)||^2 / (2 L_lambda),
     L_lambda = sum_k lambda_k L_k (Algorithm 1, Step 2);
  3. inner_steps Adam steps on F_lambda from the point of step 2.  Adam state: every weight keeps its own state, as
     every Uniform grid weight and SURF slot: if lambda_t lies within adam_keep_tol (max_k |lambda_t,k - lambda_k|)
     of the weight of a stored state, the most recently used such state is continued and its weight set to
     lambda_t; otherwise (or with adam_keep_tol=None) a new state is used (and stored);
  4. the trajectory point with the smallest ||grad F_lambda|| is added to the bundle.
Every oracle call returns the full Jacobian and counts K Gradient Calls; the initial point counts too.  The run
stops when the budget is spent (no tolerance epsilon); every inner solve takes inner_steps steps (fewer only at
the end of the budget).
"""
import time

import numpy as np

from . import lambda_solvers as ls
from .adam import Adam
from .oracle import Oracle
from .recorder import Recorder


def _grow(X, F, J, n):
    if n < len(X):
        return X, F, J
    extra = len(X)
    return (np.concatenate([X, np.empty((extra,) + X.shape[1:])]),
            np.concatenate([F, np.empty((extra,) + F.shape[1:])]),
            np.concatenate([J, np.empty((extra,) + J.shape[1:])]))


def adaptive(model, L, budget, path, *, lr, inner_steps, lambda_method='envelope', checkpoint_count=20,
             hybrid_period=10, weak_ccp=(128, 2, 30), strong_ccp=(1024, 8, 100),
             boundary_ccp_seeds=False, boundary_seed_resolution=4, fresh_ccp_seeds=False,
             ccp_keep_pool=0, adam_keep_tol=None, lp_warm_start=False, lp_cg=False, adam_beta1=.9, adam_beta2=.999):
    """lambda_method: 'envelope' (K=2) or 'periodic_strong_ccp' (K>2)."""
    K, d = model['K'], model['d']
    expected = 'envelope' if K == 2 else 'periodic_strong_ccp'
    if lambda_method != expected:
        raise ValueError(f'K={K} uses lambda_method={expected!r}')
    oracle = Oracle(model)
    config = dict(method='Adaptive bundle', lr=lr, budget=budget, inner_steps=inner_steps, seed=42,
                  lambda_method=lambda_method, L=np.asarray(L).tolist(), adam_beta1=adam_beta1,
                  adam_beta2=adam_beta2, adam_keep_tol=adam_keep_tol, adam_state='per_weight',
                  lp_warm_start=bool(lp_warm_start),
                  lp_constraint_generation=bool(lp_cg))
    if K == 2:
        solver = ls.Envelope()
    else:
        config.update(hybrid_period=hybrid_period, weak_ccp=list(weak_ccp), strong_ccp=list(strong_ccp),
                      boundary_ccp_seeds=bool(boundary_ccp_seeds), boundary_seed_resolution=boundary_seed_resolution,
                      fresh_ccp_seeds=bool(fresh_ccp_seeds), ccp_keep_pool=int(ccp_keep_pool))
        solver = ls.PeriodicStrongCCP(K, period=hybrid_period, weak=weak_ccp, strong=strong_ccp,
                                      boundary_seeds=boundary_ccp_seeds, boundary_resolution=boundary_seed_resolution,
                                      fresh_seeds=fresh_ccp_seeds, keep_pool=ccp_keep_pool)
    ls.WARM_LP = False; ls.LP_CG = False; ls._HIGHS.pop("basis_shape", None); ls._CG.clear()
    rec = Recorder(oracle, config)
    capacity = budget // (K * inner_steps) + 3
    X = np.empty((capacity, d)); F = np.empty((capacity, K)); J = np.empty((capacity, K, d))
    X[0] = 0.; F[0], J[0] = oracle(X[0]); n = 1; count = K
    solver.add(J[0] @ J[0].T)

    def cached_metric():  # K=2: the checkpoint GN is the training solver's own (exact) value
        solved = solver.solve(); value, weight = solved[:2]
        upper = value if len(solved) == 2 else solved[2]
        return np.sqrt(value), weight, np.sqrt(upper)

    checkpoint_kwargs = dict(force_exact=cached_metric) if K == 2 else {}
    rec.checkpoint(X[:n], F[:n], J[:n], count, **checkpoint_kwargs)
    lambdas = []; steps_used = []; lambda_time = 0.; inner_time = 0.
    prev_opt, kept_state, resumed_state = None, 0, 0
    state_w = np.empty((0, K)); state_opt, state_used = [], np.empty(0)  # weights, Adam states, last use
    ls.WARM_LP = bool(lp_warm_start); ls._HIGHS.pop("basis_shape", None)
    ls.LP_CG = bool(lp_cg); ls._CG.clear()
    thresholds = np.linspace(0, budget, checkpoint_count + 1)[1:]; threshold = 0
    while count + K <= budget:
        t = time.perf_counter(); solved = solver.solve(); lam = solved[1]
        lambda_time += time.perf_counter() - t; lambdas.append(lam.tolist())
        t = time.perf_counter(); ll = float(L @ lam)
        sg = np.einsum('nkd,k->nd', J[:n], lam, optimize=True)
        scores = F[:n] @ lam - np.einsum('nd,nd->n', sg, sg) / (2 * ll)
        idx = int(np.argmin(scores)); x = X[idx].copy(); g = sg[idx].copy()
        near = (np.flatnonzero(np.max(np.abs(state_w - lam), axis=1) <= adam_keep_tol) if adam_keep_tol is not None
                else np.empty(0, int))
        if len(near):
            k = int(near[np.argmax(state_used[near])]); opt = state_opt[k]
            kept_state += 1; resumed_state += int(opt is not prev_opt)
        else:
            k = len(state_opt); opt = Adam(d, lr, beta1=adam_beta1, beta2=adam_beta2); state_opt.append(opt)
            state_w = np.vstack([state_w, lam[None]]); state_used = np.r_[state_used, 0.]
        state_w[k] = lam; state_used[k] = len(lambdas); prev_opt = opt
        candidates = []; steps = min(inner_steps, (budget - count) // K)
        for _ in range(steps):
            xn = opt.step(x, g)
            f, j = oracle(xn); gn = j.T @ lam; norm = float(gn @ gn)
            candidates.append((norm, xn, f, j))
            x, g = xn, gn; count += K
        steps_used.append(len(candidates))
        best = min(candidates, key=lambda a: a[0])
        X, F, J = _grow(X, F, J, n)
        X[n], F[n], J[n] = best[1:]; solver.add(J[n] @ J[n].T); n += 1
        inner_time += time.perf_counter() - t
        if threshold < len(thresholds) and count >= thresholds[threshold] - K:
            gn = rec.checkpoint(X[:n], F[:n], J[:n], count, **checkpoint_kwargs)
            print(f'{model["name"]} adaptive calls={count} GN={gn:.5g}', flush=True)
            while threshold < len(thresholds) and count >= thresholds[threshold] - K:
                threshold += 1
    if rec.rows[-1]['component_gradients'] != count:
        rec.checkpoint(X[:n], F[:n], J[:n], count, **checkpoint_kwargs)
    extra = dict(lambda_selection_wall=lambda_time, inner_wall=inner_time, lambdas=lambdas,
                 lp_count=getattr(solver, 'lp_count', 0), inner_steps_used=steps_used,
                 adam_state_kept_iterations=kept_state, adam_state_resumed_from_earlier_weight=resumed_state,
                 adam_states_stored=len(state_opt))
    ls.WARM_LP = False; ls.LP_CG = False
    return rec.finish(path, X[:n], F[:n], J[:n], extra)
