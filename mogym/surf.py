"""SURF (Jiang et al., Algorithm 1), K=2, with the shared Adam inner solver, run to the GN plateau.

N segments -> N+1 ordered weights w_n = Phi_t^{-1}(n/N), Phi_0(w) = w, lambda_n = (1 - w_n, w_n).  Each round runs
`inner_steps` Adam steps for every slot from its previous iterate (all slots start at theta_0), measures the chord
lengths between neighbouring slots' objective values F = (F_1, F_2) (the values returned with the last gradient of
each slot; SURF eq. (12) uses h(u_n), and F = (1 - gamma) h), interpolates the normalized arc length with PCHIP on a
fine w-grid and damps the CDF update, Phi_{t+1} = alpha Phi~_t + (1 - alpha) Phi_t.  The output of a round is its
N+1 policies.  A slot keeps its Adam state while its weight moves by at most state_tol from one round to the next
and gets a new state otherwise.

The run stops by the rule of mogym.plateau on the per-round GN; readiness: the slot weights (including the next
round's) moved <= weight_tol over the window, and P95 over slots of ||grad F_lambda_n(slot n)|| <= max(own_floor,
own_ratio * GN).  `rounds` is a safety cap.  theta_0 is evaluated once, inside the training time, and counts K
Gradient Calls; each Adam step counts K.

Two kinds of checkpoints (field "kind"): "calls" at the end of the first slot past every `every` Gradient Calls up
to `budget` and every 10 x `every` afterwards (the schedule shared with GRAB; the bundle is the current N+1 slot
policies), on which the plotted point is located; "round" at the end of every round, the values of the stopping
rule.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

from . import plateau as plateau_rule
from .adam import Adam
from .oracle import Oracle
from .recorder import Recorder


def surf(model, N_segments, path, *, rounds, inner_steps, inner_lr, every, budget, rule, state_tol, alpha=0.3,
         fine_grid=2001, save_arrays=True, run_spec=None):
    if model['K'] != 2:
        raise ValueError('SURF Algorithm 1 is defined for K=2')
    K, d = model['K'], model['d']
    oracle = Oracle(model)
    rec = Recorder(oracle, dict(method='SURF', N_segments=N_segments, rounds=rounds, alpha=alpha,
                                inner_steps=inner_steps, inner_lr=inner_lr, fine_grid=fine_grid, plateau_rule=rule,
                                state_tol=state_tol, checkpoint_every=every, checkpoint_budget=budget),
                   model=model, run_spec=run_spec, save_arrays=save_arrays)
    steps = 0
    x0 = np.zeros(d)
    f0, j0 = oracle(x0)  # theta_0: evaluated once, counted (K), inside the training time
    rec.checkpoint(1, f0[None], j0[None], K, kind='start')
    quantiles = np.linspace(0.0, 1.0, N_segments + 1)
    fine_w = np.linspace(0.0, 1.0, fine_grid)
    F_vals = fine_w.copy()
    weight_history, own_p95, round_gn = [], [], []
    status, trigger = 'safety_cap', None
    slots = [(x0.copy(), f0, j0) for _ in quantiles]
    slot_opts, slot_w = [None] * len(quantiles), [None] * len(quantiles)
    mark = every
    for _ in range(rounds):
        current_w = np.interp(quantiles, F_vals, fine_w)
        weight_history.append(current_w.copy())
        f_coords = []
        for slot, w in enumerate(current_w):
            lam = np.array([1.0 - w, w])
            x, f, j = slots[slot]
            g = j.T @ lam
            if slot_opts[slot] is not None and abs(w - slot_w[slot]) <= state_tol:
                opt = slot_opts[slot]
            else:
                opt = Adam(d, inner_lr)
            slot_opts[slot], slot_w[slot] = opt, float(w)
            for _ in range(inner_steps):
                x = opt.step(x, g)
                f, j = oracle(x); steps += 1
                g = j.T @ lam
            slots[slot] = (x, f, j)
            f_coords.append([f[0], f[1]])  # front point: the objective values at the slot's last iterate
            if K * (steps + 1) >= mark:  # Gradient-Call checkpoint: the current slot policies
                rec.checkpoint(len(slots), np.array([z[1] for z in slots]), np.array([z[2] for z in slots]),
                               K * (steps + 1), kind='calls')
                while mark <= K * (steps + 1):
                    mark += every if mark < budget else 10 * every
        f_coords = np.asarray(f_coords, dtype=float)
        seg_lens = np.sqrt(np.sum(np.diff(f_coords, axis=0) ** 2, axis=1))
        s_vals = np.concatenate([[0.0], np.cumsum(seg_lens)])
        if s_vals[-1] > 1e-14:
            tilde_vals = PchipInterpolator(current_w, s_vals / s_vals[-1])(fine_w)
            F_vals = (1.0 - alpha) * F_vals + alpha * tilde_vals
            F_vals = np.maximum.accumulate(F_vals)
            F_vals[0], F_vals[-1] = 0.0, 1.0
        count = K * (steps + 1)
        gn = rec.checkpoint(len(slots), np.array([z[1] for z in slots]), np.array([z[2] for z in slots]), count,
                            kind='round')
        round_gn.append(gn)
        nxt = np.interp(quantiles, F_vals, fine_w)
        ws = weight_history[-rule["window"]:] + [nxt]
        moves = [float(np.max(np.abs(a - b))) for a, b in zip(ws[1:], ws[:-1])]
        ready = len(moves) == rule["window"] and max(moves) <= rule["weight_tol"]
        lam = np.column_stack([1.0 - current_w, current_w])
        own = np.linalg.norm(np.einsum('nkd,nk->nd', rec.last_J, lam), axis=1)
        own_p95.append(float(np.quantile(own, .95)))
        ready = ready and own_p95[-1] <= max(rule["own_floor"], rule["own_ratio"] * gn)
        trigger, stop = plateau_rule.update(round_gn, trigger, ready, rule)
        if stop:
            status = 'plateau'
            break
        if s_vals[-1] <= 1e-14:  # degenerate front (as in the SURF notebooks)
            status = 'zero_arc_length'
            break
    end = rec.rows[-1]
    print(f"{model['name']} SURF N={N_segments} rounds={len(weight_history)} {status} calls={end['component_gradients']} "
          f"GN={end['gn']:.5g}", flush=True)
    return rec.finish(path, np.asarray([z[0] for z in slots]), rec.last_F, rec.last_J, dict(
        status=status, rounds=len(weight_history), stop_calls=end['component_gradients'], stop_cpu=end['train_cpu'],
        stop_gn=end['gn'], own_gradient_p95_history=own_p95))
