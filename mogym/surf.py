"""SURF (Jiang et al., Algorithm 1), K=2, with the shared Adam inner solver.

N segments -> N+1 ordered weights w_n = Phi_t^{-1}(n/N), Phi_0(w) = w, lambda_n = (1 - w_n, w_n).  Each
round runs `inner_steps` Adam steps for every slot from its previous iterate (all slots start at
theta_0), measures reward-space chord lengths between neighbouring slots, interpolates the normalized
arc length with PCHIP on a fine w-grid and damps the CDF update, Phi_{t+1} = alpha Phi~_t + (1-alpha) Phi_t.
The output of a round is its N+1 policies.  A slot keeps its Adam state while its weight moves by at most
adam_keep_tol and gets a new state otherwise.

Instead of a fixed number of rounds T, the run stops by the rule of mogym.plateau on the per-round GN
(plateau != None); readiness: the slot weights (including the next round's) moved <= weight_tol over the
window, and P95 over slots of ||grad F_lambda_n(slot n)|| <= max(own_floor, own_ratio * GN).  `rounds` is a
safety cap.  Each Adam step counts as K Gradient Calls; the initial theta_0 gradient adds K.

Two kinds of checkpoints (field "kind"): "calls" at the end of the first slot past every `every` Gradient
Calls up to `budget` and every 10 x `every` afterwards (the schedule shared with GRAB; the bundle is the
current N+1 slot policies), on which the plotted point is located; "round" at the end of every round, the
values of the stopping rule.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

from . import plateau as plateau_rule
from .adam import Adam
from .oracle import Oracle
from .recorder import Recorder


def surf(model, N_segments, path, *, rounds, inner_steps, inner_lr, every, budget, alpha=0.3, fine_grid=2001,
         plateau=None, adam_keep_tol=None, save_arrays=True):
    if model['K'] != 2:
        raise ValueError('SURF Algorithm 1 is defined for K=2')
    K, d = model['K'], model['d']
    oracle = Oracle(model)
    rec = Recorder(oracle, dict(method='SURF', N_segments=N_segments, N_points=N_segments + 1, rounds=rounds,
                                alpha=alpha, inner_steps=inner_steps, inner_lr=inner_lr, fine_grid=fine_grid,
                                plateau_rule=plateau, adam_keep_tol=adam_keep_tol, checkpoint_every=every,
                                checkpoint_budget=budget), save_arrays=save_arrays)
    steps = 0
    rec.checkpoint(np.zeros((1, d)), count=K); rec.rows[-1]['kind'] = 'start'
    quantiles = np.linspace(0.0, 1.0, N_segments + 1)
    fine_w = np.linspace(0.0, 1.0, fine_grid)
    F_vals = fine_w.copy()
    weight_history, own_p95 = [], []
    status = 'fixed_rounds' if plateau is None else 'safety_cap'
    trigger = None
    kept_state = 0
    x0 = np.zeros(d)
    f0, j0 = oracle(x0)  # the initial gradient (counted as K above)
    slots = [(x0.copy(), f0, j0) for _ in quantiles]
    current_logits = [x0.copy() for _ in quantiles]
    slot_opts, slot_w = [None] * len(quantiles), [None] * len(quantiles)
    mark, round_gn = every, []
    for outer in range(1, rounds + 1):
        current_w = np.interp(quantiles, F_vals, fine_w)
        weight_history.append(current_w.copy())
        f_coords = []
        for slot, w in enumerate(current_w):
            lam = np.array([1.0 - w, w])
            x, f, j = slots[slot]
            g = j.T @ lam
            if adam_keep_tol is not None and slot_opts[slot] is not None and abs(w - slot_w[slot]) <= adam_keep_tol:
                opt = slot_opts[slot]; kept_state += 1
            else:
                opt = Adam(d, inner_lr)
            slot_opts[slot], slot_w[slot] = opt, float(w)
            for _ in range(inner_steps):
                x = opt.step(x, g)
                f, j = oracle(x); steps += 1
                g = j.T @ lam
            slots[slot] = (x, f, j)
            current_logits[slot] = x
            rr = oracle.evaluate(x, gradient=False)[1]  # reward vector (R1, R2, KL term): front point
            f_coords.append([rr[0], rr[1]])
            if K * (steps + 1) >= mark:  # Gradient-Call checkpoint: the current slot policies
                rec.checkpoint(np.asarray([np.asarray(z[0]).reshape(-1) for z in slots], float),
                               np.array([z[1] for z in slots]), np.array([z[2] for z in slots]), K * (steps + 1))
                rec.rows[-1]['kind'] = 'calls'
                while mark <= K * (steps + 1):
                    mark += every if mark < budget else 10 * every
        f_coords = np.asarray(f_coords, dtype=np.float32)
        seg_lens = np.sqrt(np.sum(np.diff(f_coords, axis=0) ** 2, axis=1))
        s_vals = np.concatenate([[0.0], np.cumsum(seg_lens)])
        if s_vals[-1] > 1e-14:
            tilde_vals = PchipInterpolator(current_w, s_vals / s_vals[-1])(fine_w)
            F_vals = (1.0 - alpha) * F_vals + alpha * tilde_vals
            F_vals = np.maximum.accumulate(F_vals)
            F_vals[0], F_vals[-1] = 0.0, 1.0
        theta = np.asarray([np.asarray(x).reshape(-1) for x in current_logits], float)
        count = K * (steps + 1)
        gn = rec.checkpoint(theta, np.array([z[1] for z in slots]), np.array([z[2] for z in slots]), count)
        rec.rows[-1]['kind'] = 'round'; round_gn.append(gn)
        if outer == 1 or outer % 5 == 0 or outer == rounds:
            print(f"{model['name']} SURF N={N_segments} round={outer} calls={count} GN={gn:.6g}", flush=True)
        if plateau is not None:
            nxt = np.interp(quantiles, F_vals, fine_w)
            ws = weight_history[-plateau["window"]:] + [nxt]
            moves = [float(np.max(np.abs(a - b))) for a, b in zip(ws[1:], ws[:-1])]
            ready = len(moves) == plateau["window"] and max(moves) <= plateau["weight_tol"]
            lam = np.column_stack([1.0 - current_w, current_w])
            own = np.linalg.norm(np.einsum('nkd,nk->nd', rec.last_J, lam), axis=1)
            own_p95.append(float(np.quantile(own, .95)))
            ready = ready and own_p95[-1] <= max(plateau["own_floor"], plateau["own_ratio"] * gn)
            trigger, stop = plateau_rule.update(round_gn, trigger, ready, plateau)
            if stop:
                status = 'plateau'
                break
        if s_vals[-1] <= 1e-14:  # degenerate front (as in the SURF notebooks)
            status = 'zero_arc_length'
            break
    theta = np.asarray([np.asarray(x).reshape(-1) for x in current_logits], float)
    return rec.finish(path, theta, rec.last_F, rec.last_J, dict(
        status=status, outer_rounds=len(weight_history), scalar_backward_steps=steps,
        trigger_round=trigger if status == 'plateau' else None,
        stop_calls=rec.rows[-1]['component_gradients'], stop_cpu=rec.rows[-1]['train_cpu'],
        stop_gn=rec.rows[-1]['gn'], own_gradient_p95_history=own_p95, adam_state_kept_calls=kept_state,
        weight_history=[x.tolist() for x in weight_history], trained_final_weights=weight_history[-1].tolist()))
