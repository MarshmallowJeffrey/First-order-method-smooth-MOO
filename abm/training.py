"""What all methods share: one SVRG segment, the budget meter and the record of a run.

Segment.  From an anchor (x, F(x), J(x)) at weight lambda: the full gradient g = J^T lambda, then m = ceil(n / b)
steps y <- step(y, g_S(y) - g_S(x) + g) on stratified mini-batches S of b = 1,024 images, then one joint call at
the end point y.  The segment is accepted unless F_lambda(y) > F_lambda(x) + 1e-10 (1 + |F_lambda(x)|).

Budget.  A joint call costs K gradient calls, a mini-batch gradient on b images b K / n.  A segment is started only
if its largest possible cost still fits.

Record.  Every segment end point enters the bundle: its Gram matrix J J^T, its values F, the budget spent and its
lambda.  At checkpoints (a budget-dependent cadence) the bundle size, the budget and the wall-clock time are
stored; the audits of the worst-case gradient norm run after training, on the bundle prefixes of the checkpoints.
"""

from __future__ import annotations

import time

import numpy as np

MAX_RETRIES = 4          # the 5th rejection in a row is accepted


def check_finite(f, J):
    f, J = np.asarray(f, dtype=float), np.asarray(J, dtype=float)
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(J))):
        raise FloatingPointError("the joint oracle returned non-finite values")
    return f, J


class Budget:
    def __init__(self, K, n, stoch, limit):
        self.K, self.n = K, n
        self.stoch = stoch
        self.ifo0 = stoch.ifo_count
        self.joint_calls = 0
        self.limit = float(limit)

    def spent(self):
        return self.joint_calls * self.K + (self.stoch.ifo_count - self.ifo0) * self.K / float(self.n)

    def allows_segment(self, epoch_len, batch_size):
        upper = epoch_len * 2.0 * batch_size * self.K / float(self.n) + self.K
        return self.spent() + upper <= self.limit + 1e-9


def ck_step(spent_at_last_ck, schedule):
    """Gradient calls until the next checkpoint; schedule = [(up_to, step), ...]."""
    for upto, step in schedule:
        if spent_at_last_ck < upto:
            return float(step)
    return float(schedule[-1][1])


def run_segment(problem, stepper, anchor, lam, L_lam, L_scale, epoch_len):
    """One SVRG segment from anchor = (x, f, J).  Returns (y, f_y, J_y, accepted)."""
    x, f, J = anchor
    g_a_full = J.T @ lam
    F_a = float(f @ lam)
    stepper.start_segment(x, g_a_full, L_lam, L_scale, epoch_len)
    stoch = problem.stoch
    stoch.set_anchor(x)
    y = x.copy()
    for _ in range(epoch_len):
        g_y_S, g_a_S = stoch.grad_pair(y, lam, stoch.sample_batch())
        y = stepper.step(y, (g_y_S - g_a_S + g_a_full))
    f_y, J_y = check_finite(*problem.joint(y))
    accepted = not (float(f_y @ lam) > F_a + 1e-10 * (1.0 + abs(F_a)))
    return y, f_y, J_y, accepted


class RunRecord:
    """Bundle, budget meter and checkpoints of one run (the bundle starts with theta_0)."""

    def __init__(self, problem, x0, budget, schedule):
        f0, J0 = check_finite(*problem.joint(x0))
        self.K = problem.K
        self.x0, self.f0, self.J0 = x0, f0, J0
        self.grams = [J0 @ J0.T]
        self.fvals = [np.asarray(f0, dtype=float)]
        self.seg_grads = [0.0]
        self.seg_lams = [[np.nan] * problem.K]
        self.budget = Budget(problem.K, problem.n, problem.stoch, budget)
        self.schedule = schedule
        self.ck_grads, self.ck_wall, self.ck_m = [0.0], [0.0], [1]
        self._grad_at_ck = 0.0
        self.rejections = 0
        self.decision_seconds = 0.0          # lambda search (adaptive) or slot placement (SURF)
        self.wall_seconds = None
        self.surf_rounds = None
        self.t0 = None

    def start_clock(self):
        self.t0 = time.time()

    def add(self, f_y, J_y, lam):
        self.budget.joint_calls += 1
        self.grams.append(J_y @ J_y.T)
        self.fvals.append(np.asarray(f_y, dtype=float))
        self.seg_grads.append(float(self.budget.spent()))
        self.seg_lams.append([float(t) for t in lam])

    def checkpoint_if_due(self):
        spent = self.budget.spent()
        if spent - self._grad_at_ck >= ck_step(self._grad_at_ck, self.schedule):
            self._grad_at_ck = spent
            self.ck_grads.append(spent)
            self.ck_wall.append(time.time() - self.t0)
            self.ck_m.append(len(self.grams))

    def finish(self):
        """Final checkpoint at the end of training."""
        self.wall_seconds = time.time() - self.t0
        self.ck_grads.append(self.budget.spent())
        self.ck_wall.append(self.wall_seconds)
        self.ck_m.append(len(self.grams))
