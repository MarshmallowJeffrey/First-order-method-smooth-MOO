"""Step rules: how one SVRG segment turns the variance-reduced directions v_t into parameter updates.

Contract with the methods (per run and per weight):

    st = make_stepper(name, d, params)
    new weight lambda   ->  st.on_lambda_change(lam, L_lam, L_scale)   (called only when lambda changes)
    segment start       ->  st.start_segment(anchor_x, g_a_full, L_lam, L_scale, epoch_len)
    every step          ->  y = st.step(y, v)
    segment end         ->  st.on_segment_result(accepted, L_lam, L_scale)

L_lam = lambda^T L is the smoothness estimate of F_lambda and L_scale a factor that doubles at every rejected
segment (it only enters the rules that use L).  No rule draws random numbers.

    const    eta = c / (L_lam L_scale) with heavy-ball momentum beta
    bb       Barzilai-Borwein step per segment from consecutive anchors at the same lambda (clipped), else const
    adagrad  per-coordinate cumulative scaling, started at G0 = (L_lam L_scale / c)^2, reset on a new lambda and
             after a rejection
    adam     Adam with bias correction; moments cleared on a new lambda; a rejection clears the moments and
             halves alpha
"""

from __future__ import annotations

import numpy as np

# c and beta of the constant-step rule (also used by bb and adagrad)
BASE_PARAMS = {"step_const": 0.1, "momentum": 0.5}

# the eleven rules of the step-rule experiment: (tag, name, parameters)
STEP_RULES = [
    ("const", "const", {}),
    ("bb", "bb", {}),
    ("adagrad_mult1", "adagrad", {"adagrad_alpha_mult": 1.0}),
    ("adagrad_mult3", "adagrad", {"adagrad_alpha_mult": 3.0}),
    ("adagrad_mult10", "adagrad", {"adagrad_alpha_mult": 10.0}),
    ("adam_alpha0.0001_beta20.9", "adam", {"adam_alpha": 1e-4, "adam_beta2": 0.9}),
    ("adam_alpha0.0001_beta20.99", "adam", {"adam_alpha": 1e-4, "adam_beta2": 0.99}),
    ("adam_alpha0.0003_beta20.9", "adam", {"adam_alpha": 3e-4, "adam_beta2": 0.9}),
    ("adam_alpha0.0003_beta20.99", "adam", {"adam_alpha": 3e-4, "adam_beta2": 0.99}),
    ("adam_alpha0.001_beta20.9", "adam", {"adam_alpha": 1e-3, "adam_beta2": 0.9}),
    ("adam_alpha0.001_beta20.99", "adam", {"adam_alpha": 1e-3, "adam_beta2": 0.99}),
]
STEP_RULE_BY_TAG = {tag: (name, params) for tag, name, params in STEP_RULES}


class ConstStepper:
    name = "const"

    def __init__(self, d, cfg):
        self.d = d
        self.c = float(cfg["step_const"])
        self.beta = float(cfg["momentum"])
        self.eta = None
        self.u = None

    def on_lambda_change(self, lam, L_lam, L_scale):
        pass

    def start_segment(self, anchor_x, g_a_full, L_lam, L_scale, epoch_len):
        self.eta = self.c / (L_lam * L_scale)
        self.u = np.zeros(self.d)

    def step(self, y, v):
        self.u = self.beta * self.u + v
        return y - self.eta * self.u

    def on_segment_result(self, accepted, L_lam, L_scale):
        pass


class BBStepper(ConstStepper):
    """Scalar BB step per segment from consecutive accepted anchors at the same lambda; regularized denominator
    and clip; the constant step on the first segment at a lambda, when ||s||^2 = 0 and right after a rejection."""

    name = "bb"

    def __init__(self, d, cfg):
        super().__init__(d, cfg)
        self.delta_rel = float(cfg.get("bb_delta_rel", 1e-3))
        self.c_min, self.c_max = cfg.get("bb_clip", (0.01, 1.0))
        self.x_prev = None
        self.g_prev = None
        self.retry_pending = False

    def on_lambda_change(self, lam, L_lam, L_scale):
        self.x_prev = None
        self.g_prev = None
        self.retry_pending = False

    def start_segment(self, anchor_x, g_a_full, L_lam, L_scale, epoch_len):
        L_hat = L_lam * L_scale
        eta = self.c / L_hat
        if (self.x_prev is not None) and not self.retry_pending:
            s = anchor_x - self.x_prev
            ss = float(s @ s)
            if ss > 0.0:
                r = g_a_full - self.g_prev
                D = max(float(s @ r), self.delta_rel * L_lam * ss)
                eta_bb = (1.0 - self.beta) * ss / (epoch_len * D)
                eta = min(max(eta_bb, self.c_min / L_hat), self.c_max / L_hat)
        self.retry_pending = False
        self.x_prev = anchor_x.copy()
        self.g_prev = g_a_full.copy()
        self.eta = eta
        self.u = np.zeros(self.d)

    def on_segment_result(self, accepted, L_lam, L_scale):
        if not accepted:
            self.retry_pending = True


class AdaGradStepper:
    name = "adagrad"

    def __init__(self, d, cfg):
        self.d = d
        self.c = float(cfg["step_const"])
        self.beta = float(cfg["momentum"])
        self.alpha_mult = float(cfg.get("adagrad_alpha_mult", 1.0))
        self.eps = float(cfg.get("adagrad_eps", 1e-12))
        self.G = None
        self.u = None

    def _init_G(self, L_lam, L_scale):
        L_hat = L_lam * L_scale
        self.G = np.full(self.d, (L_hat / self.c) ** 2)

    def on_lambda_change(self, lam, L_lam, L_scale):
        self._init_G(L_lam, L_scale)

    def start_segment(self, anchor_x, g_a_full, L_lam, L_scale, epoch_len):
        if self.G is None:
            self._init_G(L_lam, L_scale)
        self.u = np.zeros(self.d)

    def step(self, y, v):
        self.G = self.G + v * v
        self.u = self.beta * self.u + v
        return y - self.alpha_mult * self.u / (np.sqrt(self.G) + self.eps)

    def on_segment_result(self, accepted, L_lam, L_scale):
        if not accepted:
            self._init_G(L_lam, L_scale)          # L_scale is already doubled


class AdamStepper:
    """The first moment is the momentum (no heavy ball on top).  State (m, G, t) persists across the segments at
    one lambda."""

    name = "adam"

    def __init__(self, d, cfg):
        self.d = d
        self.alpha0 = float(cfg.get("adam_alpha", 3e-4))
        self.b1 = float(cfg.get("adam_beta1", 0.9))
        self.b2 = float(cfg.get("adam_beta2", 0.99))
        self.eps = float(cfg.get("adam_eps", 1e-8))
        self.alpha = self.alpha0
        self._clear()

    def _clear(self):
        self.m = np.zeros(self.d)
        self.G = np.zeros(self.d)
        self.t = 0

    def on_lambda_change(self, lam, L_lam, L_scale):
        self._clear()
        self.alpha = self.alpha0

    def start_segment(self, anchor_x, g_a_full, L_lam, L_scale, epoch_len):
        pass

    def step(self, y, v):
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * v
        self.G = self.b2 * self.G + (1.0 - self.b2) * (v * v)
        mhat = self.m / (1.0 - self.b1 ** self.t)
        Ghat = self.G / (1.0 - self.b2 ** self.t)
        return y - self.alpha * mhat / (np.sqrt(Ghat) + self.eps)

    def on_segment_result(self, accepted, L_lam, L_scale):
        if not accepted:
            self._clear()
            self.alpha *= 0.5


_CLASSES = {"const": ConstStepper, "bb": BBStepper, "adagrad": AdaGradStepper, "adam": AdamStepper}


def make_stepper(name: str, d: int, params: dict | None = None):
    cfg = dict(BASE_PARAMS)
    cfg.update(params or {})
    return _CLASSES[name](d, cfg)
