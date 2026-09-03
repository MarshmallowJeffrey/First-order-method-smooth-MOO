"""stepper_core.py — shared inner-loop walk rules ("steppers") for the
K = 2 pair campaign v2 (design: ADAPTIVE_STEPPERS.md, Sep-2 revision;
campaign plan: Note/Sep_2_note.md).

NEW FILE (Sep 2, 2026).  One stepper = one rule for turning the SVRG
corrected gradient v_t into a parameter update inside one MSVRG segment.
Everything around the walk (SVRG correction, descent safeguard,
budget accounting, delivery) stays in the calling executor.

Contract with the executor (per leg):

    st = make_stepper(name, d, cfg)
    ...
    lam decided ->  st.on_lambda_change(lam, L_lam, L_scale)
                    (called only when lam actually CHANGED; state that is
                    per-lambda — BB secant memory, AdaGrad G, Adam
                    moments — is reset here)
    per segment ->  st.start_segment(anchor_x, g_a_full, L_lam, L_scale,
                                     epoch_len)
    per step    ->  y = st.step(y, v)        # v = g_S(y) - g_S(anchor) + g_full
    segment end ->  st.on_segment_result(accepted, L_lam, L_scale)
                    (called AFTER the executor doubled L_scale on an
                    ascent, so re-initialisations see the new scale)

Bit-compatibility promise (Gate 0): with name = "const" the sequence of
floating-point operations is IDENTICAL to the incumbent executor's
inline walk (u = mom*u + v; y = y - eta*u with eta = c/(L_lam*L_scale),
u reset to zeros at each segment start), and no stepper consumes any
random numbers.

Steppers (formulas in ADAPTIVE_STEPPERS.md):

    const    eta = c/L_hat, heavy-ball beta          (the incumbent rule)
    bb       scalar eta_k per segment from the regularized secant
             eta_k = clip((1-beta)*||s||^2 / (m*max(s'r, delta*||s||^2)),
                          c_min/L_hat, c_max/L_hat), fallbacks -> const
    adagrad  per-coordinate, cumulative G, warm start G0 = (L_hat/c)^2
    adam     EMA moments with bias correction, short beta2 memory
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_stepper", "STEPPER_NAMES"]

STEPPER_NAMES = ("const", "bb", "adagrad", "adam")


class ConstStepper:
    """The incumbent walk: eta = c/L_hat, heavy-ball momentum."""

    name = "const"

    def __init__(self, d, cfg):
        self.d = d
        self.c = float(cfg["msvrg_step_const"])
        self.beta = float(cfg["msvrg_momentum"])
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

    def diag(self):
        return {"eta": float(self.eta) if self.eta is not None else None}


class BBStepper(ConstStepper):
    """Scalar BB step per segment from consecutive accepted anchors under
    the SAME lambda; regularized denominator + clip; const fallback on:
    first segment at a lambda, ||s||^2 ~ 0, or the segment right after an
    ascent."""

    name = "bb"

    def __init__(self, d, cfg):
        super().__init__(d, cfg)
        self.delta_rel = float(cfg.get("bb_delta_rel", 1e-3))
        self.c_min, self.c_max = cfg.get("bb_clip", (0.01, 1.0))
        self.x_prev = None
        self.g_prev = None
        self.retry_pending = False
        self.last_mode = "const"

    def on_lambda_change(self, lam, L_lam, L_scale):
        self.x_prev = None
        self.g_prev = None
        self.retry_pending = False

    def start_segment(self, anchor_x, g_a_full, L_lam, L_scale, epoch_len):
        L_hat = L_lam * L_scale
        eta_const = self.c / L_hat
        eta = eta_const
        mode = "const"
        if (self.x_prev is not None) and not self.retry_pending:
            s = anchor_x - self.x_prev
            ss = float(s @ s)
            if ss > 0.0:
                r = g_a_full - self.g_prev
                delta = self.delta_rel * L_lam
                D = max(float(s @ r), delta * ss)
                eta_bb = (1.0 - self.beta) * ss / (epoch_len * D)
                eta = min(max(eta_bb, self.c_min / L_hat),
                          self.c_max / L_hat)
                mode = "bb"
        self.retry_pending = False
        self.x_prev = anchor_x.copy()
        self.g_prev = g_a_full.copy()
        self.eta = eta
        self.last_mode = mode
        self.u = np.zeros(self.d)

    def on_segment_result(self, accepted, L_lam, L_scale):
        if not accepted:
            self.retry_pending = True

    def diag(self):
        return {"eta": float(self.eta), "mode": self.last_mode}


class AdaGradStepper:
    """Per-coordinate cumulative scaling with the G0 warm start: the first
    step equals alpha_mult * (c/L_hat) on every coordinate, afterwards
    coordinates only shrink selectively.  G persists across segments at
    the same lambda; re-initialised on lambda change and after an ascent
    (with the then-current, already doubled, L_scale)."""

    name = "adagrad"

    def __init__(self, d, cfg):
        self.d = d
        self.c = float(cfg["msvrg_step_const"])
        self.beta = float(cfg["msvrg_momentum"])
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
            self._init_G(L_lam, L_scale)   # L_scale already doubled

    def diag(self):
        g = np.sqrt(self.G) if self.G is not None else None
        return {"eff_step_min": float(self.alpha_mult / (g.max() + self.eps))
                if g is not None else None,
                "eff_step_max": float(self.alpha_mult / (g.min() + self.eps))
                if g is not None else None}


class AdamStepper:
    """EMA moments with bias correction; the EMA first moment IS the
    momentum (no extra heavy-ball on top).  State (m, G, t) persists
    across segments at the same lambda; cleared on lambda change; on an
    ascent the moments are cleared and alpha is halved (per-lambda)."""

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

    def diag(self):
        return {"alpha": float(self.alpha), "t": int(self.t)}


_CLASSES = {"const": ConstStepper, "bb": BBStepper,
            "adagrad": AdaGradStepper, "adam": AdamStepper}


def make_stepper(name, d, cfg):
    """cfg needs msvrg_step_const + msvrg_momentum, plus the stepper's own
    keys (bb_delta_rel/bb_clip, adagrad_alpha_mult/adagrad_eps,
    adam_alpha/adam_beta1/adam_beta2/adam_eps) — defaults per
    ADAPTIVE_STEPPERS.md."""
    if name not in _CLASSES:
        raise ValueError(f"unknown stepper {name!r}; "
                         f"choose from {STEPPER_NAMES}")
    return _CLASSES[name](d, cfg)
