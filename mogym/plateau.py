"""Run-to-plateau stopping rule (Uniform and SURF) and the plotted point.

g_1, g_2, ... are the GN values of the method's output at its checks: after every block of sweeps (Uniform) or rounds
(SURF) that gives each weight at least rule["check_steps"] Adam steps (every sweep / round if steps >= check_steps).
The rule triggers at check s when the last `window` relative changes |g_j - g_{j-1}| / g_{j-1} are
all <= tol and the method's readiness flag holds.  After `confirm` further checkpoints the trigger is
accepted if every value since the trigger is within tol of the latest one; otherwise it is cancelled.
"""


def relative_changes(g, window):
    return [abs(g[-k] - g[-k - 1]) / max(g[-k - 1], 1e-300) for k in range(1, window + 1)]


def block(rule, steps):
    """Sweeps (rounds) per check for `steps` Adam steps per weight in each sweep (round)."""
    return max(1, -(-int(rule.get("check_steps", 1)) // int(steps)))


def update(g, trigger, ready, rule):
    """g: values at the checks so far (s = len(g)).  Returns (trigger, stop)."""
    s = len(g)
    if trigger is not None and s - trigger >= rule["confirm"]:
        band = g[trigger - 1:]
        if all(abs(x - g[-1]) <= rule["tol"] * g[-1] for x in band):
            return trigger, True
        trigger = None
    if trigger is None and s >= rule["window"] + 1 and ready \
            and max(relative_changes(g, rule["window"])) <= rule["tol"]:
        trigger = s
    return trigger, False


def onset(g, tol):
    """0-based index of the earliest checkpoint from which every later value stays within tol of g[-1]."""
    k = len(g) - 1
    while k > 0 and abs(g[k - 1] - g[-1]) <= tol * g[-1]:
        k -= 1
    return k
