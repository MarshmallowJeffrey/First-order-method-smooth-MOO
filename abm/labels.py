"""Number labels next to the markers of the trend figures.

Every candidate position of every label is scored and the cheapest kept.
  hard (accepted only when nothing else exists): outside the axes, on the legend, on a marker, on another label,
      a leader line through another marker or label;
  soft: covering a curve (per covered sample), the distance from its own marker, a curve between the label and its
      marker, a label nearly as close to another marker as to its own, a leader line (fixed cost plus crossings).
Candidates: 16 directions x 9 gaps (1.5 to 56 pt).  Labels are placed greedily, most crowded marker first, improved
by simulated annealing (fixed seed, 400 moves per label) and a last greedy pass.  A label more than 6 pt from its
marker, or nearer to another marker than to its own, gets a thin leader line.  Only text is moved.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

GAPS_PT = (1.5, 4.0, 8.0, 13.0, 19.0, 26.0, 34.0, 44.0, 56.0)
ANGLES = np.deg2rad(np.arange(0.0, 360.0, 22.5))
LEADER_PT = 6.0
HARD = 1e6


def _overlap(b, c):
    return not (b[2] <= c[0] or b[0] >= c[2] or b[3] <= c[1] or b[1] >= c[3])


def _seg_box(p, q, b):
    """does the segment p->q pass through the box b = (x0, y0, x1, y1)? (Liang-Barsky)"""
    x0, y0, x1, y1 = b
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pk == 0:
            if qk < 0:
                return False
        else:
            t = qk / pk
            if pk < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    return t0 <= t1


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_seg(p, q, a, b):
    d1, d2, d3, d4 = _orient(p, q, a), _orient(p, q, b), _orient(a, b, p), _orient(a, b, q)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0) and 0 not in (d1, d2, d3, d4)


def place(ax, items, lines=(), fontsize=9.5, marker_pt=9.0, halo=True, manual=None, debug=False, explain=None):
    """items: [{"x", "y", "text", "color", optional "key"}] in data coordinates; lines: [(xs, ys)] curves (data
    coordinates) that labels should not cover; manual: {key: (dx_pt, dy_pt)} label-centre offsets fixed by hand.
    Returns {"n_leader": int, "hard": [texts that could not avoid a hard clash]}."""
    fig = ax.figure
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    pxpt = fig.dpi / 72.0
    T = ax.transData.transform
    axbb = ax.get_window_extent(ren)
    n = len(items)
    P = T(np.array([(it["x"], it["y"]) for it in items], dtype=float))
    rm = marker_pt / 2.0 * pxpt + 1.0
    MB = [(P[j, 0] - rm, P[j, 1] - rm, P[j, 0] + rm, P[j, 1] + rm) for j in range(n)]
    pad = 0.1 * fontsize * pxpt if halo else 0.5
    W, H = [], []
    for it in items:
        t = ax.text(0, 0, it["text"], fontsize=fontsize)
        bb = t.get_window_extent(ren)
        t.remove()
        W.append(bb.width + 2 * pad)
        H.append(bb.height + 2 * pad)

    S = []
    for lx, ly in lines:
        q = T(np.column_stack([np.asarray(lx, dtype=float), np.asarray(ly, dtype=float)]))
        q = q[np.all(np.isfinite(q), axis=1)]
        for a, b in zip(q[:-1], q[1:]):
            k = max(1, int(np.ceil(np.hypot(*(b - a)) / 1.5)))
            S.append(a + np.outer(np.arange(k) / k, b - a))
        if len(q):
            S.append(q[-1:])
    S = np.vstack(S) if S else np.zeros((0, 2))
    S = S[(S[:, 0] >= axbb.x0) & (S[:, 0] <= axbb.x1) & (S[:, 1] >= axbb.y0) & (S[:, 1] <= axbb.y1)]
    tree = cKDTree(S) if len(S) else None
    leg = ax.get_legend()
    LB = None
    if leg is not None:
        lb = leg.get_window_extent(ren)
        LB = (lb.x0 - 3, lb.y0 - 3, lb.x1 + 3, lb.y1 + 3)

    def curve_in_box(b, grow=1.0):
        if tree is None:
            return 0
        idx = tree.query_ball_point(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), np.hypot(b[2] - b[0], b[3] - b[1]) / 2 + grow)
        if not idx:
            return 0
        Q = S[idx]
        return int(np.sum((Q[:, 0] >= b[0] - grow) & (Q[:, 0] <= b[2] + grow) & (Q[:, 1] >= b[1] - grow) & (Q[:, 1] <= b[3] + grow)))

    def curve_crossings(p, q, tol=1.2):
        if tree is None:
            return 0
        k = max(2, int(np.hypot(q[0] - p[0], q[1] - p[1]) / 1.5))
        hit = np.array([len(h) > 0 for h in tree.query_ball_point(np.linspace(p, q, k), tol)])
        return int(hit[0]) + int(np.sum(hit[1:] & ~hit[:-1]))

    gap_of = {}

    def _gap_of(i, rec):
        return gap_of.get((i, rec[1]), 0.0)

    def gap_crossings(i, box, ctr, tol=1.2):
        """curves crossed between the marker edge and the label box (the parts inside the marker and inside the box
        are ignored, so a curve through the marker itself does not count)"""
        if tree is None:
            return 0
        k = max(2, int(np.hypot(ctr[0] - P[i, 0], ctr[1] - P[i, 1]) / 1.0))
        pts = np.linspace(P[i], ctr, k)
        keep = ~((np.abs(pts[:, 0] - P[i, 0]) <= rm + 1) & (np.abs(pts[:, 1] - P[i, 1]) <= rm + 1))
        keep &= ~((pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3]))
        pts = pts[keep]
        if not len(pts):
            return 0
        hit = np.array([len(h) > 0 for h in tree.query_ball_point(pts, tol)])
        return int(hit[0]) + int(np.sum(hit[1:] & ~hit[:-1]))

    def candidates(i):
        hw, hh = W[i] / 2, H[i] / 2
        for g in GAPS_PT:
            gp = g * pxpt
            for th in ANGLES:
                cx = P[i, 0] + np.cos(th) * (rm + gp + hw)
                cy = P[i, 1] + np.sin(th) * (rm + gp + hh)
                box = (cx - hw, cy - hh, cx + hw, cy + hh)
                gap_of[(i, box)] = g
                yield g, box

    boxes, leaders = {}, {}

    def cost(i, g, box, why=None):
        c = 0.0
        if box[0] < axbb.x0 + 1 or box[2] > axbb.x1 - 1 or box[1] < axbb.y0 + 1 or box[3] > axbb.y1 - 1:
            c += HARD
            if why is not None: why.append("outside")
        if LB is not None and _overlap(box, LB):
            c += HARD
            if why is not None: why.append("legend")
        c += HARD * sum(_overlap(box, MB[j]) for j in range(n))
        c += HARD * sum(_overlap(box, b) for k, b in boxes.items() if k != i)
        c += HARD / 10 * sum(_seg_box(a, e, box) for k, (a, e) in leaders.items() if k != i)
        if why is not None:
            why += [f"marker {items[j]['text']}" for j in range(n) if _overlap(box, MB[j])]
            why += [f"label {items[k]['text']}" for k, b in boxes.items() if k != i and _overlap(box, b)]
            why += [f"leader of {items[k]['text']}" for k, (a, e) in leaders.items() if k != i and _seg_box(a, e, box)]
        c += 25.0 * curve_in_box(box) + 4.0 * g + 0.15 * max(0.0, g - 13.0) ** 2     # far labels cost more and more
        ctr = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        d = np.hypot(P[:, 0] - ctr[0], P[:, 1] - ctr[1])
        lead = g > LEADER_PT or int(np.argmin(d)) != i
        others = np.delete(d, i)
        if len(others):                 # nearly as close to another marker as to its own: ambiguous
            amb = d[i] / max(float(others.min()), 1e-9)
            c += 60.0 * max(0.0, amb - 0.75) / 0.25
        # a curve between a label and its marker separates them;
        # with a leader the crossing is drawn, which is worse
        c += (80.0 if lead else 40.0) * gap_crossings(i, box, ctr)
        if lead:
            c += 30.0
            c += HARD / 10 * sum(_seg_box(ctr, P[i], (MB[j][0] + 1, MB[j][1] + 1, MB[j][2] - 1, MB[j][3] - 1))
                                 for j in range(n) if j != i)
            c += HARD / 10 * sum(_seg_box(ctr, P[i], b) for k, b in boxes.items() if k != i)
            c += 200.0 * sum(_seg_seg(ctr, P[i], a, e) for k, (a, e) in leaders.items() if k != i)
        return c, lead, ctr

    def best(i, explain=False):
        top, allc = None, []
        for g, box in candidates(i):
            c, lead, ctr = cost(i, g, box)
            gap_of[(i, box)] = g
            if explain:
                allc.append((c, g, lead, ((ctr[0] - P[i, 0]) / pxpt, (ctr[1] - P[i, 1]) / pxpt), curve_in_box(box)))
            if top is None or c < top[0]:
                top = (c, box, lead, ctr)
        if explain:
            for c, g, lead, off, ncv in sorted(allc)[:8]:
                print(f"    cand cost {c:10.1f} gap {g:4.1f} leader {lead!s:5s} offset ({off[0]:6.1f},{off[1]:6.1f}) pt curve samples {ncv}")
            for g, box in candidates(i):
                ctr = ((box[0] + box[2]) / 2 - P[i, 0]) / pxpt, ((box[1] + box[3]) / 2 - P[i, 1]) / pxpt
                if g <= 4.0 and ctr[0] > 0:
                    why = []
                    c, lead, _ = cost(i, g, box, why)
                    print(f"    right-side gap {g:3.1f} offset ({ctr[0]:5.1f},{ctr[1]:5.1f}) cost {c:10.1f} {why}")
        return top

    manual = manual or {}
    fixed = {}
    for i, it in enumerate(items):
        if it.get("key") in manual:
            dx, dy = manual[it["key"]]
            cx, cy = P[i, 0] + dx * pxpt, P[i, 1] + dy * pxpt
            box = (cx - W[i] / 2, cy - H[i] / 2, cx + W[i] / 2, cy + H[i] / 2)
            d = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
            lead = (np.hypot(dx, dy) * pxpt - rm - max(W[i], H[i]) / 2) / pxpt > LEADER_PT or int(np.argmin(d)) != i
            boxes[i] = box
            if lead:
                leaders[i] = ((cx, cy), tuple(P[i]))
            fixed[i] = (0.0, box, lead, (cx, cy))
    crowd = [int(np.sum(np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1]) < 45.0)) for i in range(n)]
    order = [i for i in sorted(range(n), key=lambda i: -crowd[i]) if i not in fixed]
    final = dict(fixed)

    def put(i, rec):
        final[i] = rec
        boxes[i] = rec[1]
        leaders.pop(i, None)
        if rec[2]:
            leaders[i] = (rec[3], tuple(P[i]))

    def sweep():
        for i in order:
            boxes.pop(i, None)
            leaders.pop(i, None)
            put(i, best(i))

    sweep()                                             # greedy, most crowded first
    # simulated annealing from the greedy layout (fixed seed): moves one label to a random candidate; a move that
    # changes the total cost by d is accepted with probability min(1, exp(-d / T)).  Moving label i changes the total
    # by cost_i(new) - cost_i(old), because cost_i holds every term that involves label i.
    cands = {i: list(candidates(i)) for i in order}
    rng = np.random.default_rng(0)
    steps = 400 * max(1, len(order))
    e_rel, e_best = 0.0, 0.0
    snap = {i: final[i] for i in order}
    for t in range(steps if order else 0):
        T = 300.0 * (2.0 / 300.0) ** (t / steps)
        i = order[int(rng.integers(len(order)))]
        g, box = cands[i][int(rng.integers(len(cands[i])))]
        boxes.pop(i, None)
        leaders.pop(i, None)
        old = final[i]
        c_old = cost(i, _gap_of(i, old), old[1])[0]
        c_new, lead, ctr = cost(i, g, box)
        d = c_new - c_old
        if d <= 0 or rng.random() < np.exp(-d / T):
            put(i, (c_new, box, lead, ctr))
            e_rel += d
            if e_rel < e_best - 1e-9:
                e_best = e_rel
                snap = {k: final[k] for k in order}
        else:
            put(i, old)
    for i in order:                                     # best layout seen, then one greedy improvement pass
        put(i, snap[i])
    sweep()
    for i, it in enumerate(items):
        if explain is not None and it.get("key") == explain:
            boxes.pop(i, None); leaders.pop(i, None)
            print(f"  explain {explain}: chosen cost {final[i][0]:.1f}")
            best(i, explain=True)
            boxes[i] = final[i][1]
            if final[i][2]:
                leaders[i] = (final[i][3], tuple(P[i]))
    hard, n_leader = [], 0
    for i, it in enumerate(items):
        c, box, lead, ctr = final[i]
        if c >= HARD / 10:
            hard.append(it["text"])
        kw = dict(textcoords="offset points", xytext=((ctr[0] - P[i, 0]) / pxpt, (ctr[1] - P[i, 1]) / pxpt),
                  ha="center", va="center", fontsize=fontsize, color=it["color"], zorder=6)
        if halo:
            kw["bbox"] = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85)
        if lead:
            n_leader += 1
            kw["arrowprops"] = dict(arrowstyle="-", lw=0.6, color=it["color"], shrinkA=0.5, shrinkB=marker_pt / 2 + 0.5)
        ax.annotate(it["text"], (it["x"], it["y"]), **kw)
        if debug:
            print(f"  label {it['text']:>3s} cost {c:9.1f} leader {lead}")
    return {"n_leader": n_leader, "hard": hard}
