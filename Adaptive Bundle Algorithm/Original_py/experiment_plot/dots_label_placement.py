"""Label placement shared by the dots figures (K = 2 Figure 3 / 7c, K = 3
Figure 5): a bare number next to every baseline dot.

Rules (user requests, Sep 9 2026):
* a label sits as close to its dot as the page allows — the first candidate
  offsets are 5 pt from the marker;
* dots that overlap on the page (centres closer than ``overlap_px`` pixels)
  cannot be told apart: the caller either omits all but one of them before
  plotting (report figures: keep one, say so in the caption) or, with
  ``merge=True``, they share ONE label listing every number;
* a label that had to move away from its dot (offset beyond ``leader_pt``
  points) or whose dot has a neighbour within ``crowd_px`` pixels gets a
  thin leader line, so a displaced number can never be read as belonging
  to the neighbouring dot.
"""
from __future__ import annotations
import os
import numpy as np
from matplotlib.text import Text
DEBUG = bool(os.environ.get("DOTS_DEBUG"))

CANDS = [(5, 4, "left", "bottom"), (5, -4, "left", "top"), (-5, 4, "right", "bottom"), (-5, -4, "right", "top"),
         (0, 7, "center", "bottom"), (0, -7, "center", "top"), (8, 0, "left", "center"), (-8, 0, "right", "center"),
         (10, 10, "left", "bottom"), (10, -10, "left", "top"), (-10, 10, "right", "bottom"), (-10, -10, "right", "top"),
         (0, 16, "center", "bottom"), (0, -16, "center", "top"), (18, 6, "left", "bottom"), (-18, 6, "right", "bottom"),
         (18, -6, "left", "top"), (-18, -6, "right", "top"), (0, 26, "center", "bottom"), (0, -26, "center", "top"),
         (26, 16, "left", "bottom"), (-26, 16, "right", "bottom"), (26, -16, "left", "top"), (-26, -16, "right", "top"),
         (0, 38, "center", "bottom"), (0, -38, "center", "top")]


def _clash(b, others):
    return any(not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3]) for o in others)


def _seg_hits_box(p, q, b):
    """does the segment p->q pass through the box b = (x0, y0, x1, y1)?
    (Liang-Barsky clipping; touching the border counts as a hit)"""
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


def _groups(pts, colors, thr):
    """single-linkage groups (same colour) of points closer than thr pixels"""
    n = len(pts); parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if colors[i] == colors[j] and np.hypot(*(pts[i] - pts[j])) < thr:
                parent[find(i)] = find(j)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _fan_out(fig, ax, ren, merged, mpts, obstacles, placed, group, fontsize, bbox, axbb,
             gap_px=32.0, dy_px=30.0, segs_px=(), foreign=None):
    foreign = foreign or {i: list(segs_px) for i in group}
    """Crowded group (user requests Sep 9): members on the rim of the group
    (leftmost / rightmost / lowest / highest) get their number right next to
    the dot with NO leader — below the dot for the lower half of the group,
    above or beside it for the upper half; the remaining, interior members
    are laid out in a row of evenly spaced slots above the group, each with a
    leader line (a member whose leader would cross another dot goes below
    instead).  Returns the boxes it placed (labels drawn), or None."""
    dpi = fig.dpi; px2pt = 72.0 / dpi
    n_all = len(mpts)
    gx = np.array([mpts[i][0] for i in group]); gy = np.array([mpts[i][1] for i in group])
    cx, cy = gx.mean(), gy.mean()
    others_of = {i: [obstacles[j] for j in range(n_all) if j not in merged[i][4]] for i in group}
    drawn, boxes = [], []

    def fail():
        for d in drawn:
            d.remove()
        return None

    # ---- 1. rim members: direct labels
    rim = {group[int(np.argmin(gx))], group[int(np.argmax(gx))], group[int(np.argmin(gy))], group[int(np.argmax(gy))]}
    low = [(0, -7, "center", "top"), (-5, -4, "right", "top"), (5, -4, "left", "top"), (-8, 0, "right", "center"), (8, 0, "left", "center")]
    high = [(0, 7, "center", "bottom"), (5, 4, "left", "bottom"), (8, 0, "left", "center"), (-5, 4, "right", "bottom"), (-8, 0, "right", "center")]
    direct = set()
    for i in sorted(rim, key=lambda i: -abs(mpts[i][0] - cx)):
        cands = low if mpts[i][1] < cy else high
        grown = [(o[0] - 4, o[1] - 4, o[2] + 4, o[3] + 4) for o in others_of[i]]
        for dx, dy, ha, va in cands:
            t = ax.annotate(merged[i][2], (merged[i][0], merged[i][1]), textcoords="offset points", xytext=(dx, dy),
                            ha=ha, va=va, fontsize=fontsize, color=merged[i][3], zorder=6, bbox=bbox)
            t.update_positions(ren); bb = Text.get_window_extent(t, ren)
            box = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
            inside = box[0] >= axbb.x0 - 2 and box[2] <= axbb.x1 + 2 and box[1] >= axbb.y0 - 2 and box[3] <= axbb.y1 + 2
            if inside and not _clash(box, grown) and not _clash(box, placed + boxes) and not _box_hits_segments(box, segs_px):
                drawn.append(t); boxes.append(box); direct.add(i); break
            t.remove()
    interior = [i for i in group if i not in direct]
    if DEBUG:
        print(f"  fan-out group {[merged[i][2] for i in group]}: direct={[merged[i][2] for i in direct]} interior={[merged[i][2] for i in interior]}")
    if not interior:
        return boxes

    # ---- 2. interior members: spaced row with leaders
    widths = {}
    for i in interior:
        t = ax.annotate(merged[i][2], (merged[i][0], merged[i][1]), textcoords="offset points", xytext=(0, 0), fontsize=fontsize)
        bb = t.get_window_extent(ren); widths[i] = bb.x1 - bb.x0; t.remove()
    gap = max(gap_px, max(widths.values()) + 10.0)
    top, bottom = gy.max(), gy.min()

    def row(members, side, dy):
        members = sorted(members, key=lambda i: mpts[i][0])
        m = len(members)
        xs = cx + (np.arange(m) - (m - 1) / 2.0) * gap
        shift = min(0.0, axbb.x1 - 6 - (xs[-1] + widths[members[-1]] / 2)); xs = xs + shift
        shift = max(0.0, axbb.x0 + 6 - (xs[0] - widths[members[0]] / 2)); xs = xs + shift
        y = top + dy if side == "above" else bottom - dy
        out, bad = [], []
        for i, x in zip(members, xs):
            if any(_seg_hits_box((x, y), tuple(mpts[i]), o) for o in others_of[i]) or \
                    _segments_cross((x, y), tuple(mpts[i]), foreign[i]):
                bad.append(i)
            else:
                out.append((i, x, y))
        return out, bad

    for dy in (dy_px, dy_px + 12, dy_px + 24):
        above, bad = row(interior, "above", dy)
        for _ in range(3):
            if not bad:
                break
            below_members = list(bad)
            above, bad2 = row([i for i in interior if i not in below_members], "above", dy)
            if not bad2:
                bad = below_members
                break
            bad = list(set(bad) | set(bad2))
        below, bad_b = row(bad, "below", dy) if bad else ([], [])
        if bad_b:
            continue
        trial, tboxes, ok = [], [], True
        for side, items in (("above", above), ("below", below)):
            for i, x, y in items:
                dx, dyy = (x - mpts[i][0]) * px2pt, (y - mpts[i][1]) * px2pt
                t = ax.annotate(merged[i][2], xy=(merged[i][0], merged[i][1]), xycoords="data",
                                xytext=(dx, dyy), textcoords="offset points", ha="center",
                                va="bottom" if side == "above" else "top", fontsize=fontsize,
                                color=merged[i][3], zorder=6, bbox=bbox,
                                arrowprops=dict(arrowstyle="-", lw=0.6, color=merged[i][3], shrinkA=0, shrinkB=3))
                t.update_positions(ren); bb = Text.get_window_extent(t, ren)
                box = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
                inside = box[0] >= axbb.x0 - 2 and box[2] <= axbb.x1 + 2 and box[1] >= axbb.y0 - 2 and box[3] <= axbb.y1 + 2
                if not inside or _clash(box, others_of[i]) or _clash(box, placed + boxes + tboxes) or _box_hits_segments(box, segs_px):
                    ok = False
                trial.append(t); tboxes.append(box)
        if ok:
            drawn.extend(trial); boxes.extend(tboxes)
            if DEBUG:
                print(f"    row dy={dy:.0f}px above={[merged[i][2] for i,_x,_y in above]} below={[merged[i][2] for i,_x,_y in below]}")
            return boxes
        for t in trial:
            t.remove()
    return fail()


NAMED = {"upper-right": (5, 4, "left", "bottom"), "lower-right": (5, -4, "left", "top"),
         "upper-left": (-5, 4, "right", "bottom"), "lower-left": (-5, -4, "right", "top"),
         "above": (0, 7, "center", "bottom"), "below": (0, -7, "center", "top"),
         "right": (8, 0, "left", "center"), "left": (-8, 0, "right", "center"),
         "far-above": (0, 22, "center", "bottom"), "far-below": (0, -22, "center", "top"),
         "far-left": (-22, 0, "right", "center"), "far-right": (22, 0, "left", "center")}


def _box_hits_segments(box, segs):
    return any(_seg_hits_box(p, q, box) for p, q in segs)


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _foreign_segments(segs, own_pts, tol=3.0):
    """segments that do not start or end at one of the label's own dots"""
    out = []
    for a, b in segs:
        if any(np.hypot(a[0] - x, a[1] - y) < tol or np.hypot(b[0] - x, b[1] - y) < tol for x, y in own_pts):
            continue
        out.append((a, b))
    return out


def _segments_cross(p, q, segs):
    """does the segment p->q cross any of the segments (a, b) in segs?"""
    for a, b in segs:
        d1, d2 = _orient(p, q, a), _orient(p, q, b)
        d3, d4 = _orient(a, b, p), _orient(a, b, q)
        if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0:
            return True
    return False


def place_labels(fig, ax, labs, fontsize=8.5, leader_pt=9.0, overlap_px=10.0, crowd_px=30.0,
                 dot_px=9.0, merge=True, halo=True, fan_min=3, fan_px=40.0, segments=(), prefer=None):
    """labs: list of (x, y, text, colour) in data coordinates.
    segments: connector lines ((x0, y0), (x1, y1)) in data coordinates that
    labels must not cover (user request Sep 9: '120 挡住虚线了').
    prefer: {(colour, text): named position} tried first for that label.
    Returns {"n_leader": int, "merged": [text, ...]}."""
    fig.canvas.draw(); ren = fig.canvas.get_renderer()
    segs_px = [tuple(map(tuple, ax.transData.transform(np.array([p, q], dtype=float)))) for p, q in segments]
    prefer = prefer or {}
    pts = ax.transData.transform(np.array([(x, y) for x, y, _t, _c in labs], dtype=float))
    n = len(labs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    if merge:
        for i in range(n):
            for j in range(i + 1, n):
                if labs[i][3] == labs[j][3] and np.hypot(*(pts[i] - pts[j])) < overlap_px:
                    parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    merged = []
    for members in groups.values():
        members.sort(key=lambda i: float(labs[i][2]) if labs[i][2].replace(".", "").isdigit() else 0.0)
        x = float(np.mean([labs[i][0] for i in members])); y = float(np.mean([labs[i][1] for i in members]))
        merged.append((x, y, ", ".join(labs[i][2] for i in members), labs[members[0]][3], members))
    mpts = ax.transData.transform(np.array([(m[0], m[1]) for m in merged], dtype=float))
    obstacles = [(px - dot_px, py - dot_px, px + dot_px, py + dot_px) for px, py in pts]
    leg = ax.get_legend()
    if leg is not None:
        bb = leg.get_window_extent(ren); obstacles.append((bb.x0, bb.y0, bb.x1, bb.y1))
    axbb = ax.get_window_extent(ren)
    crowded = [any(k not in m[4] and np.hypot(*(mpts[i] - pts[k])) < crowd_px for k in range(n))
               for i, m in enumerate(merged)]
    bbox = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85) if halo else None
    foreign = [_foreign_segments(segs_px, [tuple(pts[j]) for j in m[4]]) for m in merged]
    placed, n_leader = [], 0
    handled = set()
    for g in _groups(mpts, [m[3] for m in merged], fan_px):
        if DEBUG:
            print(f"  group {[merged[i][2] for i in g]}")
        if len(g) < fan_min:
            continue
        boxes = _fan_out(fig, ax, ren, merged, mpts, obstacles, placed, g, fontsize, bbox, axbb, segs_px=segs_px,
                         foreign={i: foreign[i] for i in g})
        if boxes is not None:
            placed.extend(boxes); handled.update(g); n_leader += len(g)
    order = sorted(range(len(merged)), key=lambda i: (-len(merged[i][4]), -int(crowded[i])))
    for i in order:
        if i in handled:
            continue
        x, y, text, color, members = merged[i]
        start = 8 if (crowded[i] or len(members) > 1) else 0
        seq = CANDS[start:] + CANDS[:start]
        if mpts[i][0] > axbb.x1 - 150:            # near the right edge: left-hand spots first
            seq = [c for c in seq if c[2] == "right"] + [c for c in seq if c[2] != "right"]
        pref = prefer.get((color, text))
        if pref in NAMED:
            seq = [NAMED[pref]] + [c for c in seq if c != NAMED[pref]]
        t = ax.annotate(text, (x, y), textcoords="offset points", xytext=seq[0][:2], ha=seq[0][2], va=seq[0][3],
                        fontsize=fontsize, color=color, zorder=6, bbox=bbox)
        chosen = seq[-1]
        own = [obstacles[j] for j in members]          # the dot(s) this label belongs to
        others = [o for j, o in enumerate(obstacles) if j < n and j not in members]
        for c in seq:
            t.set_position(c[:2]); t.set_ha(c[2]); t.set_va(c[3])
            bb = t.get_window_extent(ren); box = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
            inside = box[0] >= axbb.x0 - 2 and box[2] <= axbb.x1 + 2 and box[1] >= axbb.y0 - 2 and box[3] <= axbb.y1 + 2
            if DEBUG and pref:
                print(f"    cand {c[:2]} for {text}: inside={inside} obst={_clash(box, obstacles)} placed={_clash(box, placed)} "
                      f"segs={_box_hits_segments(box, segs_px)}")
            if not (inside and not _clash(box, obstacles) and not _clash(box, placed)) or _box_hits_segments(box, segs_px):
                continue
            # a leader line (drawn from the label box centre to the dot) must not
            # run through another dot (user request Sep 9: K=3 CPU panel)
            centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            if any(_seg_hits_box(centre, tuple(mpts[i]), o) for o in others):
                continue
            needs_leader = crowded[i] or len(members) > 1 or np.hypot(*c[:2]) > leader_pt
            if DEBUG and pref:
                print(f"      leader: dot-hit={any(_seg_hits_box(centre, tuple(mpts[i]), o) for o in others)} "
                      f"cross={needs_leader and _segments_cross(centre, tuple(mpts[i]), foreign[i])}")
            if needs_leader and _segments_cross(centre, tuple(mpts[i]), foreign[i]):
                continue                                   # a leader must not cross the curve or a connector
            chosen = c; break
        placed.append(box)
        if crowded[i] or len(members) > 1 or np.hypot(*chosen[:2]) > leader_pt:
            t.remove(); n_leader += 1
            ax.annotate(text, xy=(x, y), xycoords="data", xytext=chosen[:2], textcoords="offset points",
                        ha=chosen[2], va=chosen[3], fontsize=fontsize, color=color, zorder=6, bbox=bbox,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=color, shrinkA=0, shrinkB=3))
    return {"n_leader": n_leader, "merged": [m[2] for m in merged if len(m[4]) > 1]}
