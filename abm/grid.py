"""The uniform grid on the simplex and the order in which uniform discretization visits it."""

from __future__ import annotations

from typing import List

import numpy as np


def simplex_grid(K: int, r: int) -> np.ndarray:
    """All points of Delta_K with coordinates in {0, 1/r, ..., 1}: C(r + K - 1, K - 1) rows, lexicographic order."""
    points: List[List[int]] = []

    def _recurse(remaining: int, depth: int, current: List[int]) -> None:
        if depth == K - 1:
            current.append(remaining)
            points.append(current[:])
            current.pop()
            return
        for v in range(remaining + 1):
            current.append(v)
            _recurse(remaining - v, depth + 1, current)
            current.pop()

    _recurse(int(r), 0, [])
    return np.asarray(points, dtype=float) / r


def _snake_compositions(s: int, m: int, forward: bool = True) -> List[List[int]]:
    """Compositions of s into m parts, ordered so that consecutive ones differ by one unit moved between two
    coordinates (boustrophedon)."""
    if m == 1:
        return [[s]]
    out: List[List[int]] = []
    for a in range(s + 1):
        block = _snake_compositions(s - a, m - 1, forward=(a % 2 == 0))
        out.extend([a] + tail for tail in block)
    return out if forward else out[::-1]


def snake_grid(K: int, r: int) -> np.ndarray:
    """The grid of resolution r in snake order: consecutive nodes are neighbours (l1 distance 2/r).
    For K = 2 this is increasing lambda_1."""
    grid = simplex_grid(K, r)
    order_keys = {tuple(comp): i for i, comp in enumerate(_snake_compositions(int(r), K))}
    counts = np.rint(grid * r).astype(int)
    order = np.argsort([order_keys[tuple(row)] for row in counts])
    return grid[order]


def node_index(i: int, n: int) -> int:
    """Node of visit i (0-based) on a snake list of n nodes: forward on even passes, backward on odd passes."""
    p, j = divmod(int(i), int(n))
    if p % 2 == 1:
        j = n - 1 - j
    return int(j)
