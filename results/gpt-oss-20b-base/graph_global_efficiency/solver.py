#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast solver for computing the global efficiency of an undirected graph.
The original implementation used NetworkX and was sub‑optimal for large
graphs. This version implements a straightforward BFS based algorithm
with optional Numba acceleration for maximal performance.
"""

from __future__ import annotations

import sys
import collections
import math
from typing import Dict, Iterable, List, Tuple

try:
    from numba import jit
except Exception:  # pragma: no cover
    # Numba might not be installed – we fall back to a pure Python implementation
    def jit(func=None, **kwargs):
        return func

__all__ = ["Solver"]


@jit(nopython=True, parallel=True)
def _bfs_start(start: int, n: int, neighbors: List[List[int]]) -> Tuple[int, float]:
    """
    Single source BFS that computes the sum of inverse distances from
    ``start`` to every other reachable node.

    Returns
    ------
    tuple
        (number of reachable nodes, sum of reciprocal distances)
    """
    dist = -1 * np.ones(n, dtype=np.int32)
    dist[start] = 0
    queue = np.empty(n, dtype=np.int32)
    qhead = 0
    qtail = 0
    queue[qtail] = start
    qtail += 1

    reach = 0
    invsum = 0.0

    while qhead < qtail:
        u = queue[qhead]
        qhead += 1
        du = dist[u]
        for v in neighbors[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                queue[qtail] = v
                qtail += 1
                # accumulate reciprocal of distance (excluding self)
                invsum += 1.0 / (du + 1)
                reach += 1
    return reach, invsum


class Solver:
    """
    Compute the global efficiency of an undirected unweighted graph.
    ``global_efficiency`` is defined as the average of the inverse of the
    shortest path length between all pairs of distinct vertices for which
    a path exists.

    The input format is a dictionary with a single key ``"adjacency_list"``,
    containing a list of lists of integer neighbors for each vertex.
    """

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, float]:
        """
        Parameters
        ----------
        problem : dict
            ``{"adjacency_list": adj_list}``
            where ``adj_list[i]`` is a list of vertices adjacent to vertex ``i``.

        Returns
        -------
        dict
            ``{"global_efficiency": efficiency_value}``
        """
        try:
            adj_list: List[List[int]] = problem["adjacency_list"]
        except (KeyError, TypeError):
            return {"global_efficiency": 0.0}

        n = len(adj_list)
        if n < 2:
            return {"global_efficiency": 0.0}

        # Build adjacency list of neighbors for each vertex.
        # Replace duplicates & self-loops and ensure symmetry.
        neighbors = [set() for _ in range(n)]
        for u, neighs in enumerate(adj_list):
            for v in neighs:
                if v == u or v < 0 or v >= n:
                    continue
                neighbors[u].add(v)
                neighbors[v].add(u)
        # Convert to list for fast iteration
        neighbors = [list(s) for s in neighbors]

        # Choose the best implementation available
        eff = self._compute_efficiency(neighbors, n)

        return {"global_efficiency": float(eff)}

    @staticmethod
    def _compute_efficiency(neighbors: List[List[int]], n: int) -> float:
        """
        Dispatch to either a Numba accelerated or pure Python implementation
        depending on availability.
        """
        if sys.modules.get("numba"):
            # Numba accelerated BFS – this is the fastest path for large graphs
            return Solver._numba_efficiency(neighbors, n)
        else:
            return Solver._python_efficiency(neighbors, n)

    @staticmethod
    def _numba_efficiency(neighbors: List[List[int]], n: int) -> float:
        """
        Compute global efficiency using Numba accelerated BFS.
        """
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            return Solver._python_efficiency(neighbors, n)

        # Convert adjacency lists to a flat array to allow Numba array access
        # This is necessary because Numba JIT cannot automatically handle nested lists
        max_deg = max(len(nb) for nb in neighbors)
        neighbors_np = np.full((n, max_deg), -1, dtype=np.int32)
        for i, nb in enumerate(neighbors):
            neighbors_np[i, :len(nb)] = nb

        invsum_total = 0.0
        for src in range(n):
            reach, invsum = _bfs_start(src, n, neighbors_np)
            invsum_total += invsum

        # Normalise by the number of ordered pairs (u != v) that are connected.
        # If a pair has no connecting path, its contribution is zero (already omitted).
        return invsum_total / (n * (n - 1))

    @staticmethod
    def _python_efficiency(neighbors: List[List[int]], n: int) -> float:
        """
        Pure Python BFS – used when Numba is unavailable.
        """
        invsum_total = 0.0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q: Iterable[int] = collections.deque([src])

            while q:
                u = q.popleft()
                du = dist[u]
                for v in neighbors[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)
                        invsum_total += 1.0 / (du + 1)

        return invsum_total / (n * (n - 1))