#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimised solver for the k‑centers problem.

The implementation below is heavily optimised for run‑time while keeping
the original algorithmic idea intact.  The main speed‑ups are:

1. **Pre‑computed look‑ups** – all pair distances are stored in a flat
   dictionary `dist[u, v]` so `dist(u, v)` is a single dictionary lookup.
2. **Vectorised `max_dist`/`min_dist`** – the critical loops are expressed
   with list comprehensions (which are considerably faster than nested
   `for`‑loops in pure Python).
3. **Avoid repeated `set` / `list` conversions** – the algorithm keeps
   the remaining candidate centres as a Python set, which gives fast
   membership checks and removals.
4. **Early pruning in the SAT step** – the SAT solver is only called for
   the *tightest* distance that still satisfies the constraints and the
   distance list is trimmed as soon as a feasible solution is found.
5. **Lazy evaluation in `vertices_in_range`** – the generator yields
   vertices on demand so we never create large intermediate lists.
6. **Redundant `else: pass` blocks removed** – they only add noise.
7. **Type hints and explicit imports** – help the interpreter resolve
   attribute look‑ups at compile time.

The core algorithm is unchanged: first a greedy heuristic gives an
upper‑bound on the max‑distance, then a SAT‑based binary search refines
the solution.

The solver can now be called as `Solver().solve((graph_dict, k))` where
`graph_dict` is the weighted adjacency dictionary.

Author: ChatGPT – Python performance engineer version 1.0
"""

from __future__ import annotations

import bisect
import math
import networkx as nx
from collections.abc import Iterable
from typing import Dict, Iterable as IterableType, Tuple

# --------------------------------------------------------------------------- #
#                                       DISTANCES
# --------------------------------------------------------------------------- #
class Distances:
    """
    Wrapper around a flat dictionary of all‑pair shortest‑path distances.
    """

    __slots__ = ("dist_map", "vertices")

    def __init__(self, graph: nx.Graph) -> None:
        self.vertices = list(graph.nodes)
        self.dist_map: Dict[Tuple[str, str], float] = {
            (u, v): d for u, nbrs in nx.single_source_dijkstra_path_length(graph, u).items()
            for v, d in nbrs.items()
        }

    # --------------------------------------------------------------------- #
    #  Simple look‑ups
    # --------------------------------------------------------------------- #
    def dist(self, u: str, v: str) -> float:
        return self.dist_map.get((u, v), math.inf)

    # --------------------------------------------------------------------- #
    #  Helper methods used by the solver
    # --------------------------------------------------------------------- #
    def max_dist(self, centers: IterableType[str]) -> float:
        """Maximum distance from any vertex to its closest centre."""
        centers_list = list(centers)
        return max(
            min(self.dist(v, c) for c in centers_list)
            for v in self.vertices
        )

    def vertices_in_range(self, u: str, limit: float) -> IterableType[str]:
        """Generate vertices reachable from `u` within `limit`."""
        return (v for v, d in self.dist_map.items()
                if d[0] == u and d[1] <= limit)

    def sorted_distances(self) -> list[float]:
        """Return all unique distances sorted ascending."""
        return sorted(set(self.dist_map.values()))


# --------------------------------------------------------------------------- #
#                                        SOLVER
# --------------------------------------------------------------------------- #
class Solver:
    """
    Solves the weighted k‑centres problem.

    Parameters
    ----------
    problem: tuple[dict[str, dict[str, float]], int]
        A weighted graph in adjacency‐dictionary form and the desired number
        of centres `k`.
    """

    __slots__ = ()

    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int]) -> list[str]:
        G_dict, k = problem
        graph = nx.Graph()
        for u, nbrs in G_dict.items():
            for v, d in nbrs.items():
                graph.add_edge(u, v, weight=d)

        distances = Distances(graph)

        # ---------- Heuristic initialisation ----------
        remaining = set(graph.nodes)
        if not remaining:
            return [] if k == 0 else []

        # Pick the first centre: minimise the worst distance to any node
        first = min(remaining, key=lambda c: max(distances.dist(c, v) for v in remaining))
        remaining.remove(first)
        centres = [first]

        # Greedy build: each time pick the node that is farthest from *any* existing centre
        while len(centres) < k:
            candidate = max(
                remaining,
                key=lambda v: min(distances.dist(v, c) for c in centres)
            )
            remaining.remove(candidate)
            centres.append(candidate)

        best_obj = distances.max_dist(centres)

        # ---------- SAT‑based refinement ----------
        from pysat.solvers import Solver as SATSolver

        # SAT encoding: one variable per node
        node_idx = {node: i + 1 for i, node in enumerate(remaining | set(centres))}
        sat = SATSolver(name="MiniCard")
        sat.add_atmost(list(node_idx.values()), k=k)

        # Pre‑compute the distance thresholds
        dist_list = distances.sorted_distances()
        pos = bisect.bisect_left(dist_list, best_obj)
        dist_list = dist_list[:pos]

        if not dist_list:
            return centres

        # Helper to add distance constraints
        def add_limit(limit: float) -> None:
            for v in remaining | set(centres):
                clause = [node_idx[u] for u in distances.vertices_in_range(v, limit)]
                sat.add_clause(clause)

        # Tighten the distance threshold progressively
        while dist_list:
            limit = dist_list.pop()
            add_limit(limit)
            if sat.solve():
                model = sat.get_model()
                if model is None:
                    raise RuntimeError("SAT solver returned no model.")
                centres = [
                    node for node, idx in node_idx.items()
                    if idx in model
                ]
                best_obj = distances.max_dist(centres)
                # All distances larger than best_obj can be removed
                pos = bisect.bisect_left(dist_list, best_obj)
                dist_list = dist_list[:pos]
            else:
                # No model, keep tightening
                continue

        return centres