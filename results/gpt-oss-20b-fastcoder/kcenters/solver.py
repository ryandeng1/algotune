#!/usr/bin/env python3
"""
solver.py – highly‑optimised K‑center solver

The implementation below focuses on reducing Python overhead by:
1. Converting the graph into index‑based structures (lists) to avoid
   dictionary lookups during distance computations.
2. Pre‑computing all‑pairs shortest‑path lengths only once with
   Floyd–Warshall (fast for medium dense graphs).
3. Replacing the per‑node lambda loops with inlined loops that use
   local variables.
4. Keeping the SAT‑solver logic unchanged to preserve correctness,
   but reducing the number of clauses added by using lists directly.
"""

from __future__ import annotations
import math
import bisect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, List

import networkx as nx
from pysat.solvers import Solver as SATSolver


# --------------------------------------------------------------------------- #
#               Utility – fast all‑pairs shortest paths (Floyd‑Warshall)     #
# --------------------------------------------------------------------------- #
def floyd_warshall(graph: nx.Graph) -> List[List[float]]:
    """
    Compute all‑pairs shortest path lengths using Floyd–Warshall.
    Returns a 2‑D list L where L[i][j] is the distance from node i to j.
    Infinity is represented by math.inf.
    The function assumes that graph is undirected and weighted.
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    INF = math.inf

    # initialise distance matrix
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0
    for u, v, d in graph.edges(data="weight"):
        i, j = idx[u], idx[v]
        w = float(d)
        if w < dist[i][j]:
            dist[i][j] = dist[j][i] = w

    # Floyd‑Warshall
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                continue
            di = dist[i]
            for j in range(n):
                djk = dk[j]
                if djk == INF:
                    continue
                nd = dik + djk
                if nd < di[j]:
                    di[j] = nd
    return dist, nodes, {node: i for i, node in enumerate(nodes)}


# --------------------------------------------------------------------------- #
#                            Distances class (index‑based)                   #
# --------------------------------------------------------------------------- #
class Distances:
    """
    Stores a full distance matrix in a compact list‑of‑lists representation
    for fast access. Provides the same API as the original class.
    """
    __slots__ = ("_dist", "_nodes", "_by_name", "_inv")

    def __init__(self, graph: nx.Graph) -> None:
        self._dist, self._nodes, self._by_name = floyd_warshall(graph)
        self._inv = {v: i for i, v in enumerate(self._nodes)}

    def all_vertices(self) -> Iterable[str]:
        return self._nodes

    def dist(self, u: str, v: str) -> float:
        i, j = self._inv[u], self._inv[v]
        return self._dist[i][j]

    def max_dist(self, centers: Iterable[str]) -> float:
        inv = self._inv
        dmat = self._dist
        max_d = 0.0
        for u in self._nodes:
            min_d = math.inf
            for c in centers:
                dc = dmat[inv[u]][inv[c]]
                if dc < min_d:
                    min_d = dc
            if min_d > max_d:
                max_d = min_d
        return max_d

    def vertices_in_range(self, u: str, limit: float) -> Iterable[str]:
        i = self._inv[u]
        row = self._dist[i]
        return (v for v, d in zip(self._nodes, row) if d <= limit)

    def sorted_distances(self) -> List[float]:
        dmat = self._dist
        return sorted(d for row in dmat for d in row if d != 0)


# --------------------------------------------------------------------------- #
#                   SAT decision variant (unchanged logic)                  #
# --------------------------------------------------------------------------- #
class KCenterDecisionVariant:
    """
    Decision version of the k‑centers problem.
    Uses a SAT solver to decide feasibility for a given radius.
    """
    __slots__ = ("distances", "_node_vars", "_sat_solver", "_solution")

    def __init__(self, distances: Distances, k: int) -> None:
        self.distances = distances
        self._node_vars = {node: i + 1 for i, node in enumerate(distances.all_vertices())}
        self._sat_solver = SATSolver("MiniCard")
        self._sat_solver.add_atmost(list(self._node_vars.values()), k=k)
        self._solution = None

    def limit_distance(self, limit: float) -> None:
        inv = self._node_vars
        for v in self.distances.all_vertices():
            clause = [inv[u] for u in self.distances.vertices_in_range(v, limit)]
            self._sat_solver.add_clause(clause)

    def solve(self) -> List[str] | None:
        if not self._sat_solver.solve():
            return None
        model = self._sat_solver.get_model()
        if model is None:
            raise RuntimeError("SAT solver returned no model despite solving successfully.")
        inv = self._node_vars
        # model contains both positive and negative literals; we only keep positives
        self._solution = [node for node, lit in inv.items() if lit in model]
        return self._solution

    def get_solution(self) -> List[str]:
        if self._solution is None:
            raise ValueError("No solution available. Ensure 'solve' is called first.")
        return self._solution


# --------------------------------------------------------------------------- #
#                           Main solver class                                #
# --------------------------------------------------------------------------- #
class Solver:
    """
    Top level solver exposed to the kata’s environment.
    """
    __slots__ = ("_init_hints")

    def __init__(self) -> None:  # pragma: no cover – nothing to init
        pass

    def solve(self, problem: tuple[dict[str, dict[str, float]], int]) -> List[str]:
        G_dict, k = problem

        # Build a lightweight graph first – avoid adding duplicate edges
        graph = nx.Graph()
        for u, adj in G_dict.items():
            for v, w in adj.items():
                if u == v:
                    continue
                graph.add_edge(u, v, weight=w)

        distances = Distances(graph)

        # --------------------------- Heuristic --------------------------------
        nodes = list(graph.nodes)
        if not nodes:
            return []

        # Pre‑compute distance matrix indices for speed
        inv = distances._inv
        dmat = distances._dist

        # First center – node with minimal maximum distance to all others
        best_c = nodes[0]
        best_val = math.inf
        for c in nodes:
            max_d = max(dmat[inv[c]])
            if max_d < best_val:
                best_val, best_c = max_d, c

        centers = [best_c]
        remaining = set(nodes)
        remaining.remove(best_c)

        # Greedy add remaining centers
        while len(centers) < k:
            # For each candidate, find the closest current center
            best_cand = None
            best_dist = -1.0
            for cand in remaining:
                min_d = math.inf
                ic = inv[cand]
                for c in centers:
                    if dmat[inv[c]][ic] < min_d:
                        min_d = dmat[inv[c]][ic]
                if min_d > best_dist:
                    best_dist, best_cand = min_d, cand
            centers.append(best_cand)
            remaining.remove(best_cand)

        # ------------------------------------------------------
        # Binary search using SAT decisions to tighten the radius
        # ------------------------------------------------------
        all_dist = distances.sorted_distances()
        # Search space: distances that are > 0
        lo = 0
        hi = len(all_dist)

        # Initial objective from greedy
        cur_obj = distances.max_dist(centers)

        # Build decision variant once
        decision = KCenterDecisionVariant(distances, k)

        # While there might be a better smaller radius
        while lo < hi:
            mid_idx = (lo + hi) // 2
            test = all_dist[mid_idx]
            decision._sat_solver.reset()
            decision._sat_solver.add_atmost(list(decision._node_vars.values()), k=k)
            # Add coverage constraints
            for v in distances.all_vertices():
                clause = [decision._node_vars[u] for u in distances.vertices_in_range(v, test)]
                decision._sat_solver.add_clause(clause)
            if decision.solve() is not None:
                # Feasible: try smaller radius
                hi = mid_idx
                centers = decision.get_solution()
                cur_obj = min(cur_obj, distances.max_dist(centers))
            else:
                lo = mid_idx + 1

        return centers


# The module exposes only the Solver class as required.