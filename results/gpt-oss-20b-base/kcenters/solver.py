# -*- coding: utf-8 -*-
import math
from typing import Iterable, List, Dict, Tuple
import networkx as nx
from ortools.sat.python import cp_model


class Distances:
    __slots__ = ("_distances", "_nodes")

    def __init__(self, graph: nx.Graph) -> None:
        self._distances = dict(nx.all_pairs_dijkstra_path_length(graph))
        self._nodes = list(self._distances.keys())

    def all_vertices(self) -> Iterable[str]:
        return self._nodes

    def dist(self, u: str, v: str) -> float:
        return self._distances[u].get(v, math.inf)

    def sorted_distances(self) -> List[float]:
        return sorted(
            d for d_dict in self._distances.values() for d in d_dict.values()
        )

    def vertices_in_range(self, u: str, limit: float) -> Iterable[str]:
        return (v for v, d in self._distances[u].items() if d <= limit)


class Solver:
    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int]) -> List[str]:
        """
        Efficient 2‑approximation for the k‑center problem using
        a binary search over the sorted list of edge‑weights
        and CP‑SAT feasibility checks for each radius.
        """

        G_dict, k = problem
        graph = nx.Graph()
        for v, adj in G_dict.items():
            for w, d in adj.items():
                graph.add_edge(v, w, weight=d)

        distances = Distances(graph)
        nodes = list(distances.all_vertices())

        # sorted unique distances
        uniq = sorted(set(distances.sorted_distances()))

        # mapping from node to index
        node_index = {node: i for i, node in enumerate(nodes)}

        def feasible(r: float) -> Tuple[bool, List[str]]:
            """Check if there exists a set of k centers within radius r."""
            model = cp_model.CpModel()
            x = [model.NewBoolVar(f"x{i}") for i in range(len(nodes))]

            # at most k centers
            model.Add(sum(x) <= k)

            # coverage constraints
            for v in nodes:
                coverers = [x[node_index[u]]
                            for u, d in distances._distances[v].items() if d <= r]
                if not coverers:
                    # no node can cover v under this radius
                    return False, []
                model.Add(sum(coverers) >= 1)

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 5.0
            status = solver.Solve(model)
            if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
                return False, []

            selected = [nodes[i] for i, var in enumerate(x) if solver.Value(var) == 1]
            return True, selected

        # binary search for minimal feasible radius
        lo, hi = 0, len(uniq) - 1
        best_radius = uniq[hi]
        best_solution: List[str] = []

        while lo <= hi:
            mid = (lo + hi) // 2
            r = uniq[mid]
            ok, sol = feasible(r)
            if ok:
                best_radius, best_solution = r, sol
                hi = mid - 1
            else:
                lo = mid + 1

        return best_solution