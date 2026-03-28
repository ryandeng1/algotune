# Optimised k‑centers solver
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import math
import networkx as nx
from pysat.solvers import Solver as SATSolver
import bisect

# ---------------------------------------------------------------------------

@dataclass
class Distances:
    """
    Helper that stores all‑pairs shortest path lengths in a flat dictionary
    ``dist[u][v]`` for O(1) look‑ups.  The underlying data is produced by
    networkx once during initialisation, then never recomputed.
    """
    dist: Dict[str, Dict[str, float]]

    @classmethod
    def from_graph(cls, graph: nx.Graph) -> "Distances":
        # networkx's all_pairs_dijkstra_path_length returns an iterator of pairs
        d: Dict[str, Dict[str, float]] = {}
        for u, inner in nx.all_pairs_dijkstra_path_length(graph):
            d[u] = dict(inner)
        return cls(d)

    def all_vertices(self) -> Iterable[str]:
        return self.dist.keys()

    def dist_uv(self, u: str, v: str) -> float:
        return self.dist[u].get(v, math.inf)

    def max_dist(self, centers: Iterable[str]) -> float:
        # maximise over all vertices the distance to the nearest centre
        cset = set(centers)
        maxd = 0.0
        for u in self.dist:
            # min distance to any centre
            mind = min(self.dist_uv(c, u) for c in cset)
            if mind > maxd:
                maxd = mind
        return maxd

    def vertices_in_range(self, u: str, limit: float) -> Iterable[str]:
        return (v for v, d in self.dist[u].items() if d <= limit)

    def sorted_distances(self) -> List[float]:
        all_dists = []
        for inner in self.dist.values():
            all_dists.extend(inner.values())
        return sorted(all_dists)


# ---------------------------------------------------------------------------

class KCenterDecisionVariant:
    """
    SAT‑based decision solver adapted to the pre‑computed distance data.
    Builds a single SAT solver instance and incrementally adds clauses
    corresponding to the distance threshold.
    """
    def __init__(self, distances: Distances, k: int):
        self.distances = distances
        # map node -> SAT variable index (1‑based)
        self.node_vars: Dict[str, int] = {node: i for i, node in enumerate(distances.all_vertices(), start=1)}
        self._sat = SATSolver(name='MiniCard')
        self._sat.add_atmost(list(self.node_vars.values()), k=k)
        self._solution: List[str] | None = None

    def limit_distance(self, limit: float):
        for v in self.distances.all_vertices():
            clause = [self.node_vars[u] for u in self.distances.vertices_in_range(v, limit)]
            self._sat.add_clause(clause)

    def solve(self) -> List[str] | None:
        if not self._sat.solve():
            return None
        model = self._sat.get_model()
        self._solution = [node for node, var in self.node_vars.items() if var in model]
        return self._solution

    def get_solution(self) -> List[str]:
        if self._solution is None:
            raise RuntimeError("No solution after calling solve()")
        return self._solution


# ---------------------------------------------------------------------------

class KCentersSolver:
    """
    Main solver that first obtains a heuristic solution and then
    improves it by solving a sequence of SAT decisions in descending order
    of possible maximum distances.
    """
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.distances = Distances.from_graph(graph)

    def _heuristic(self, k: int) -> List[str]:
        nodes = set(self.graph.nodes)
        if not nodes:
            return [] if k == 0 else []
        if k == 0:
            return []

        # Pick first centre as the node that minimises the farthest distance
        first = min(nodes, key=lambda c: max(self.distances.dist_uv(c, u) for u in nodes))
        centres: List[str] = [first]
        nodes.discard(first)

        while len(centres) < k:
            # choose node that maximises its distance to the nearest centre
            nxt = max(nodes, key=lambda v: min(self.distances.dist_uv(c, v) for c in centres))
            nodes.remove(nxt)
            centres.append(nxt)
        return centres

    def solve(self, k: int) -> List[str]:
        centres = self._heuristic(k)
        obj = self.distances.max_dist(centres)

        # Prepare SAT solver with distance threshold decision list
        decision = KCenterDecisionVariant(self.distances, k)
        dists = self.distances.sorted_distances()
        idx = bisect.bisect_left(dists, obj)
        dists = dists[:idx]  # only values lower than current objective

        if not dists:
            return centres

        decision.limit_distance(dists[-1])
        while True:
            sol = decision.solve()
            if sol is None:
                break
            centres = sol
            obj = self.distances.max_dist(centres)
            idx = bisect.bisect_left(dists, obj)
            dists = dists[:idx]
            if not dists:
                break
            decision.limit_distance(dists.pop())
        return centres


# ---------------------------------------------------------------------------

def solve(problem: Tuple[Dict[str, Dict[str, float]], int]) -> List[str]:
    """Entry point used by the evaluation script."""
    G_dict, k = problem
    graph = nx.Graph()
    for u, adj in G_dict.items():
        for v, w in adj.items():
            graph.add_edge(u, v, weight=w)
    solver = KCentersSolver(graph)
    return solver.solve(k)