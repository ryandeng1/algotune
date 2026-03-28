from typing import Any, Iterable, Tuple, List, Dict
import math
import bisect
import networkx as nx
from pysat.solvers import Solver as SATSolver

class Distances:
    """Pre‑compute all‑pairs shortest path lengths and provide convenient queries."""

    def __init__(self, graph: nx.Graph) -> None:
        # Use Dijkstra from each node; dict of dict
        self._dist = dict(nx.all_pairs_dijkstra_path_length(graph))

    def all_vertices(self) -> Iterable[str]:
        return self._dist.keys()

    def dist(self, u: str, v: str) -> float:
        return self._dist[u].get(v, math.inf)

    def max_dist(self, centers: Iterable[str]) -> float:
        # For each vertex take its nearest center
        maxdist = 0.0
        for v in self._dist:
            best = min(self.dist(c, v) for c in centers)
            if best > maxdist:
                maxdist = best
        return maxdist

    def vertices_in_range(self, u: str, limit: float) -> Iterable[str]:
        return (v for v, d in self._dist[u].items() if d <= limit)

    def sorted_distances(self) -> List[float]:
        return sorted(d for d in (dist for dd in self._dist.values() for dist in dd.values()))


class Solver:

    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int]) -> List[str]:
        G_dict, k = problem
        graph = nx.Graph()
        for v, neigh in G_dict.items():
            for w, d in neigh.items():
                graph.add_edge(v, w, weight=d)

        distances = Distances(graph)

        # --- greedy baseline --------------------------------------------------
        def greedy(k: int) -> List[str]:
            if k == 0:
                return []
            nodes = set(graph.nodes)
            # choose first center that minimises the furthest distance
            first = min(nodes, key=lambda c: max(distances.dist(c, u) for u in nodes))
            nodes.remove(first)
            centers = [first]
            while len(centers) < k:
                # farthest point from current centers
                furthest = max(nodes, key=lambda v: min(distances.dist(c, v) for c in centers))
                nodes.remove(furthest)
                centers.append(furthest)
            return centers

        # --- SAT refinement ----------------------------------------------------
        class KCenterDecisionVariant:

            def __init__(self, distances: Distances, k: int):
                self.distances = distances
                # node vars are 1‑based for pysat
                self._node_vars = {n: i + 1 for i, n in enumerate(distances.all_vertices())}
                self._sat = SATSolver('MiniCard')
                self._sat.add_atmost(list(self._node_vars.values()), k)
                self._solution = None

            def limit_distance(self, limit: float):
                for v in self.distances.all_vertices():
                    clause = [self._node_vars[u] for u in self.distances.vertices_in_range(v, limit)]
                    self._sat.add_clause(clause)

            def solve(self) -> List[str] | None:
                if not self._sat.solve():
                    return None
                model = set(self._sat.get_model())
                self._solution = [node for node, var in self._node_vars.items() if var in model]
                return self._solution

            def get_solution(self) -> List[str]:
                assert self._solution is not None
                return self._solution

        # -------------------------------------------------------
        # binary search over sorted unique distances using SAT
        # -------------------------------------------------------
        sorted_dists = distances.sorted_distances()
        lo, hi = 0, len(sorted_dists) - 1
        best_centers = greedy(k)

        while lo <= hi:
            mid = (lo + hi) // 2
            limit = sorted_dists[mid]
            variant = KCenterDecisionVariant(distances, k)
            variant.limit_distance(limit)
            sol = variant.solve()
            if sol is not None:
                best_centers = sol
                hi = mid - 1
            else:
                lo = mid + 1

        return best_centers