from typing import Iterable, Tuple, List, Dict
import math
import bisect
import networkx as nx
from pysat.solvers import Solver as SATSolver


class Distances:
    def __init__(self, graph: nx.Graph) -> None:
        # Fast all‑pairs shortest paths with Dijkstra
        self._distances = dict(nx.all_pairs_dijkstra_path_length(graph))

    def all_vertices(self) -> Iterable[str]:
        return self._distances.keys()

    def dist(self, u: str, v: str) -> float:
        return self._distances[u].get(v, math.inf)

    def max_dist(self, centers: Iterable[str]) -> float:
        return max(
            (
                min((self.dist(c, u) for c in centers))
                for u in self.all_vertices()
            )
        )

    def vertices_in_range(self, u: str, limit: float) -> Iterable[str]:
        return (v for v, d in self._distances[u].items() if d <= limit)

    def sorted_distances(self) -> List[float]:
        return sorted((d for d_dict in self._distances.values() for d in d_dict.values()))


class Solver:
    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int]) -> List[str]:
        G_dict, k = problem
        graph = nx.Graph()
        for u, neigh in G_dict.items():
            for v, w in neigh.items():
                graph.add_edge(u, v, weight=w)

        class DecisionVariant:
            def __init__(self, distances: Distances, k: int) -> None:
                self._node_vars = {node: i for i, node in enumerate(distances.all_vertices(), start=1)}
                self._sat = SATSolver('MiniCard')
                self._sat.add_atmost(list(self._node_vars.values()), k=k)
                self._distances = distances

            def limit_distance(self, limit: float) -> None:
                for v in self._distances.all_vertices():
                    clause = [self._node_vars[u] for u in self._distances.vertices_in_range(v, limit)]
                    self._sat.add_clause(clause)

            def solve(self) -> List[str] | None:
                if not self._sat.solve():
                    return None
                model = self._sat.get_model()
                return [node for node, var in self._node_vars.items() if var in model]

        class KCentersSolver:
            def __init__(self, graph: nx.Graph) -> None:
                self.graph = graph
                self.distances = Distances(graph)

            def heuristic(self, k: int) -> List[str]:
                remaining = set(self.graph.nodes)
                if k == 0:
                    return []
                # first greedy center
                first = min(remaining, key=lambda c: max(self.distances.dist(c, u) for u in remaining))
                remaining.remove(first)
                centers = [first]
                while len(centers) < k and remaining:
                    # furthest from current centers
                    v = max(remaining, key=lambda x: min(self.distances.dist(c, x) for c in centers))
                    remaining.remove(v)
                    centers.append(v)
                return centers

            def solve(self, k: int) -> List[str]:
                centers = self.heuristic(k)
                obj = self.distances.max_dist(centers)
                decisions = DecisionVariant(self.distances, k)
                dists = self.distances.sorted_distances()
                idx = bisect.bisect_left(dists, obj)
                dists = dists[:idx]
                if not dists:
                    return centers
                decisions.limit_distance(dists[-1])
                while True:
                    sol = decisions.solve()
                    if sol is None:
                        break
                    centers = sol
                    obj = self.distances.max_dist(centers)
                    idx = bisect.bisect_left(dists, obj)
                    dists = dists[:idx]
                    if not dists:
                        break
                    decisions.limit_distance(dists.pop())
                return centers

        return KCentersSolver(graph).solve(k)