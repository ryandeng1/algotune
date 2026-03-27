from collections.abc import Iterable
import math
import bisect
import networkx as nx
from pysat.solvers import Solver as SATSolver


class Distances:
    __slots__ = ("_distances",)

    def __init__(self, graph: nx.Graph) -> None:
        self._distances = dict(nx.all_pairs_dijkstra_path_length(graph))

    def all_vertices(self) -> Iterable[str]:
        return self._distances.keys()

    def dist(self, u: str, v: str) -> float:
        return self._distances[u].get(v, math.inf)

    def max_dist(self, centers: Iterable[str]) -> float:
        return max(
            min(self.dist(c, u) for c in centers) for u in self.all_vertices()
        )

    def vertices_in_range(self, u: str, limit: float) -> Iterable[str]:
        return (v for v, d in self._distances[u].items() if d <= limit)

    def sorted_distances(self) -> list[float]:
        return sorted(
            d
            for d_dict in self._distances.values()
            for d in d_dict.values()
        )


class Solver:
    def solve(
        self, problem: tuple[dict[str, dict[str, float]], int]
    ) -> list[str]:
        G_dict, k = problem

        # Build the graph
        graph = nx.Graph()
        for v, adj in G_dict.items():
            for w, d in adj.items():
                graph.add_edge(v, w, weight=d)

        distances = Distances(graph)

        class KCenterDecisionVariant:
            def __init__(self, distances: Distances, k: int) -> None:
                self.distances = distances
                self._node_vars = {node: i for i, node in enumerate(distances.all_vertices(), start=1)}
                self._sat_solver = SATSolver("MiniCard")
                self._sat_solver.add_atmost(list(self._node_vars.values()), k=k)
                self._solution = None

            def limit_distance(self, limit: float) -> None:
                for v in self.distances.all_vertices():
                    clause = [
                        self._node_vars[u]
                        for u in self.distances.vertices_in_range(v, limit)
                    ]
                    if clause:
                        self._sat_solver.add_clause(clause)

            def solve(self) -> list[str] | None:
                if not self._sat_solver.solve():
                    return None
                model = self._sat_solver.get_model()
                if model is None:
                    raise RuntimeError("SAT solver returned no model.")
                self._solution = [
                    node for node, var in self._node_vars.items() if var in model
                ]
                return self._solution

            def get_solution(self) -> list[str]:
                if self._solution is None:
                    raise ValueError("No solution available. Ensure 'solve' is called first.")
                return self._solution

        class KCentersSolver:
            def __init__(self, graph: nx.Graph) -> None:
                self.graph = graph
                self.distances = distances

            def solve_heur(self, k: int) -> list[str]:
                if k == 0 or not self.graph.nodes:
                    return []
                remaining = set(self.graph.nodes)
                # First center: minimal maximum distance
                first = min(
                    remaining,
                    key=lambda c: max(self.distances.dist(c, u) for u in remaining),
                )
                remaining.remove(first)
                centers = [first]
                while len(centers) < k:
                    nxt = max(
                        remaining,
                        key=lambda v: min(self.distances.dist(c, v) for c in centers),
                    )
                    remaining.remove(nxt)
                    centers.append(nxt)
                return centers

            def solve(self, k: int) -> list[str]:
                centers = self.solve_heur(k)
                obj = self.distances.max_dist(centers)
                dv = KCenterDecisionVariant(self.distances, k)
                dists = self.distances.sorted_distances()
                idx = bisect.bisect_left(dists, obj)
                cand = dists[:idx]
                if not cand:
                    return centers
                dv.limit_distance(cand[-1])
                while dv.solve() is not None:
                    centers = dv.get_solution()
                    obj = self.distances.max_dist(centers)
                    idx = bisect.bisect_left(dists, obj)
                    if idx == 0:
                        break
                    dv.limit_distance(dists[idx - 1])
                return centers

        return KCentersSolver(graph).solve(k)