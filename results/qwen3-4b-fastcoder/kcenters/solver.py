from typing import Any
import math
from collections.abc import Iterable
import networkx as nx
from pysat.solvers import Solver as SATSolver

class Distances:
    def __init__(self, graph: nx.Graph) -> None:
        self._distances = dict(nx.all_pairs_dijkstra_path_length(graph))

    def all_vertices(self) -> Iterable[str]:
        return self._distances.keys()

    def dist(self, u: str, v: str) -> float:
        return self._distances[u].get(v, math.inf)

    def max_dist(self, centers: Iterable[str]) -> float:
        return max(min(self.dist(c, u) for c in centers) for u in self.all_vertices())

    def vertices_in_range(self, u: str, limit: float) -> Iterable[str]:
        return (v for v, d in self._distances[u].items() if d <= limit)

    def sorted_distances(self) -> list[float]:
        return sorted(dist for dist_dict in self._distances.values() for dist in dist_dict.values())

class Solver:
    def solve(self, problem: tuple[dict[str, dict[str, float]], int]) -> list[str]:
        G_dict, k = problem

        graph = nx.Graph()
        for v, adj in G_dict.items():
            for w, d in adj.items():
                graph.add_edge(v, w, weight=d)

        class KCenterDecisionVariant:
            def __init__(self, distances: Distances, k: int) -> None:
                self.distances = distances
                self._node_vars = {
                    node: i for i, node in enumerate(self.distances.all_vertices(), start=1)
                }
                self._sat_solver = SATSolver("MiniCard")
                self._sat_solver.add_atmost(list(self._node_vars.values()), k=k)
                self._solution = None

            def limit_distance(self, limit: float) -> None:
                for v in self.distances.all_vertices():
                    nodes_in_range = [u for u in self.distances.vertices_in_range(v, limit)]
                    clause = [self._node_vars[u] for u in nodes_in_range]
                    self._sat_solver.add_clause(clause)

            def solve(self) -> list[str] | None:
                if not self._sat_solver.solve():
                    return None
                model = self._sat_solver.get_model()
                if model is None:
                    raise RuntimeError("SAT solver returned no model despite solving successfully.")
                self._solution = [node for node, var in self._node_vars.items() if var in model]
                return self._solution

            def get_solution(self) -> list[str]:
                if self._solution is None:
                    raise ValueError("No solution available. Ensure 'solve' is called first.")
                return self._solution

        class KCentersSolver:
            def __init__(self, graph: nx.Graph) -> None:
                self.graph = graph
                self.distances = Distances(graph)

            def solve_heur(self, k: int) -> list[str]:
                remaining_nodes = set(self.graph.nodes)
                if not remaining_nodes:
                    if k == 0:
                        return []
                    else:
                        raise ValueError(f"Cannot find {k} centers in an empty graph.")
                if k == 0:
                    return []

                max_dist_for_node = {}
                for c in remaining_nodes:
                    max_dist = 0
                    for u in remaining_nodes:
                        d = self.distances.dist(c, u)
                        if d > max_dist:
                            max_dist = d
                    max_dist_for_node[c] = max_dist

                first_center = min(remaining_nodes, key=lambda c: max_dist_for_node[c])
                remaining_nodes.remove(first_center)
                centers = [first_center]
                while len(centers) < k:
                    min_dist_for_node = {}
                    for v in remaining_nodes:
                        min_dist = math.inf
                        for c in centers:
                            d = self.distances.dist(c, v)
                            if d < min_dist:
                                min_dist = d
                        min_dist_for_node[v] = min_dist

                    max_dist_node = max(remaining_nodes, key=lambda v: min_dist_for_node[v])
                    remaining_nodes.remove(max_dist_node)
                    centers.append(max_dist_node)
                return centers

            def solve(self, k: int) -> list[str]:
                centers = self.solve_heur(k)
                obj = self.distances.max_dist(centers)
                decision_variant = KCenterDecisionVariant(self.distances, k)
                distances = self.distances.sorted_distances()
                index = bisect.bisect_left(distances, obj)
                distances = distances[:index]
                if not distances:
                    raise ValueError("No feasible distances less than the current objective.")
                decision_variant.limit_distance(distances[-1])
                while decision_variant.solve() is not None:
                    centers = decision_variant.get_solution()
                    obj = self.distances.max_dist(centers)
                    index = bisect.bisect_left(distances, obj)
                    distances = distances[:index]
                    if not distances:
                        break
                    decision_variant.limit_distance(distances.pop())
                return centers

        solver = KCentersSolver(graph)
        return solver.solve(k)