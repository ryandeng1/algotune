import math
import bisect
from typing import Iterable
from pysat.solvers import Solver as SATSolver
import networkx as nx


class Distances:
    def __init__(self, graph: nx.Graph) -> None:
        # Pre‑compute the all‑pairs distances once.
        self._distances = dict(nx.all_pairs_dijkstra_path_length(graph))

    def all_vertices(self) -> Iterable[str]:
        return self._distances.keys()

    def dist(self, u: str, v: str) -> float:
        return self._distances[u].get(v, math.inf)

    def max_dist(self, centers: Iterable[str]) -> float:
        maxd = 0.0
        for u in self.all_vertices():
            mind = math.inf
            for c in centers:
                d = self.dist(c, u)
                if d < mind:
                    mind = d
            if mind > maxd:
                maxd = mind
        return maxd

    def vertices_in_range(self, u: str, limit: float) -> Iterable[str]:
        return (v for v, d in self._distances[u].items() if d <= limit)

    def sorted_distances(self) -> list[float]:
        return sorted(
            d for dist_dict in self._distances.values() for d in dist_dict.values()
        )


class Solver:
    def solve(self, problem: tuple[dict[str, dict[str, float]], int]) -> list[str]:
        G_dict, k = problem
        graph = nx.Graph()
        for src, neigh in G_dict.items():
            for dst, w in neigh.items():
                graph.add_edge(src, dst, weight=w)

        class KCenterDecisionVariant:
            def __init__(self, distances: Distances, k: int) -> None:
                self.distances = distances
                self._node_vars = {node: i for i, node in enumerate(self.distances.all_vertices(), start=1)}
                self._sat_solver = SATSolver("MiniCard")
                self._sat_solver.add_atmost(list(self._node_vars.values()), k=k)
                self._solution = None

            def limit_distance(self, limit: float) -> None:
                for v in self.distances.all_vertices():
                    clause = [self._node_vars[u] for u in self.distances.vertices_in_range(v, limit)]
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
                    raise ValueError("No solution available. Call `solve` first.")
                return self._solution

        class KCentersSolver:
            def __init__(self, graph: nx.Graph) -> None:
                self.graph = graph
                self.distances = Distances(graph)

            def solve_heur(self, k: int) -> list[str]:
                nodes = set(self.graph.nodes)
                if not nodes:
                    return []
                if k == 0:
                    return []

                def farthest_from(node_set):
                    best = None
                    bestd = -1
                    for n in node_set:
                        dmax = 0.0
                        for m in node_set:
                            d = self.distances.dist(n, m)
                            if d > dmax:
                                dmax = d
                        if dmax > bestd:
                            bestd = dmax
                            best = n
                    return best

                center = farthest_from(nodes)
                nodes.remove(center)
                centers = [center]
                while len(centers) < k:
                    # choose node that is farthest from its nearest center
                    best = None
                    bestdist = -1
                    for n in nodes:
                        mind = min(self.distances.dist(c, n) for c in centers)
                        if mind > bestdist:
                            bestdist = mind
                            best = n
                    nodes.remove(best)
                    centers.append(best)
                return centers

            def solve(self, k: int) -> list[str]:
                centers = self.solve_heur(k)
                best_obj = self.distances.max_dist(centers)

                dec_var = KCenterDecisionVariant(self.distances, k)
                all_dists = self.distances.sorted_distances()

                # binary search initial bound
                i = bisect.bisect_left(all_dists, best_obj)
                dist_slice = all_dists[:i]
                if not dist_slice:
                    return centers
                dec_var.limit_distance(dist_slice[-1])

                while True:
                    sol = dec_var.solve()
                    if sol is None:
                        break
                    centers = sol
                    best_obj = self.distances.max_dist(centers)
                    i = bisect.bisect_left(all_dists, best_obj)
                    dist_slice = all_dists[:i]
                    if not dist_slice:
                        break
                    dec_var.limit_distance(dist_slice.pop())
                return centers

        solver = KCentersSolver(graph)
        return solver.solve(k)