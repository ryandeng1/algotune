import math
import bisect
from typing import Iterable, List, Tuple, Dict

# Lightweight graph representation and all‑pairs shortest distances
class Distances:
    def __init__(self, graph: Dict[str, Dict[str, float]]) -> None:
        """
        graph: adjacency dict {node: {nbr: weight}}
        Computes all‑pairs shortest distances via Floyd‑Warshall.
        """
        nodes = list(graph)
        index = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        # initialize distance matrix
        INF = math.inf
        dist = [[INF] * n for _ in range(n)]
        for u in nodes:
            i = index[u]
            dist[i][i] = 0
            for v, w in graph[u].items():
                j = index[v]
                dist[i][j] = min(dist[i][j], w)

        # Floyd‑Warshall
        for k in range(n):
            dk = dist[k]
            for i in range(n):
                di = dist[i]
                aik = di[k]
                if aik == INF:
                    continue
                for j in range(n):
                    bk = dk[j]
                    if bk == INF:
                        continue
                    new = aik + bk
                    if new < di[j]:
                        di[j] = new

        self._nodes = nodes
        self._index = index
        self._dist = dist

    def all_vertices(self) -> Iterable[str]:
        return self._nodes

    def dist(self, u: str, v: str) -> float:
        i, j = self._index[u], self._index[v]
        return self._dist[i][j]

    def max_dist(self, centers: List[str]) -> float:
        ci = [self._index[c] for c in centers]
        maxd = 0
        for u in self._nodes:
            iu = self._index[u]
            mind = math.inf
            for c in ci:
                d = self._dist[iu][c]
                if d < mind:
                    mind = d
                    if mind == 0:
                        break
            if mind > maxd:
                maxd = mind
        return maxd

    def vertices_in_range(self, u: str, limit: float) -> Iterable[str]:
        iu = self._index[u]
        return (self._nodes[j]
                for j, d in enumerate(self._dist[iu])
                if d <= limit)

    def sorted_distances(self) -> List[float]:
        return sorted(d for row in self._dist for d in row if d < math.inf)

# SAT solver wrapper (requires `pysat`)
try:
    from pysat.solvers import Solver as SATSolver
except Exception:  # pragma: no cover
    SATSolver = None  # SAT path can be omitted if unavailable


class Solver:
    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int]) -> List[str]:
        G_dict, k = problem

        distances = Distances(G_dict)

        # Simple greedy heuristic
        def solve_heur(k: int) -> List[str]:
            if not G_dict:
                return [] if k == 0 else []

            nodes = set(G_dict)
            if k == 0:
                return []

            # First center: min max distance
            first = min(nodes,
                        key=lambda c: max(distances.dist(c, u) for u in nodes))
            centers = [first]
            nodes.remove(first)

            while len(centers) < k:
                best = max(nodes,
                           key=lambda v: min(distances.dist(c, v) for c in centers))
                centers.append(best)
                nodes.remove(best)
            return centers

        # SAT decision variant
        class KCenterDecisionVariant:
            def __init__(self, distances: Distances, k: int):
                self.dist = distances
                self.k = k
                self.node_vars = {
                    v: i + 1 for i, v in enumerate(distances.all_vertices())
                }
                if SATSolver is None:
                    raise RuntimeError("pysat not available")
                self.solver = SATSolver("MiniCard")
                self.solver.add_atmost(list(self.node_vars.values()), k=k)
                self.solver_model = None

            def limit_distance(self, limit: float):
                for v in self.dist.all_vertices():
                    clause = [self.node_vars[u]
                              for u in self.dist.vertices_in_range(v, limit)]
                    if clause:
                        self.solver.add_clause(clause)

            def solve(self) -> List[str] | None:
                if not self.solver.solve():
                    return None
                model = self.solver.get_model()
                chosen = [v for v, var in self.node_vars.items() if var in model]
                return chosen

        # Combine heuristic and SAT refinement
        cent = solve_heur(k)
        obj = distances.max_dist(cent)
        variant = KCenterDecisionVariant(distances, k)

        all_dists = distances.sorted_distances()
        idx = bisect.bisect_left(all_dists, obj)
        cand_dists = all_dists[:idx]
        if not cand_dists:
            return cent
        variant.limit_distance(cand_dists[-1])

        while True:
            sol = variant.solve()
            if sol is None:
                break
            cent = sol
            obj = distances.max_dist(cent)
            idx = bisect.bisect_left(all_dists, obj)
            if idx == 0:
                break
            cand_dists = all_dists[:idx]
            variant.limit_distance(cand_dists[-1])

        return cent