from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import math
import bisect
import heapq


# ------------------------------ utilities --------------------------------

def dijkstra_all_pairs(G: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute shortest path lengths between all pairs using Dijkstra from every node.
    Graph is assumed weighted and undirected; G[v][w] gives weight."""
    dist_all: Dict[str, Dict[str, float]] = {}
    for src in G.keys():
        dist = {src: 0.0}
        seen = set()
        heap: List[Tuple[float, str]] = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            for w, wgt in G[u].items():
                nd = d + wgt
                if w not in dist or nd < dist[w]:
                    dist[w] = nd
                    heapq.heappush(heap, (nd, w))
        dist_all[src] = dist
    return dist_all


# ------------------------------ distances ---------------------------------

class Distances:
    def __init__(self, graph: Dict[str, Dict[str, float]]) -> None:
        self._distances = dijkstra_all_pairs(graph)

    def all_vertices(self) -> Iterable[str]:
        return self._distances.keys()

    def dist(self, u: str, v: str) -> float:
        return self._distances[u].get(v, math.inf)

    def max_dist(self, centers: Iterable[str]) -> float:
        return max(
            min((self.dist(c, u) for c in centers))
            for u in self._distances.keys()
        )

    def vertices_in_range(self, u: str, limit: float) -> List[str]:
        return [v for v, d in self._distances[u].items() if d <= limit]

    def sorted_distances(self) -> List[float]:
        return sorted(
            d
            for dists in self._distances.values()
            for d in dists.values()
        )


# ------------------------------ SAT solver --------------------------------
# A fast straight‑forward SAT solver using PySAT is employed in the original code. 
# For performance and due to the expected small graph sizes, we replace it with a
# custom backtracking algorithm that handles only the “at most k” and “range” clauses.
# This removes external dependencies and eliminates solver overhead.

class SimpleSatSolver:
    """Very small SAT solver for the specific CNF used in k‑center decision."""
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self.clauses: List[List[int]] = []

    def add_clause(self, clause: List[int]) -> None:
        if clause:
            self.clauses.append(clause)

    def _ok(self, assignment: List[bool]) -> bool:
        if sum(assignment) > self.k:
            return False
        for clause in self.clauses:
            if not any(assignment[v - 1] for v in clause):
                return False
        return True

    def solve(self) -> Tuple[bool, List[bool]]:
        assignment = [False] * self.n

        def backtrack(i: int) -> bool:
            if i == self.n:
                return self._ok(assignment)
            assignment[i] = False
            if backtrack(i + 1):
                return True
            assignment[i] = True
            if backtrack(i + 1):
                return True
            assignment[i] = False
            return False

        ok = backtrack(0)
        return ok, assignment


# ------------------------------ solver -----------------------------------

class Solver:
    def __init__(self):
        pass

    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int]) -> List[str]:
        graph, k = problem
        dist_obj = Distances(graph)

        # ---- heuristic initial solution (Greedy Farther) ----
        nodes = set(graph.keys())
        if not nodes:
            return [] if k == 0 else []
        if k == 0:
            return []

        # pick first center: node with smallest maximum distance to all others
        first_center = min(
            nodes,
            key=lambda c: max(dist_obj.dist(c, u) for u in nodes)
        )
        centers = [first_center]
        nodes.remove(first_center)

        while len(centers) < k:
            # choose node that maximizes distance to nearest center so far
            next_center = max(
                nodes,
                key=lambda v: min(dist_obj.dist(c, v) for c in centers)
            )
            centers.append(next_center)
            nodes.remove(next_center)

        current_obj = dist_obj.max_dist(centers)
        distances_all = dist_obj.sorted_distances()
        # search for a better radius via binary search and SAT check
        lo = 0
        hi = bisect.bisect_left(distances_all, current_obj)

        best_centers = centers
        best_obj = current_obj

        node_list = sorted(dist_obj.all_vertices())
        var_index = {node: i + 1 for i, node in enumerate(node_list)}  # SAT variable index starts at 1

        while lo < hi:
            mid = (lo + hi) // 2
            radius = distances_all[mid]
            sat = SimpleSatSolver(len(node_list), k)
            # add range clauses: each node must be covered by some center within radius
            for u in node_list:
                possible = dist_obj.vertices_in_range(u, radius)
                clause = [var_index[v] for v in possible]
                sat.add_clause(clause)

            sat.solve()  # no need to capture result; if unsat, hi = mid
            ok, model = sat.solve()
            if ok:
                # extract solution
                centers = [node_list[i] for i, val in enumerate(model) if val]
                # refine objective
                obj = dist_obj.max_dist(centers)
                if obj < best_obj:
                    best_obj = obj
                    best_centers = centers
                hi = mid
            else:
                lo = mid + 1

        return best_centers