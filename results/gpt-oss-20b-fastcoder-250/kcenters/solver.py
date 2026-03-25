import itertools
import heapq
from typing import Any, Dict, Iterable, Set, Tuple

class Solver:
    def _all_pairs_shortest(self, graph: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute shortest path distances using Dijkstra for each source node."""
        dist = {}
        for source in graph:
            d = {source: 0.0}
            visited = set()
            heap = [(0.0, source)]
            while heap:
                du, u = heapq.heappop(heap)
                if u in visited:
                    continue
                visited.add(u)
                for v, w in graph[u].items():
                    if v not in d or du + w < d[v]:
                        d[v] = du + w
                        heapq.heappush(heap, (d[v], v))
            dist[source] = d
        return dist

    def _max_distance(self, centers: Iterable[str], all_pairs: Dict[str, Dict[str, float]]) -> float:
        """Maximum distance from any node to its nearest center."""
        best = 0.0
        nodes = all_pairs.keys()
        for node in nodes:
            min_d = min(all_pairs[node][c] for c in centers)
            if min_d > best:
                best = min_d
        return best

    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int]) -> Set[str]:
        graph, k = problem
        # Handle trivial cases
        nodes = list(graph.keys())
        n = len(nodes)
        if k <= 0 or n == 0:
            return set()

        if k >= n:
            return set(nodes)

        all_pairs = self._all_pairs_shortest(graph)

        best_obj = float("inf")
        best_set: Set[str] = set()

        # iterate all combinations of size k
        for comb in itertools.combinations(nodes, k):
            obj = self._max_distance(comb, all_pairs)
            if obj < best_obj:
                best_obj = obj
                best_set = set(comb)
                # early exit if zero distance achieved
                if best_obj == 0.0:
                    break

        return best_set
