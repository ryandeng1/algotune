import heapq
import math
from typing import Any, Dict, List, Optional

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[Optional[float]]]]:
        # build adjacency lists from CSR
        data = problem["data"]
        indices = problem["indices"]
        indptr = problem["indptr"]
        n = problem["shape"][0]
        edges: List[List[tuple[int, float]]] = [[] for _ in range(n)]
        for u in range(n):
            start = indptr[u]
            end = indptr[u + 1]
            for idx in range(start, end):
                v = indices[idx]
                w = data[idx]
                edges[u].append((v, w))

        sources: List[int] = problem["source_indices"]
        if not sources:
            return {"distances": []}

        results: List[List[Optional[float]]] = []

        for src in sources:
            dist = [math.inf] * n
            dist[src] = 0.0
            heap = [(0.0, src)]
            while heap:
                d, u = heapq.heappop(heap)
                if d != dist[u]:
                    continue
                for v, w in edges[u]:
                    nd = d + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
            # convert inf to None
            results.append([None if math.isinf(d) else d for d in dist])

        return {"distances": results}
