import numpy as np
import heapq
from typing import Any, Dict, List, Tuple

class Solver:
    def __init__(self):
        # Pre-allocate structures for speed
        self._neighbors: List[List[int]] = []
        self._weights: List[List[float]] = []

    def _build_adjacency(self, indptr: List[int], indices: List[int], data: List[float]) -> None:
        """Convert CSR representation to adjacency list."""
        n = len(indptr) - 1
        self._neighbors = [[] for _ in range(n)]
        self._weights = [[] for _ in range(n)]
        for i in range(n):
            start, end = indptr[i], indptr[i + 1]
            self._neighbors[i] = indices[start:end]
            self._weights[i] = data[start:end]

    def _dijkstra_from_source(self, src: int) -> List[float]:
        """Compute shortest distances from a single source using a binary heap."""
        n = len(self._neighbors)
        dist = [float('inf')] * n
        dist[src] = 0.0
        visited = [False] * n
        heap: List[Tuple[float, int]] = [(0.0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            if d != dist[u]:
                continue  # stale entry
            for v, w in zip(self._neighbors[u], self._weights[u]):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[Any]]]:
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            source_indices = problem["source_indices"]
            if not isinstance(source_indices, list) or not source_indices:
                return {"distances": []}
        except Exception:
            return {"distances": []}

        self._build_adjacency(indptr, indices, data)

        result: List[List[Any]] = []
        for src in source_indices:
            dists = self._dijkstra_from_source(src)
            result.append([(None if np.isinf(dd) else dd) for dd in dists])

        return {"distances": result}
