import numpy as np
from numba import njit
import heapq

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        data = np.array(problem["data"], dtype=np.float64)
        indices = np.array(problem["indices"], dtype=np.int64)
        indptr = np.array(problem["indptr"], dtype=np.int64)
        n = problem["shape"][0]
        source_indices = problem["source_indices"]
        
        distances = []
        
        for src in source_indices:
            dist = self._dijkstra_single_source(n, data, indices, indptr, src)
            distances.append(dist)
        
        result = []
        for dist in distances:
            row = [(None if d == np.inf else d) for d in dist]
            result.append(row)
        
        return {"distances": result}
    
    @njit
    def _dijkstra_single_source(self, n, data, indices, indptr, src):
        dist = np.full(n, np.inf, dtype=np.float64)
        dist[src] = 0.0
        
        heap = []
        heapq.heappush(heap, (0.0, src))
        
        while heap:
            d, node = heapq.heappop(heap)
            if d != dist[node]:
                continue
            start_idx = indptr[node]
            end_idx = indptr[node + 1]
            for j in range(start_idx, end_idx):
                neighbor = indices[j]
                weight = data[j]
                new_dist = d + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))
        return dist
