import heapq
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        data = problem["data"]
        indices = problem["indices"]
        indptr = problem["indptr"]
        n = problem["shape"][0]
        source_indices = problem["source_indices"]
        
        neighbor_lists = []
        for i in range(n):
            start = indptr[i]
            end = indptr[i + 1]
            neighbors = []
            for j in range(start, end):
                neighbors.append((indices[j], data[j]))
            neighbor_lists.append(neighbors)
        
        distances = []
        
        for source in source_indices:
            dist = [float('inf')] * n
            dist[source] = 0.0
            heap = [(0.0, source)]
            
            while heap:
                d, u = heapq.heappop(heap)
                if d != dist[u]:
                    continue
                for v, weight in neighbor_lists[u]:
                    new_dist = d + weight
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        heapq.heappush(heap, (new_dist, v))
            
            distances.append([None if x == float('inf') else x for x in dist])
        
        return {"distances": distances}
