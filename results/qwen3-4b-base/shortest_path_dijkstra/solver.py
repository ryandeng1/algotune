import numpy as np
from numba import njit
from typing import List, Dict, Any

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        n = problem["shape"][0]
        adj = [[] for _ in range(n)]
        
        for i in range(n):
            start_idx = problem["indptr"][i]
            end_idx = problem["indptr"][i + 1]
            for j in range(start_idx, end_idx):
                neighbor = problem["indices"][j]
                weight = problem["data"][j]
                adj[i].append((neighbor, weight))
        
        @njit
        def dijkstra(start):
            dist = np.full(n, np.inf)
            dist[start] = 0.0
            heap = []
            heap.append((0.0, start))
            
            while heap:
                d, node = heap.pop(0)
                if d != dist[node]:
                    continue
                for neighbor, weight in adj[node]:
                    new_dist = d + weight
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        heap.append((new_dist, neighbor))
            return dist
        
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            dist_matrix[i] = dijkstra(i)
        
        dist_matrix_list = [[None if np.isinf(d) else d for d in row] for row in dist_matrix]
        return {"distance_matrix": dist_matrix_list}
