import numpy as np
from numba import njit

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        data = problem["data"]
        indices = problem["indices"]
        indptr = problem["indptr"]
        shape = problem["shape"]
        n = shape[0]
        
        # Build adjacency list from CSR format
        adj = [[] for _ in range(n)]
        for i in range(n):
            start = indptr[i]
            end = indptr[i+1] if i < n-1 else len(data)
            for j in range(start, end):
                neighbor = indices[j]
                weight = data[j]
                adj[i].append((neighbor, weight))
        
        # Numba-compiled Dijkstra implementation
        @njit
        def dijkstra(start):
            dist = np.full(n, np.inf)
            dist[start] = 0.0
            visited = np.zeros(n, dtype=np.bool_)
            for _ in range(n):
                u = -1
                min_dist = np.inf
                for i in range(n):
                    if not visited[i] and dist[i] < min_dist:
                        min_dist = dist[i]
                        u = i
                if u == -1:
                    break
                visited[u] = True
                for neighbor, weight in adj[u]:
                    if not visited[neighbor]:
                        new_dist = dist[u] + weight
                        if new_dist < dist[neighbor]:
                            dist[neighbor] = new_dist
            return dist
        
        # Compute all pairs shortest paths
        dist_matrix = []
        for i in range(n):
            dist_i = dijkstra(i)
            dist_matrix.append(dist_i.tolist())
        
        # Convert infinity to None
        result = {"distance_matrix": [[None if d == np.inf else d for d in row] for row in dist_matrix]}
        return result
