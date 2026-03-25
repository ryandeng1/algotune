import math
from collections import deque

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, float]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        total_sum = 0.0
        for u in range(n):
            dist = [float('inf')] * n
            dist[u] = 0
            queue = deque([u])
            current_sum = 0.0
            while queue:
                current = queue.popleft()
                for neighbor in adj_list[current]:
                    if dist[neighbor] == float('inf'):
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
                        current_sum += 1.0 / dist[neighbor]
            total_sum += current_sum
        
        efficiency = total_sum / (n * (n - 1))
        return {"global_efficiency": float(efficiency)}
