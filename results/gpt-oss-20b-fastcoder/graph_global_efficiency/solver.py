from collections import deque
from typing import List, Dict

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, float]:
        """
        Calculates the global efficiency of the graph without external
        libraries. The efficiency is computed as the average of the
        reciprocal shortest path lengths between all unordered pairs of
        distinct nodes.

        Args:
            problem: A dictionary containing the adjacency list of the
                     graph as {"adjacency_list": adj_list}
                     where adj_list is a list of lists of integers.

        Returns:
            A dictionary containing the global efficiency as
            {"global_efficiency": efficiency_value}.
        """
        adj = problem["adjacency_list"]
        n = len(adj)
        if n <= 1:
            return {"global_efficiency": 0.0}

        # Ensure the adjacency list is symmetric and contains unique neighbors
        for i, nb in enumerate(adj):
            adj[i] = set(nb)

        total = 0.0
        pair_count = n * (n - 1)

        for src in range(n):
            # BFS to compute shortest distances from src
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)

            # Accumulate reciprocals of distances for pairs (src, j)
            for j in range(src + 1, n):
                d = dist[j]
                if d > 0:
                    total += 1.0 / d
                # If d == 0 (disconnected), contribute 0

        global_efficiency = total * 2.0 / pair_count
        return {"global_efficiency": global_efficiency}