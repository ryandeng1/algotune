# solver.py
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        """
        Find all articulation points in an undirected graph using
        Tarjan's algorithm (DFS with discovery times and low links).
        The implementation is pure Python and avoids external libraries
        for maximum speed and minimal overhead.
        """
        n = problem["num_nodes"]
        edges = problem["edges"]

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        sys.setrecursionlimit(max(1000000, n * 2 + 10))

        disc = [-1] * n  # discovery times
        low = [0] * n    # low-link values
        parent = [-1] * n
        ap = [False] * n
        time = 0

        def dfs(u: int):
            nonlocal time
            disc[u] = low[u] = time
            time += 1
            children = 0
            for v in adj[u]:
                if disc[v] == -1:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = low[u] if low[u] < low[v] else low[v]
                    if parent[u] == -1 and children > 1:
                        ap[u] = True
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True
                elif v != parent[u]:
                    if low[u] > disc[v]:
                        low[u] = disc[v]

        for i in range(n):
            if disc[i] == -1:
                dfs(i)

        result = [i for i, is_ap in enumerate(ap) if is_ap]
        result.sort()
        return {"articulation_points": result}
