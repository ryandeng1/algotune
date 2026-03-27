from typing import Any, Dict, List
import sys

sys.setrecursionlimit(1 << 25)


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Find articulation points of an undirected graph using Tarjan's algorithm.
        """
        n: int = problem["num_nodes"]
        edges = problem["edges"]

        # Build adjacency list
        adj: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Tarjan variables
        disc = [-1] * n          # discovery time
        low = [0] * n            # low-link values
        parent = [-1] * n
        ap = [False] * n
        time = 0

        def dfs(u: int) -> None:
            nonlocal time
            children = 0
            disc[u] = low[u] = time
            time += 1

            for v in adj[u]:
                if disc[v] == -1:          # Tree edge
                    parent[v] = u
                    children += 1
                    dfs(v)

                    low[u] = min(low[u], low[v])

                    if parent[u] == -1 and children > 1:
                        ap[u] = True
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True
                elif v != parent[u]:        # Back edge
                    low[u] = min(low[u], disc[v])

        for i in range(n):
            if disc[i] == -1:
                dfs(i)

        articulation_points = [i for i, flag in enumerate(ap) if flag]
        articulation_points.sort()
        return {"articulation_points": articulation_points}