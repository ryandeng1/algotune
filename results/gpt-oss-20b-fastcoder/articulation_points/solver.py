from typing import Any, List, Dict

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, List[int]]:
        """
        Find all articulation points in an undirected graph using Tarjan's algorithm.
        Returns a sorted list of articulation points.
        """
        n = problem["num_nodes"]
        edges = problem["edges"]

        # Build adjacency list
        adj: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        low = [0] * n
        disc = [0] * n
        parent = [-1] * n
        ap = [False] * n
        time = 1  # Start time from 1 to avoid confusion with default 0

        def dfs(u: int) -> None:
            nonlocal time
            children = 0
            disc[u] = low[u] = time
            time += 1

            for v in adj[u]:
                if disc[v] == 0:          # Tree edge
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # (1) u is root and has two or more children
                    if parent[u] == -1 and children > 1:
                        ap[u] = True
                    # (2) u is non-root and low[v] >= disc[u]
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True
                elif v != parent[u]:      # Back edge
                    low[u] = min(low[u], disc[v])

        for i in range(n):
            if disc[i] == 0:
                dfs(i)

        result = [i for i, is_ap in enumerate(ap) if is_ap]
        result.sort()
        return {"articulation_points": result}