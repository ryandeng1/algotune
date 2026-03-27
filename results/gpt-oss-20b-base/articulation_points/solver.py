from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Find articulation points in an undirected graph using Tarjan's algorithm
        (O(V + E) time, O(V) memory).  Returns a sorted list of articulation points.
        """
        n = problem["num_nodes"]
        edges = problem["edges"]

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        disc = [-1] * n          # discovery times
        low = [0] * n            # low-link values
        parent = [-1] * n
        ap = [False] * n
        time = 0

        def dfs(u: int):
            nonlocal time
            disc[u] = low[u] = time
            time += 1
            children = 0
            for v in adj[u]:
                if disc[v] == -1:        # Tree edge
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # Articulation point test
                    if parent[u] == -1 and children > 1:
                        ap[u] = True
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True
                elif v != parent[u]:      # Back edge
                    low[u] = min(low[u], disc[v])

        for i in range(n):
            if disc[i] == -1:
                dfs(i)

        result = [i for i, flag in enumerate(ap) if flag]
        result.sort()
        return {"articulation_points": result}