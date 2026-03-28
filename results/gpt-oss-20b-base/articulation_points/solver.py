from typing import Any, Dict, List

class Solver:
    """
    Find articulation points in an undirected graph using Tarjan's algorithm.
    Complexity: O(V + E)
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        n: int = problem["num_nodes"]
        edges = problem["edges"]

        # Build adjacency list
        adj: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        disc = [-1] * n   # discovery times
        low = [0] * n
        time = 0
        ap = [False] * n

        def dfs(u: int, parent: int) -> None:
            nonlocal time
            disc[u] = low[u] = time
            time += 1
            children = 0

            for v in adj[u]:
                if disc[v] == -1:
                    children += 1
                    dfs(v, u)
                    low[u] = min(low[u], low[v])

                    if parent == -1 and children > 1:
                        ap[u] = True
                    if parent != -1 and low[v] >= disc[u]:
                        ap[u] = True
                elif v != parent:
                    low[u] = min(low[u], disc[v])

        for i in range(n):
            if disc[i] == -1:
                dfs(i, -1)

        result = [i for i, flag in enumerate(ap) if flag]
        return {"articulation_points": result}