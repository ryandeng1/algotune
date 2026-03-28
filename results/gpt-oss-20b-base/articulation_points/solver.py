from typing import Any, Dict, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, List[int]]:
        """
        Find articulation points in an undirected graph using Tarjan's algorithm.
        """
        n = problem['num_nodes']
        adj = [[] for _ in range(n)]
        for u, v in problem['edges']:
            adj[u].append(v)
            adj[v].append(u)

        discovery = [0] * n
        low = [0] * n
        parent = [-1] * n
        ap = [False] * n
        time = 1

        def dfs(u: int):
            nonlocal time
            discovery[u] = low[u] = time
            time += 1
            children = 0
            for v in adj[u]:
                if not discovery[v]:
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = low[u] if low[u] < low[v] else low[v]
                    if parent[u] == -1 and children > 1:
                        ap[u] = True
                    if parent[u] != -1 and low[v] >= discovery[u]:
                        ap[u] = True
                elif v != parent[u]:
                    low[u] = low[u] if low[u] < discovery[v] else discovery[v]

        for i in range(n):
            if not discovery[i]:
                dfs(i)

        result = [i for i, is_ap in enumerate(ap) if is_ap]
        result.sort()
        return {'articulation_points': result}