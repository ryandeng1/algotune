import sys
from collections import defaultdict

class Solver:

    def solve(self, problem: dict) -> dict[str, list[int]]:
        n = problem['num_nodes']
        edges = problem['edges']

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        disc = [-1] * n
        low = [0] * n
        parent = [-1] * n
        ap = [False] * n
        time = 0

        sys.setrecursionlimit(max(1000000, n * 2))

        def dfs(u: int):
            nonlocal time
            children = 0
            disc[u] = low[u] = time
            time += 1
            for v in adj[u]:
                if disc[v] == -1:            # Tree edge
                    parent[v] = u
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    if parent[u] == -1 and children > 1:
                        ap[u] = True
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        for i in range(n):
            if disc[i] == -1:
                dfs(i)

        vertices = [i for i, val in enumerate(ap) if val]
        vertices.sort()
        return {'articulation_points': vertices}