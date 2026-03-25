import sys
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[int]]:
        """
        Find all articulation points in an undirected graph using an iterative
        implementation of Tarjan's algorithm. This avoids the overhead of
        networkx and is fast for large sparse graphs.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - "num_nodes": int, number of nodes.
                - "edges": list of [u, v] edges with 0 <= u < v < num_nodes.

        Returns
        -------
        dict
            A dictionary with a single key "articulation_points" containing a
            sorted list of articulation point node indices.
        """
        n = problem["num_nodes"]
        edges = problem["edges"]

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        ids = [-1] * n           # dfs ids
        low = [0] * n            # low-link values
        visited = [False] * n
        parent = [-1] * n
        ap = set()
        cur_id = 0

        # Use explicit stack for iterative DFS
        stack = []
        for start in range(n):
            if visited[start]:
                continue
            # Start new component
            stack.append((start, iter(adj[start])))
            visited[start] = True
            ids[start] = cur_id
            low[start] = cur_id
            cur_id += 1
            root_child_count = 0

            while stack:
                node, children = stack[-1]
                try:
                    nei = next(children)
                    if not visited[nei]:
                        parent[nei] = node
                        visited[nei] = True
                        ids[nei] = cur_id
                        low[nei] = cur_id
                        cur_id += 1
                        stack.append((nei, iter(adj[nei])))
                        if parent[node] == -1:
                            root_child_count += 1
                    elif nei != parent[node]:
                        # back edge
                        low[node] = min(low[node], ids[nei])
                except StopIteration:
                    stack.pop()
                    if parent[node] == -1:
                        # root node
                        if root_child_count > 1:
                            ap.add(node)
                    else:
                        p = parent[node]
                        low[p] = min(low[p], low[node])
                        if low[node] >= ids[p]:
                            ap.add(p)

        return {"articulation_points": sorted(ap)}
