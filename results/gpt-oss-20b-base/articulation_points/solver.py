# solver.py

from __future__ import annotations

from typing import Any, Dict, List

class Solver:
    """
    Find articulation points (cut vertices) in an undirected graph.
    Implementation is a direct Tarjan DFS; no heavy dependencies.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Parameters
        ----------
        problem : dict
            Must contain 'num_nodes' (int) and 'edges' (list of tuple[int, int]).

        Returns
        -------
        dict
            ``{'articulation_points': sorted(list[int])}``
        """
        n = problem["num_nodes"]
        edges = problem["edges"]

        # Build adjacency list
        adj: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Tarjan's algorithm variables
        disc = [-1] * n            # discovery times
        low = [0] * n              # low values
        parent = [-1] * n
        ap = [False] * n           # articulation point flag
        time = [0]                 # mutable counter

        def dfs(u: int):
            """
            Recursive DFS counting articulation points.
            Uses lists to avoid local stack growth overhead.
            """
            children = 0
            disc[u] = low[u] = time[0]
            time[0] += 1

            for v in adj[u]:
                if disc[v] == -1:          # tree edge
                    parent[v] = u
                    children += 1
                    dfs(v)

                    # Update low[v]
                    if low[v] < low[u]:
                        low[u] = low[v]

                    # Articulation point check
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True

                elif v != parent[u]:       # back edge
                    if disc[v] < low[u]:
                        low[u] = disc[v]

            # Special case for root
            if parent[u] == -1 and children > 1:
                ap[u] = True

        # Run DFS from all disconnected components
        for i in range(n):
            if disc[i] == -1:
                dfs(i)

        # Collect and sort the result
        result = [i for i, is_ap in enumerate(ap) if is_ap]
        result.sort()
        return {"articulation_points": result}