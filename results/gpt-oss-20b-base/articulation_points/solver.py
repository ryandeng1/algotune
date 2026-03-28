from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        """Find articulation points in an undirected graph using Tarjan's algorithm."""
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

        # iterative DFS stack: (node, iterator, children)
        stack = []

        for root in range(n):
            if disc[root] != -1:
                continue
            stack.append((root, iter(adj[root]), 0))
            disc[root] = low[root] = time
            time += 1

            while stack:
                u, it, child_cnt = stack[-1]
                try:
                    v = next(it)
                except StopIteration:
                    # finished exploring u
                    if parent[u] != -1:
                        p = parent[u]
                        low[p] = min(low[p], low[u])
                        if parent[p] != -1 and low[u] >= disc[p]:
                            ap[p] = True
                        if parent[p] == -1 and child_cnt > 1:
                            ap[p] = True
                    stack.pop()
                    continue

                if disc[v] == -1:             # tree edge
                    parent[v] = u
                    child_cnt += 1
                    stack[-1] = (u, it, child_cnt)
                    disc[v] = low[v] = time
                    time += 1
                    stack.append((v, iter(adj[v]), 0))
                elif v != parent[u]:          # back edge
                    low[u] = min(low[u], disc[v])

        ap_list = [i for i, flag in enumerate(ap) if flag]
        ap_list.sort()
        return {"articulation_points": ap_list}