from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        """
        Find articulation points with a linear‑time Tarjan algorithm.
        """
        n = problem["num_nodes"]
        edges = problem["edges"]
        # Build adjacency lists
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        disc = [-1] * n        # discovery time
        low = [0] * n          # low value
        parent = [-1] * n
        ap = [False] * n
        time = 0

        # iterative DFS to avoid recursion limits
        stack = [(0, 0, 0)]  # node, parent, iterator index
        while stack:
            u, p, i = stack.pop()
            if i == 0:          # first time visiting u
                disc[u] = low[u] = time
                time += 1
                parent[u] = p
            # process neighbors
            if i < len(adj[u]):
                v = adj[u][i]
                stack.append((u, p, i + 1))  # resume after child
                if disc[v] == -1:
                    # tree edge
                    stack.append((v, u, 0))
                elif v != parent[u]:
                    # back edge
                    low[u] = low[u] if low[u] < disc[v] else disc[v]
            else:
                # finished all neighbors, propagate low to parent
                if parent[u] != -1:
                    low[parent[u]] = low[parent[u]] if low[parent[u]] < low[u] else low[u]
                # articulation test
                if parent[u] == -1 and sum(1 for _ in adj[u] if parent[_]==u) > 1:
                    ap[u] = True
                elif parent[u] != -1 and any(low[v] >= disc[u] for v in adj[u] if parent[v]==u):
                    ap[u] = True

        ap_list = [i for i, val in enumerate(ap) if val]
        ap_list.sort()
        return {"articulation_points": ap_list}