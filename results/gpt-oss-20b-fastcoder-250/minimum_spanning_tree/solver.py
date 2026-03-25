from __future__ import annotations
from typing import Any, List

class Solver:
    """Fast MST solver using Kruskal's algorithm with union‑find."""

    class DSU:
        __slots__ = ("parent", "rank")

        def __init__(self, n: int):
            self.parent = list(range(n))
            self.rank = [0] * n

        def find(self, x: int) -> int:
            parent = self.parent
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(self, x: int, y: int) -> bool:
            xr, yr = self.find(x), self.find(y)
            if xr == yr:
                return False
            rank = self.rank
            if rank[xr] < rank[yr]:
                self.parent[xr] = yr
            elif rank[xr] > rank[yr]:
                self.parent[yr] = xr
            else:
                self.parent[yr] = xr
                rank[xr] += 1
            return True

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, List[List[Any]]]:
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]

        # Sort edges by weight
        edges_sorted = sorted(edges, key=lambda e: e[2])

        dsu = self.DSU(num_nodes)
        mst_edges: List[List[Any]] = []

        for u, v, w in edges_sorted:
            if dsu.union(u, v):
                if u > v:
                    u, v = v, u
                mst_edges.append([u, v, w])
                if len(mst_edges) == num_nodes - 1:
                    break

        # Sort by (u, v) for deterministic output
        mst_edges.sort(key=lambda e: (e[0], e[1]))
        return {"mst_edges": mst_edges}
