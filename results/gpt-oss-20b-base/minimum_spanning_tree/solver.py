# solver.py
from __future__ import annotations
from typing import Any, List

# ----------------------------------------------------------------------
# Union‑Find (Disjoint Set) data structure
# ----------------------------------------------------------------------
class _UF:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        # Path compression (iterative)
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


# ----------------------------------------------------------------------
# Solver class
# ----------------------------------------------------------------------
class Solver:
    """
    Compute the Minimum Spanning Tree for an undirected weighted graph
    provided in `problem`.  The graph is given as a list of edges
    ``[(u, v, w), ...]`` where ``u`` and ``v`` are node indices and
    ``w`` is the weight.  The result is a list of edges in the MST,
    sorted lexicographically by their (smaller, larger) node pair.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, List[List[float]]]:
        num_nodes: int = problem["num_nodes"]
        edges: List[tuple[int, int, float]] = problem["edges"]

        # Sort edges by weight (ascending) – required for Kruskal's algorithm.
        edges_sorted = sorted(edges, key=lambda e: e[2])

        uf = _UF(num_nodes)
        mst_edges: List[List[float]] = []
        for u, v, w in edges_sorted:
            if uf.union(u, v):
                if u > v:
                    u, v = v, u
                mst_edges.append([u, v, w])
                if len(mst_edges) == num_nodes - 1:
                    break

        # Final lexicographic sort of the MST edges
        mst_edges.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": mst_edges}